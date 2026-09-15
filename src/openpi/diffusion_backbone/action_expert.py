"""Standalone flow-matching action expert (decoupled from the backbone).

Why standalone: pi0.5's native action expert is *fused* into the backbone's
self-attention (it is a second transformer stream that shares each layer's
attention, RoPE, and KV cache, and therefore requires identical layer count /
head geometry / adaRMS norms; see models_pytorch/gemma_pytorch.py). That fusion
cannot be kept identical while the backbone changes from Gemma to Qwen2.5/Dream.

So for the controlled study we use a standalone flow expert, shared *identically*
across variants A'/B/C/D: it reads the backbone's action-token hidden states
`h_action` and outputs the flow-matching vector field, decoupled from the
backbone's attention. The ONLY variant-specific parameter is the input adapter
(backbone width -> common expert width); the expert stack itself is identical,
which keeps the C-D contrast clean.

Architecture: a small DiT-style transformer over the H action tokens.
Bidirectional self-attention (action chunk is one coherent block), with the
flow-time conditioning injected via adaLN-Zero (zero-initialized modulation, as
in DiT), so at init the expert is near-identity and training is stable.

    h_action [B, H, d_backbone]  --input adapter-->  [B, H, d_model]
        + learned positional embedding over H
        -> N DiT blocks conditioned on flow-time s
        -> out_proj -> v_pred [B, H, action_dim]

Flow-matching (matching openpi's pi0.5 convention):
    x_t = t * noise + (1 - t) * actions,   u_t = noise - actions,
    loss = MSE(v_pred, u_t).
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass
class ExpertConfig:
    input_dim: int = 2048      # backbone hidden width (Gemma 2048; Qwen/Dream 3584)
    d_model: int = 1024        # shared expert width (identical across A'/B/C/D)
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    action_dim: int = 32       # openpi pads LIBERO's 7-DoF to model_action_dim=32
    max_horizon: int = 32      # >= action_horizon (LIBERO uses 10)
    time_embed_dim: int = 256

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.num_heads == 0, "d_model must divide num_heads"
        return self.d_model // self.num_heads


def sinusoidal_time_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of flow-time s in [0,1]. s: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half
    )
    args = s.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DiTBlock(nn.Module):
    """DiT block: adaLN-Zero modulated bidirectional attention + MLP."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.num_heads
        self.num_heads, self.head_dim = h, cfg.head_dim
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        # adaLN-Zero: 6 modulation params (shift/scale/gate for attn and mlp).
        self.modulation = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, H, D = x.shape
        qkv = self.qkv(x).reshape(B, H, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, H, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)    # bidirectional (no mask)
        out = out.transpose(1, 2).reshape(B, H, D)
        return self.proj(out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=-1)
        # broadcast [B, D] -> [B, 1, D] over the H action tokens
        x = x + gate1[:, None] * self._attn(self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None])
        x = x + gate2[:, None] * self.mlp(self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None])
        return x


class StandaloneActionExpert(nn.Module):
    """Shared flow-matching action expert. Consumes backbone h_action + flow-time."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.input_adapter = nn.Linear(cfg.input_dim, cfg.d_model)  # only variant-specific part
        self.pos_emb = nn.Parameter(torch.zeros(cfg.max_horizon, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.blocks = nn.ModuleList(DiTBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.action_dim)
        nn.init.zeros_(self.out_proj.weight)  # start near zero vector field (stable)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h_action: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """h_action: [B, H, input_dim], s: [B] in [0,1]  ->  v_pred: [B, H, action_dim]."""
        B, H, _ = h_action.shape
        if H > self.cfg.max_horizon:
            raise ValueError(f"horizon {H} exceeds max_horizon {self.cfg.max_horizon}")
        x = self.input_adapter(h_action) + self.pos_emb[:H][None]
        cond = self.time_mlp(sinusoidal_time_embedding(s, self.cfg.time_embed_dim))
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(self.final_norm(x))


def flow_matching_loss(
    v_pred: torch.Tensor, noise: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """MSE(v_pred, u_t) with u_t = noise - actions (openpi pi0.5 convention)."""
    return F.mse_loss(v_pred, noise - actions)
