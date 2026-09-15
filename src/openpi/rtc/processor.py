"""Real-Time Chunking (RTC) prefix guidance for flow-matching policies.

Ported from LeRobot's RTCProcessor, adapted for OpenPI's time convention (t=1 noise, t=0 data).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from openpi.rtc import config as _rtc_config


class RTCProcessor:
    """Applies prefix guidance during flow-matching denoising steps."""

    def __init__(self, rtc_config: _rtc_config.RTCConfig) -> None:
        self.rtc_config = rtc_config

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        time: float | Tensor,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        *,
        steps_executed: int | None = None,
    ) -> Tensor:
        """Wrap a denoising step with RTC prefix guidance."""
        if prev_chunk_left_over is None or not self.rtc_config.enabled:
            return original_denoise_step_partial(x_t)

        if isinstance(time, Tensor):
            time_value = float(time.item())
        else:
            time_value = float(time)

        x_t = x_t.clone().detach()

        squeezed = False
        if x_t.ndim < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        batch_size, action_chunk_size, action_dim = x_t.shape
        prefix_attention_end = self.rtc_config.prefix_attention_end(steps_executed, action_chunk_size)

        if (
            prev_chunk_left_over.shape[1] < action_chunk_size
            or prev_chunk_left_over.shape[2] < action_dim
        ):
            padded = torch.zeros(batch_size, action_chunk_size, action_dim, device=x_t.device, dtype=x_t.dtype)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        if prev_chunk_left_over.shape != x_t.shape:
            raise ValueError(
                f"prev_chunk_left_over shape {prev_chunk_left_over.shape} must match x_t shape {x_t.shape}"
            )

        weights = (
            self.get_prefix_weights(inference_delay, prefix_attention_end, action_chunk_size)
            .to(x_t.device, dtype=x_t.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t = x_t.detach().requires_grad_(True)
            x1_t = x_t - time_value * v_t
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

            tau = 1.0 - time_value
            max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight, device=x_t.device)
            tau_tensor = torch.as_tensor(tau, device=x_t.device, dtype=x_t.dtype)
            squared_one_minus_tau = (1 - tau_tensor) ** 2
            inv_r2 = (squared_one_minus_tau + tau_tensor**2) / squared_one_minus_tau
            c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
            guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
            guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

            result = v_t - guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)

        return result

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)

        schedule = self.rtc_config.prefix_attention_schedule
        if schedule == _rtc_config.RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif schedule == _rtc_config.RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif schedule == _rtc_config.RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif schedule == _rtc_config.RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        else:
            raise ValueError(f"Unknown prefix attention schedule: {schedule}")

        return weights

    def _linweights(self, start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    def _add_leading_ones(self, weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])
