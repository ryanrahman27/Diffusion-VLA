"""Extract frozen action-token features from a trained C or D checkpoint (paper Sec 9.2).

Runs the trained backbone over LIBERO observations at flow time tau=1 and dumps the
action-token hidden states (mean-pooled + first-token) plus the labels we probe
against (action chunk, immediate action, proprioceptive state). C and D use the same
config seed, so with shuffle the two dumps see the SAME frames in the SAME order --
run_probes.py verifies that alignment before comparing.

Usage (on the GPU box, main venv):
    uv run scripts/extract_probe_features.py \
        --config db_C_seed0_lora \
        --ckpt checkpoints/db_C_seed0_lora/db_C_seed0_v2/30000 \
        --out probe_feats/c_seed0.npz --num-samples 4000
    uv run scripts/extract_probe_features.py \
        --config db_D_seed0_lora \
        --ckpt checkpoints/db_D_seed0_lora/db_D_seed0_v2/30000 \
        --out probe_feats/d_seed0.npz --num-samples 4000
"""

import argparse
import logging
import os

import jax
import numpy as np
import safetensors.torch
import torch

import openpi.models.pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.diffusion_backbone import model_registry as _db_registry


def _build_model(config, device):
    """Mirror scripts/train_pytorch.py's model build so keys line up with the ckpt."""
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    return _db_registry.build_pytorch_model(config.name, model_cfg).to(device)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="train config name, e.g. db_C_seed0_lora")
    p.add_argument("--ckpt", required=True, help="checkpoint dir containing model.safetensors")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--num-samples", type=int, default=4000, help="#observations to dump")
    # Default: NO shuffle. C and D then iterate the dataset in the same order and
    # see identical frames (run_probes verifies this), which is what makes C-D a
    # matched contrast. Adjacent frames are temporally correlated, so absolute R^2
    # is mildly optimistic -- but that inflation is common to C and D and cancels
    # in C-D. Pass --shuffle only if you have a decorrelated held-out split.
    p.add_argument("--shuffle", action="store_true", default=False, help="shuffle frames (breaks C/D frame alignment)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config.get_config(args.config)
    logging.info(f"config={args.config} seed={config.seed} action_horizon={config.model.action_horizon}")

    model = _build_model(config, device)
    ckpt_path = os.path.join(args.ckpt, "model.safetensors")
    missing, unexpected = safetensors.torch.load_model(model, ckpt_path, strict=False)
    logging.info(f"[load] {len(missing)} missing, {len(unexpected)} unexpected from {ckpt_path}")
    if missing:
        # Trained ckpt should restore everything; a large/backbone miss means a real problem.
        logging.info(f"[load] sample missing: {sorted(missing)[:10]}")
    model.eval()

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=args.shuffle)

    feat_mean, feat_first, acts, states = [], [], [], []
    n = 0
    for observation, actions in loader:
        observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
        actions = actions.to(device)  # noqa: PLW2901
        h = model.action_token_features(observation).float()  # [B, H, d]
        feat_mean.append(h.mean(dim=1).cpu().numpy())  # [B, d]
        feat_first.append(h[:, 0].cpu().numpy())  # [B, d]
        acts.append(actions.float().cpu().numpy())  # [B, H, A]
        states.append(observation.state.float().cpu().numpy())  # [B, s]
        n += h.shape[0]
        if n % 512 < h.shape[0]:
            logging.info(f"  extracted {n}/{args.num_samples}")
        if n >= args.num_samples:
            break

    out = {
        "feat_mean": np.concatenate(feat_mean)[: args.num_samples],
        "feat_first": np.concatenate(feat_first)[: args.num_samples],
        "actions": np.concatenate(acts)[: args.num_samples],
        "state": np.concatenate(states)[: args.num_samples],
        "config": args.config,
        "ckpt": args.ckpt,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    logging.info(
        f"saved {out['feat_mean'].shape[0]} samples -> {args.out} "
        f"(feat dim {out['feat_mean'].shape[1]}, action {out['actions'].shape[1:]} , state {out['state'].shape[1:]})"
    )


if __name__ == "__main__":
    main()
