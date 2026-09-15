"""Wandb helpers for π₀.₅ + depth training (RGB/depth share ``observation.images``)."""

import numpy as np
import wandb

from openpi.depth.constants import DEPTH_KEYS
from openpi.depth.constants import RGB_KEYS
from openpi.models import model as _model


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a single frame to HWC uint8 for wandb.Image."""
    img = np.asarray(frame)
    if img.shape[-1] == 2:
        raise ValueError("expected RGB (C=3), got depth (C=2)")
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0 and img.min() >= -1.1:
            img = (img.astype(np.float32) + 1.0) * 127.5
        else:
            img = img.astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def log_rgb_camera_views(observation: _model.Observation, *, max_images: int = 5) -> None:
    """Log a horizontal strip of RGB views only (depth slots are 2-channel)."""
    images = observation.images
    keys = [k for k in RGB_KEYS if k in images]
    if not keys:
        keys = [k for k in images if k not in DEPTH_KEYS and images[k].shape[-1] == 3]
    if not keys:
        return

    batch_len = len(next(iter(images.values())))
    n = min(max_images, batch_len)
    strips = []
    for i in range(n):
        strips.append(
            np.concatenate([_to_uint8_rgb(np.asarray(images[k][i])) for k in keys], axis=1)
        )
    wandb.log({"camera_views": [wandb.Image(s) for s in strips]}, step=0)


def log_depth_meters_preview(observation: _model.Observation, *, max_images: int = 3) -> None:
    """Optional: log normalized depth (ch0) as grayscale for debugging."""
    images = observation.images
    keys = [k for k in DEPTH_KEYS if k in images]
    if not keys:
        return
    n = min(max_images, len(next(iter(images.values()))))
    previews = []
    for i in range(n):
        row = []
        for k in keys:
            d = np.asarray(images[k][i])
            if d.ndim == 3 and d.shape[0] == 2:
                d = np.transpose(d, (1, 2, 0))
            ch0 = d[..., 0]
            vis = ((ch0.astype(np.float32) + 1.0) * 127.5).astype(np.uint8)
            row.append(np.stack([vis, vis, vis], axis=-1))
        previews.append(np.concatenate(row, axis=1))
    wandb.log({"depth_norm_preview": [wandb.Image(p) for p in previews]}, step=0)
