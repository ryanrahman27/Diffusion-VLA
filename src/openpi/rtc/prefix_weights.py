"""Prefix attention weights for Real-Time Chunking."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import torch

from openpi.rtc import config as _rtc_config


def get_prefix_weights_np(
    start: int,
    end: int,
    total: int,
    schedule: _rtc_config.RTCAttentionSchedule,
) -> np.ndarray:
    start = min(start, end)
    if schedule == _rtc_config.RTCAttentionSchedule.ZEROS:
        weights = np.zeros(total, dtype=np.float32)
        weights[:start] = 1.0
    elif schedule == _rtc_config.RTCAttentionSchedule.ONES:
        weights = np.ones(total, dtype=np.float32)
        weights[end:] = 0.0
    elif schedule in (_rtc_config.RTCAttentionSchedule.LINEAR, _rtc_config.RTCAttentionSchedule.EXP):
        lin_weights = _linweights_np(start, end, total)
        if schedule == _rtc_config.RTCAttentionSchedule.EXP:
            lin_weights = lin_weights * np.expm1(lin_weights) / (math.e - 1)
        weights = _add_trailing_zeros_np(lin_weights, total, end)
        weights = _add_leading_ones_np(weights, start, total)
    else:
        raise ValueError(f"Unknown prefix attention schedule: {schedule}")
    return weights.astype(np.float32)


def get_prefix_weights_jax(
    start: int,
    end: int,
    total: int,
    schedule: _rtc_config.RTCAttentionSchedule,
) -> jnp.ndarray:
    return jnp.asarray(get_prefix_weights_np(start, end, total, schedule))


def get_prefix_weights_torch(
    start: int,
    end: int,
    total: int,
    schedule: _rtc_config.RTCAttentionSchedule,
) -> torch.Tensor:
    return torch.from_numpy(get_prefix_weights_np(start, end, total, schedule))


def _linweights_np(start: int, end: int, total: int) -> np.ndarray:
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return np.array([], dtype=np.float32)
    return np.linspace(1, 0, linspace_steps + 2, dtype=np.float32)[1:-1]


def _add_trailing_zeros_np(weights: np.ndarray, total: int, end: int) -> np.ndarray:
    zeros_len = total - end
    if zeros_len <= 0:
        return weights
    return np.concatenate([weights, np.zeros(zeros_len, dtype=np.float32)])


def _add_leading_ones_np(weights: np.ndarray, start: int, total: int) -> np.ndarray:
    ones_len = min(start, total)
    if ones_len <= 0:
        return weights
    return np.concatenate([np.ones(ones_len, dtype=np.float32), weights])
