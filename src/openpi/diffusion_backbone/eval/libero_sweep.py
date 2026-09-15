"""Run the LIBERO evaluation sweep.

Wraps openpi's existing LIBERO harness (examples/libero/main.py,
openpi.policies.libero_policy) to evaluate a trained policy across the four
suites at the full standard protocol (500 rollouts/suite), plus the low-data
(few-shot) conditions. Reports LIBERO-Long as the primary discriminator.
"""

from __future__ import annotations

from typing import Any

from openpi.diffusion_backbone import constants


def evaluate_policy(
    policy: Any,
    suites: tuple[str, ...] = constants.LIBERO_SUITES,
    rollouts_per_task: int = constants.ROLLOUTS_PER_TASK,
) -> dict[str, Any]:
    """Evaluate one policy over the given suites; return per-suite rollout records.

    Records success, final pose/placement error, contact-timing error, jerk, and
    inference latency (at the policy's k) per rollout, over the shared
    deterministic initial-condition set so comparisons are paired.
    """
    raise NotImplementedError


def sweep(variants: tuple[str, ...], seeds: tuple[int, ...] = constants.SEEDS) -> dict[str, Any]:
    """Evaluate all (variant, seed) policies. 4 variants x 3 seeds = 12 sweeps."""
    raise NotImplementedError
