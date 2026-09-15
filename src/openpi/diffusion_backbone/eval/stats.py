"""Statistics for the LIBERO comparison (proposal Sec. 9.1, "Statistical power").

Comparisons are never reported as single success numbers. Success rates get
Wilson confidence intervals; continuous metrics get bootstrap intervals;
significance is assessed with paired tests over the shared, deterministic
initial-condition set (each variant sees the same resets, so pairing is valid).

The headline contrast is C - D (matched). C - A is reported as a confounded
upper bound. Watch for the three outcomes the proposal calls out: positive,
null, and the probe/rollout dissociation.
"""

from __future__ import annotations

from typing import Any


def wilson_interval(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a success rate."""
    raise NotImplementedError


def bootstrap_interval(values: Any, *, alpha: float = 0.05, n_boot: int = 10_000) -> tuple[float, float]:
    """Bootstrap CI for a continuous metric."""
    raise NotImplementedError


def paired_test(metric_a: Any, metric_b: Any) -> Any:
    """Paired test over matched initial conditions (e.g. C vs D on the same resets)."""
    raise NotImplementedError


def contrast(records: dict[str, Any], left: str, right: str) -> Any:
    """Compute a named contrast (e.g. contrast(records, 'C', 'D')) with CIs + p-values."""
    raise NotImplementedError
