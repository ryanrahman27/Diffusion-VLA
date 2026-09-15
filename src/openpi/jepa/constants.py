"""Keys and defaults for π₀.₅ + multi-horizon JEPA prefix tokens."""

from openpi.models import model as _model

RGB_KEYS: tuple[str, ...] = _model.IMAGE_KEYS

# Horizons at dataset frame rate (~short/medium/long for fold motions at 10–30 Hz).
DEFAULT_JEPA_HORIZON_STEPS: tuple[int, ...] = (2, 5, 10)
DEFAULT_JEPA_QUERIES_PER_HORIZON: int = 4
DEFAULT_JEPA_LOSS_WEIGHT: float = 0.1


def future_rgb_key(rgb_key: str, horizon: int) -> str:
    return f"{rgb_key}_future_{horizon}"


def future_rgb_keys_for_horizons(horizons: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(future_rgb_key(k, h) for h in horizons for k in RGB_KEYS)
