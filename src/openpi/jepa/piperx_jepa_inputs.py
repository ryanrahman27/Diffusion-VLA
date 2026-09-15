"""PiperX transforms: current RGB + multi-horizon future RGB for JEPA."""

import dataclasses

import einops
import numpy as np

from openpi.jepa.constants import DEFAULT_JEPA_HORIZON_STEPS
from openpi.policies import piperx_policy
from openpi.policies.piperx_policy import build_piperx_image_inputs


def _convert_piperx_image_chw_to_hwc(img: np.ndarray) -> np.ndarray:
    """Match ``PiperXInputs`` / ``_decode_piperx`` (LeRobot CHW → model HWC)."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


def _split_temporal_stack(
    img: np.ndarray,
    horizon_steps: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray] | None]:
    """LeRobot temporal stack ``[1 + len(horizons), C, H, W]`` → current + futures."""
    arr = np.asarray(img)
    expected_len = 1 + len(horizon_steps)
    if arr.ndim != 4 or arr.shape[0] != expected_len:
        return arr, None
    current = arr[0]
    futures = {horizon: arr[idx + 1] for idx, horizon in enumerate(horizon_steps)}
    return current, futures


@dataclasses.dataclass(frozen=True)
class PiperXJepaInputs(piperx_policy.PiperXInputs):
    """Maps PiperX cameras to pi0 slots and ``*_future_<h>`` slots for each horizon."""

    jepa_horizon_steps: tuple[int, ...] = DEFAULT_JEPA_HORIZON_STEPS

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        current_images: dict[str, np.ndarray] = {}
        future_by_horizon: dict[int, dict[str, np.ndarray]] = {}

        for cam, img in in_images.items():
            now, futures = _split_temporal_stack(img, self.jepa_horizon_steps)
            current_images[cam] = now
            if futures is not None:
                for horizon, frame in futures.items():
                    future_by_horizon.setdefault(horizon, {})[cam] = frame

        out = super().__call__({**data, "images": current_images})

        for horizon, future_images in future_by_horizon.items():
            future_slots, future_masks = build_piperx_image_inputs(
                future_images,
                use_front_camera=self.use_front_camera,
            )
            for slot, arr in future_slots.items():
                out["image"][f"{slot}_future_{horizon}"] = _convert_piperx_image_chw_to_hwc(arr)
                out["image_mask"][f"{slot}_future_{horizon}"] = future_masks[slot]
        return out
