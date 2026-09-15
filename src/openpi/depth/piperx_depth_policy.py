"""PiperX transforms that load metric depth into π₀ depth slots."""

import dataclasses
from typing import ClassVar

import numpy as np
from PIL import Image

from openpi.depth.constants import DEFAULT_DEPTH_MAX_M
from openpi.depth.constants import DEFAULT_DEPTH_MIN_M
from openpi.depth.constants import DEFAULT_DEPTH_RANGE_M_PER_CAMERA
from openpi.depth.constants import DEFAULT_DEPTH_SCALE_M
from openpi.depth.constants import DEPTH_INPUT_CHANNELS
from openpi.depth.constants import PIPERX_CAMERA_TO_SLOTS
from openpi.policies import piperx_policy
from openpi import transforms


def _load_depth_array(depth: np.ndarray) -> np.ndarray:
    """Load depth to float32 (H, W). Accepts uint16 Z16, float, or PIL-backed arrays."""
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            raise ValueError(f"expected depth with 1 channel, got shape {depth.shape}")
    if depth.ndim != 2:
        raise ValueError(f"expected depth (H, W), got shape {depth.shape}")
    if depth.dtype == np.uint16:
        return depth.astype(np.float32)
    return depth.astype(np.float32)


def resolve_depth_range_m(
    camera: str,
    depth_range_per_camera: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Look up (min_m, max_m) for a PiperX camera name."""
    if camera in depth_range_per_camera:
        depth_min_m, depth_max_m = depth_range_per_camera[camera]
    else:
        depth_min_m, depth_max_m = DEFAULT_DEPTH_MIN_M, DEFAULT_DEPTH_MAX_M
    if depth_max_m <= depth_min_m:
        raise ValueError(f"{camera}: depth_max_m ({depth_max_m}) must be > depth_min_m ({depth_min_m})")
    return depth_min_m, depth_max_m


def normalize_depth_to_channels(
    depth_mm_or_raw: np.ndarray,
    *,
    depth_scale_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Build HWC depth input: channel 0 = meters in [-1, 1], channel 1 = valid mask."""
    depth_m = depth_mm_or_raw * depth_scale_m
    valid = depth_mm_or_raw > 0
    denom = max(depth_max_m - depth_min_m, 1e-6)
    norm = np.clip((depth_m - depth_min_m) / denom, 0.0, 1.0) * 2.0 - 1.0
    norm = np.where(valid, norm, 0.0).astype(np.float32)
    mask = valid.astype(np.float32)
    if DEPTH_INPUT_CHANNELS != 2:
        raise ValueError(f"expected DEPTH_INPUT_CHANNELS=2, got {DEPTH_INPUT_CHANNELS}")
    return np.stack([norm, mask], axis=-1)


def decode_depth_png_if_needed(depth: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(depth, Image.Image):
        return np.asarray(depth)
    return _load_depth_array(depth)


@dataclasses.dataclass(frozen=True)
class PiperXDepthInputs(piperx_policy.PiperXInputs):
    """Extends PiperX RGB mapping with parallel depth slots in the ``image`` dict."""

    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M
    depth_range_per_camera: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_DEPTH_RANGE_M_PER_CAMERA)
    )

    EXPECTED_DEPTH_CAMERAS: ClassVar[tuple[str, ...]] = tuple(PIPERX_CAMERA_TO_SLOTS)

    def __call__(self, data: dict) -> dict:
        # Run PiperX RGB + state logic (writes image / image_mask / state / actions / prompt).
        out = super().__call__(data)

        if "depth" not in data:
            raise ValueError(
                "PiperXDepthInputs expects a 'depth' dict in the batch. "
                "Use LeRobotPiperXDepthDataConfig repack transforms."
            )

        in_depth = data["depth"]
        image = dict(out["image"])
        image_mask = dict(out["image_mask"])

        for cam, (_rgb_slot, depth_slot) in PIPERX_CAMERA_TO_SLOTS.items():
            if cam not in in_depth:
                continue
            if not self.use_front_camera and cam == "cam_front":
                continue

            depth_min_m, depth_max_m = resolve_depth_range_m(cam, self.depth_range_per_camera)
            raw = decode_depth_png_if_needed(in_depth[cam])
            depth_hwc = normalize_depth_to_channels(
                raw,
                depth_scale_m=self.depth_scale_m,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            # HWC: [normalized_meters, valid_mask]; same layout as RGB for ResizeImages.
            image[depth_slot] = depth_hwc
            image_mask[depth_slot] = np.bool_(np.any(depth_hwc[..., 1] > 0))

        out["image"] = image
        out["image_mask"] = image_mask
        return out


def make_piperx_depth_example() -> dict:
    """Fake observation for tests (RGB + depth CHW under camera names)."""
    ex = piperx_policy.make_piperx_example()
    depth = {
        cam: np.random.randint(500, 2000, size=(1, 224, 224), dtype=np.uint16)
        for cam in ("cam_front", "cam_left_wrist", "cam_right_wrist")
    }
    ex["depth"] = depth
    return ex
