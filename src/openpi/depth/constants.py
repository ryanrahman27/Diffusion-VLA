"""Image slot names for RGB + depth token streams."""

from openpi.models import model as _model

# Depth tensor layout: HWC with [normalized_meters, valid_mask] per pixel.
DEPTH_INPUT_CHANNELS: int = 2

# SigLIP So400m/14 internal width (RGB stem); depth encoder runs here then projects to Gemma width.
SIGLIP_SO400M_WIDTH: int = 1152

# Z16 mm -> meters (RealSense default in PiperX HDF5).
DEFAULT_DEPTH_SCALE_M: float = 0.001

# Per-camera metric clip ranges (meters), from laundry HDF5 p1/p99 + margin.
# Front sees farther scene; wrists see close manipulation volume.
DEFAULT_DEPTH_RANGE_M_PER_CAMERA: dict[str, tuple[float, float]] = {
    "cam_front": (0.45, 1.95),
    "cam_left_wrist": (0.15, 1.55),
    "cam_right_wrist": (0.15, 1.55),
}

# Legacy single range (fallback if a camera is missing from the dict).
DEFAULT_DEPTH_MIN_M: float = 0.15
DEFAULT_DEPTH_MAX_M: float = 2.0

# Standard π₀ RGB slots (same as openpi.models.model.IMAGE_KEYS).
RGB_KEYS: tuple[str, ...] = _model.IMAGE_KEYS

# Parallel depth slots appended after each RGB block in the prefix.
DEPTH_KEYS: tuple[str, ...] = (
    "base_0_depth",
    "left_wrist_0_depth",
    "right_wrist_0_depth",
)

# PiperX LeRobot camera name -> (rgb_slot, depth_slot).
PIPERX_CAMERA_TO_SLOTS: dict[str, tuple[str, str]] = {
    "cam_front": ("base_0_rgb", "base_0_depth"),
    "cam_left_wrist": ("left_wrist_0_rgb", "left_wrist_0_depth"),
    "cam_right_wrist": ("right_wrist_0_rgb", "right_wrist_0_depth"),
}
