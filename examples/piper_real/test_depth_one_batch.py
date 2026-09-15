"""Smoke test: π₀.₅+depth repack → transforms → model input spec (no GPU required)."""

import numpy as np

from openpi.depth.constants import DEFAULT_DEPTH_RANGE_M_PER_CAMERA
from openpi.depth.constants import DEPTH_INPUT_CHANNELS
from openpi.depth.piperx_depth_policy import PiperXDepthInputs
from openpi.depth.piperx_depth_policy import normalize_depth_to_channels
from openpi.depth.piperx_depth_policy import resolve_depth_range_m
from openpi.depth.pi0_depth_config import Pi0DepthConfig
import openpi.transforms as transforms


def test_normalize_depth_to_channels():
    raw = np.array([[0, 500, 1500], [0, 0, 2000]], dtype=np.float32)
    hwc = normalize_depth_to_channels(raw, depth_scale_m=0.001, depth_min_m=0.15, depth_max_m=1.55)
    assert hwc.shape == (2, 3, DEPTH_INPUT_CHANNELS)
    assert hwc[0, 0, 1] == 0.0  # invalid hole
    assert hwc[0, 1, 1] == 1.0  # 0.5 m valid
    assert hwc[0, 0, 0] == 0.0  # norm ignored where invalid


def test_per_camera_ranges():
    """Same meter value maps to different norm depending on camera range."""
    raw = np.array([[1000]], dtype=np.float32)  # 1.0 m
    _, front_max = resolve_depth_range_m("cam_front", DEFAULT_DEPTH_RANGE_M_PER_CAMERA)
    front_min, _ = resolve_depth_range_m("cam_front", DEFAULT_DEPTH_RANGE_M_PER_CAMERA)
    wrist_min, wrist_max = resolve_depth_range_m("cam_left_wrist", DEFAULT_DEPTH_RANGE_M_PER_CAMERA)

    front_hwc = normalize_depth_to_channels(
        raw, depth_scale_m=0.001, depth_min_m=front_min, depth_max_m=front_max
    )
    wrist_hwc = normalize_depth_to_channels(
        raw, depth_scale_m=0.001, depth_min_m=wrist_min, depth_max_m=wrist_max
    )
    # 1.0 m is mid-range for front (~0.45–1.95) but high for wrist (~0.15–1.55).
    assert front_hwc[0, 0, 0] < wrist_hwc[0, 0, 0]


def main():
    test_normalize_depth_to_channels()
    test_per_camera_ranges()

    cfg = Pi0DepthConfig()
    assert cfg.pi05 is True
    assert cfg.discrete_state_input is True
    assert cfg.depth_range_per_camera["cam_front"] == (0.45, 1.95)

    raw = {
        "images": {
            "cam_front": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        },
        "depth": {
            "cam_front": np.random.randint(400, 1900, (224, 224), dtype=np.uint16),
            "cam_left_wrist": np.random.randint(150, 1500, (224, 224), dtype=np.uint16),
            "cam_right_wrist": np.random.randint(150, 1500, (224, 224), dtype=np.uint16),
        },
        "state": np.zeros((14,), dtype=np.float32),
        "prompt": "flatten towel",
    }

    out = PiperXDepthInputs()(raw)
    assert "base_0_rgb" in out["image"]
    assert "base_0_depth" in out["image"]
    assert out["image"]["base_0_depth"].shape[-1] == DEPTH_INPUT_CHANNELS

    resized = transforms.ResizeImages(224, 224)(out)
    assert resized["image"]["base_0_depth"].shape == (224, 224, DEPTH_INPUT_CHANNELS)

    obs_spec, _ = cfg.inputs_spec(batch_size=2)
    print("π₀.₅+depth OK — per-camera ranges:", cfg.depth_range_per_camera)
    print("  image keys:", sorted(obs_spec.images.keys()))


if __name__ == "__main__":
    main()
