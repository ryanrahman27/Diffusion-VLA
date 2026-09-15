import numpy as np

from openpi.policies.piperx_ee_rel_proprio import RelativeToCurrentEeProprio
from openpi.policies.piperx_ee_rel_proprio import build_state_with_relative_proprio
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.training import config as config_module


def _identity_rot6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def test_pure_x_translation_past_behind_current():
    current = np.zeros(20, dtype=np.float32)
    current[:3] = [0.10, 0.0, 0.0]
    current[3:9] = _identity_rot6d()
    current[9] = 0.05
    current[10:13] = [0.0, 0.0, 0.30]
    current[13:19] = _identity_rot6d()
    current[19] = 0.05

    past = current.copy()
    past[:3] = [0.05, 0.0, 0.0]

    encoded = build_state_with_relative_proprio(current, past)
    rel_left_xyz = encoded[20:23]
    np.testing.assert_allclose(rel_left_xyz, [-0.05, 0.0, 0.0], atol=1e-4)

    # Past grippers stay absolute.
    assert encoded[29] == past[9]
    assert encoded[39] == past[19]
    # Current block unchanged.
    np.testing.assert_allclose(encoded[:20], current)


def test_forward_matches_hand_computed_se3():
    current = np.zeros(20, dtype=np.float32)
    current[:3] = [0.03, -0.01, 0.29]
    current[3:9] = [0.40, -0.05, 0.91, -0.36, -0.93, 0.11]

    past = current.copy()
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = [0.02, 0.0, 0.0]
    past[:9] = relative_pose_9d_from_transform(pose6d_to_se3(current[:3], current[3:9]) @ offset)

    encoded = build_state_with_relative_proprio(current, past)
    hand = np.zeros(9, dtype=np.float32)
    hand[:] = relative_pose_9d_from_transform(
        np.linalg.inv(pose6d_to_se3(current[:3], current[3:9]))
        @ pose6d_to_se3(past[:3], past[3:9])
    )
    np.testing.assert_allclose(encoded[20:29], hand, atol=1e-4)


def test_transform_from_state_history_stack():
    current = np.linspace(0, 1, 20, dtype=np.float32)
    past = current - 0.1
    item = {"state": np.stack([past, current], axis=0)}
    out = RelativeToCurrentEeProprio()(item)
    expected = build_state_with_relative_proprio(current, past)
    np.testing.assert_allclose(out["state"], expected, atol=1e-4)
    assert out["__relproprio_encoded__"] is True


def test_rel_traj_combined_config_wiring():
    cfg = config_module.get_config("pi05_piperx_laundry_ee_rel_traj")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    output_names = [type(t).__name__ for t in data_cfg.data_transforms.outputs]
    assert "RelativeToCurrentEeProprio" in names
    assert "RelativeToT0EeActions" in names
    assert "AbsoluteFromT0EeActions" in output_names
    assert "DeltaActions" not in names
    assert data_cfg.state_sequence_keys == ("observation.state",)
    assert data_cfg.proprio_history_steps == 1


def test_pd21_only_has_t0_not_proprio():
    cfg = config_module.get_config("pi05_piperx_laundry_ee_t0")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    assert "RelativeToT0EeActions" in names
    assert "RelativeToCurrentEeProprio" not in names
    assert data_cfg.state_sequence_keys == ()


def test_baseline_unchanged():
    data_cfg = config_module.get_config("pi05_piperx_laundry_ee").data.create(
        config_module.get_config("pi05_piperx_laundry_ee").assets_dirs,
        config_module.get_config("pi05_piperx_laundry_ee").model,
    )
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    assert "DeltaActions" in names
    assert "RelativeToCurrentEeProprio" not in names
    assert "RelativeToT0EeActions" not in names
