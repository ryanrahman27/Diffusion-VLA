import numpy as np

from openpi.policies.piper_ee_single_rel_proprio import IDENTITY_POSE_9D
from openpi.policies.piper_ee_single_rel_proprio import RelativeToCurrentEeProprio
from openpi.policies.piper_ee_single_rel_proprio import build_umi_relative_state
from openpi.training import config as config_module


def _identity_rot6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def test_pure_x_translation_past_behind_current():
    current = np.zeros(10, dtype=np.float32)
    current[:3] = [0.10, 0.0, 0.0]
    current[3:9] = _identity_rot6d()
    current[9] = 0.05

    past = current.copy()
    past[:3] = [0.05, 0.0, 0.0]

    encoded = build_umi_relative_state(current, past)
    np.testing.assert_allclose(encoded[:9], IDENTITY_POSE_9D, atol=1e-4)
    assert encoded[9] == current[9]
    np.testing.assert_allclose(encoded[10:13], [-0.05, 0.0, 0.0], atol=1e-4)
    assert encoded[19] == past[9]


def test_transform_from_state_history_stack():
    current = np.linspace(0, 1, 10, dtype=np.float32)
    past = current - 0.1
    item = {"state": np.stack([past, current], axis=0)}
    out = RelativeToCurrentEeProprio()(item)
    expected = build_umi_relative_state(current, past)
    np.testing.assert_allclose(out["state"], expected, atol=1e-4)
    np.testing.assert_allclose(out["state_absolute"], current, atol=1e-4)


def test_rel_traj_combined_config_wiring():
    cfg = config_module.get_config("pi05_piper_umi_ee_rel_traj")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    output_names = [type(t).__name__ for t in data_cfg.data_transforms.outputs]
    assert names.index("RelativeToT0EeActions") < names.index("RelativeToCurrentEeProprio")
    assert "RelativeToCurrentEeProprio" in names
    assert "RelativeToT0EeActions" in names
    assert "AbsoluteFromT0EeActions" in output_names
    assert "DeltaActions" not in names
    assert data_cfg.state_sequence_keys == ("observation.state",)
    assert data_cfg.proprio_history_steps == 1
    assert data_cfg.proprio_history_stride == cfg.model.action_horizon


def test_pd21_only_has_t0_not_proprio():
    cfg = config_module.get_config("pi05_piper_umi_ee_t0")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    assert "RelativeToT0EeActions" in names
    assert "RelativeToCurrentEeProprio" not in names
    assert data_cfg.state_sequence_keys == ()
    assert data_cfg.proprio_history_stride is None


def test_baseline_unchanged():
    data_cfg = config_module.get_config("pi05_piper_umi_ee").data.create(
        config_module.get_config("pi05_piper_umi_ee").assets_dirs,
        config_module.get_config("pi05_piper_umi_ee").model,
    )
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    assert "DeltaActions" in names
    assert "RelativeToCurrentEeProprio" not in names
    assert "RelativeToT0EeActions" not in names
