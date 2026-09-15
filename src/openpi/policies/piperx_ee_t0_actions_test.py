import numpy as np

from openpi.policies.piperx_ee_policy import make_piperx_ee_example
from openpi.policies.piperx_ee_t0_actions import AbsoluteFromT0EeActions
from openpi.policies.piperx_ee_t0_actions import RelativeToT0EeActions
from openpi.policies.piperx_ee_t0_actions import decode_ee_actions_relative_to_t0
from openpi.policies.piperx_ee_t0_actions import encode_ee_actions_relative_to_t0
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.training import config as config_module


def _make_absolute_chunk(state: np.ndarray, horizon: int = 4) -> np.ndarray:
    """Build a synthetic absolute action chunk with valid SE(3) poses per arm."""
    actions = np.tile(state, (horizon, 1)).astype(np.float32)
    for h in range(horizon):
        for pose_slice in (slice(0, 9), slice(10, 19)):
            offset = np.eye(4, dtype=np.float64)
            offset[:3, 3] = [0.01 * (h + 1), 0.0, 0.0]
            base = pose6d_to_se3(state[pose_slice][:3], state[pose_slice][3:9])
            actions[h, pose_slice] = relative_pose_9d_from_transform(base @ offset)
    return actions


def test_roundtrip_single_timestep():
    state = make_piperx_ee_example()["state"]
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = [0.02, -0.01, 0.005]
    action = state.copy()
    for pose_slice in (slice(0, 9), slice(10, 19)):
        base = pose6d_to_se3(state[pose_slice][:3], state[pose_slice][3:9])
        action[pose_slice] = relative_pose_9d_from_transform(base @ offset)

    encoded = encode_ee_actions_relative_to_t0(state, action)
    decoded = decode_ee_actions_relative_to_t0(state, encoded)

    np.testing.assert_allclose(decoded, action, rtol=1e-5, atol=1e-4)
    assert encoded[9] == action[9]
    assert encoded[19] == action[19]


def test_roundtrip_action_chunk_same_anchor():
    state = make_piperx_ee_example()["state"]
    actions = _make_absolute_chunk(state, horizon=5)

    encoded = encode_ee_actions_relative_to_t0(state, actions)
    decoded = decode_ee_actions_relative_to_t0(state, encoded)

    np.testing.assert_allclose(decoded, actions, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(encoded[:, 9], actions[:, 9])
    np.testing.assert_allclose(encoded[:, 19], actions[:, 19])

    # Step 0 relative translation should match SE(3) composition, not vector difference.
    rel_t0 = encode_ee_actions_relative_to_t0(state, actions[0:1])[0, :3]
    naive_delta = actions[0, :3] - state[:3]
    assert not np.allclose(rel_t0, naive_delta, atol=1e-3)


def test_transform_classes_mirror_encode_decode():
    state = make_piperx_ee_example()["state"]
    actions = _make_absolute_chunk(state, horizon=3)

    item = {"state": state, "actions": actions.copy()}
    encoded_item = RelativeToT0EeActions()(item)
    decoded_item = AbsoluteFromT0EeActions()(encoded_item)

    np.testing.assert_allclose(encoded_item["actions"], encode_ee_actions_relative_to_t0(state, actions))
    np.testing.assert_allclose(decoded_item["actions"], actions, rtol=1e-5, atol=1e-4)


def test_laundry_ee_t0_config_wiring():
    cfg = config_module.get_config("pi05_piperx_laundry_ee_t0")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    transform_names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    output_names = [type(t).__name__ for t in data_cfg.data_transforms.outputs]

    assert "RelativeToT0EeActions" in transform_names
    assert "AbsoluteFromT0EeActions" in output_names
    assert "DeltaActions" not in transform_names


def test_baseline_laundry_ee_unchanged():
    cfg = config_module.get_config("pi05_piperx_laundry_ee")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    transform_names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]

    assert "DeltaActions" in transform_names
    assert "RelativeToT0EeActions" not in transform_names
