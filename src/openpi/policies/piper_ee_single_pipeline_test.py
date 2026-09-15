"""End-to-end checks: UMI rel_traj transforms produce no absolute EE pose in model inputs."""

import numpy as np

from openpi.policies.piper_ee_single_rel_proprio import IDENTITY_POSE_9D
from openpi.policies.piper_ee_single_rel_proprio import PIPER_UMI_STATE_DIM
from openpi.policies.piper_ee_single_t0_actions import decode_ee_actions_relative_to_current
from openpi.policies.piper_ee_single_t0_actions import encode_ee_actions_relative_to_current
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.training import config as config_module


def _make_absolute_state_and_actions():
    current = np.array(
        [0.03, -0.01, 0.29, 0.40, -0.05, 0.91, -0.36, -0.93, 0.11, 0.05],
        dtype=np.float32,
    )
    past = current.copy()
    past[:3] = current[:3] + np.array([-0.02, 0.0, 0.0], dtype=np.float32)
    future = current.copy()
    future[:3] = current[:3] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    actions = np.stack([future, future], axis=0)
    return past, current, actions


def test_rel_traj_config_wires_relative_only():
    cfg = config_module.get_config("pi05_piper_umi_ee_rel_traj")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_cfg.data_transforms.inputs]
    assert names.index("RelativeToT0EeActions") < names.index("RelativeToCurrentEeProprio")
    assert "RelativeToT0EeActions" in names
    assert "RelativeToCurrentEeProprio" in names
    assert "DeltaActions" not in names
    assert data_cfg.proprio_history_stride == cfg.model.action_horizon


def test_model_inputs_have_no_absolute_pose():
    from openpi.policies.piper_ee_single_policy import PiperEeSingleInputs
    from openpi.policies.piper_ee_single_rel_proprio import RelativeToCurrentEeProprio
    from openpi.policies.piper_ee_single_t0_actions import RelativeToT0EeActions

    past, current, actions = _make_absolute_state_and_actions()
    item = {
        "state": np.stack([past, current], axis=0),
        "actions": actions,
        "images": {"cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8)},
        "prompt": "pick cube and place in bin",
    }

    for transform in (
        RelativeToT0EeActions(),
        RelativeToCurrentEeProprio(),
        PiperEeSingleInputs(adapt_to_pi=False, gripper_indices=(9, 19)),
    ):
        item = transform(item)

    state = item["state"]
    assert state.shape == (PIPER_UMI_STATE_DIM,)
    np.testing.assert_allclose(state[:9], IDENTITY_POSE_9D, atol=1e-5)
    assert not np.allclose(state[10:13], current[:3], atol=1e-3)

    decoded = decode_ee_actions_relative_to_current(item, item["actions"])
    np.testing.assert_allclose(decoded[0, :3], actions[0, :3], atol=1e-4)


def test_action_pose_is_relative_to_current_not_absolute():
    past, current, actions = _make_absolute_state_and_actions()
    data = {"state": np.stack([past, current], axis=0)}
    rel = encode_ee_actions_relative_to_current(data, actions)
    assert not np.allclose(rel[0, :3], actions[0, :3], atol=1e-3)

    hand = relative_pose_9d_from_transform(
        np.linalg.inv(pose6d_to_se3(current[:3], current[3:9]))
        @ pose6d_to_se3(actions[0, :3], actions[0, 3:9])
    )
    np.testing.assert_allclose(rel[0, :9], hand, atol=1e-4)
