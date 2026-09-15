"""UMI-style relative EE proprio for single-arm Piper policies.

History: current frame + one past frame spaced by the action execution interval
(typically ``action_horizon`` frames at dataset fps, not a single timestep).

Model state (20-D) — no absolute EE pose in the input:
    [current_pose_identity(9), current_gripper(1),
     past_pose_relative_to_current(9), past_gripper(1)]

Current pose is always identity: translation [0, 0, 0], rot6d [1, 0, 0, 0, 1, 0].
Past pose is ``inv(T_current) @ T_past`` as [xyz(3), rot6d(6)].
Grippers are absolute widths (meters before gripper normalization).
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms
from openpi.policies.piper_ee_single_policy import PIPER_EE_DIM
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.policies.piperx_rel_pose import se3_inverse

PIPER_UMI_STATE_DIM = 20
PIPER_EE_STATE_WITH_REL_PROPRIO_DIM = PIPER_UMI_STATE_DIM
_RELPROPRIO_ENCODED_KEY = "__relproprio_encoded__"
_STATE_ABSOLUTE_KEY = "state_absolute"

IDENTITY_POSE_9D = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

CURRENT_GRIP_IDX = 9
_PAST_POSE_SLICE = slice(10, 19)
_PAST_GRIP_IDX = 19

GRIPPER_INDICES_WITH_REL_PROPRIO = (CURRENT_GRIP_IDX, _PAST_GRIP_IDX)


def _encode_pose_relative_to_current(
    current_pose_9: np.ndarray,
    past_pose_9: np.ndarray,
) -> np.ndarray:
    transform_current = pose6d_to_se3(current_pose_9[:3], current_pose_9[3:9])
    transform_past = pose6d_to_se3(past_pose_9[:3], past_pose_9[3:9])
    transform_rel = se3_inverse(transform_current) @ transform_past
    return relative_pose_9d_from_transform(transform_rel)


def _resolve_current_and_past(data: dict) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(data["state"], dtype=np.float32)

    if state.ndim == 2:
        if state.shape[0] < 2 or state.shape[-1] != PIPER_EE_DIM:
            raise ValueError(
                f"Expected state history shape (K, {PIPER_EE_DIM}) with K>=2, got {state.shape}"
            )
        return state[-1], state[0]

    if state.shape[-1] != PIPER_EE_DIM:
        raise ValueError(f"Expected current state dim {PIPER_EE_DIM}, got {state.shape}")

    if "state_history" in data:
        past = np.asarray(data["state_history"], dtype=np.float32)
        if past.shape[-1] != PIPER_EE_DIM:
            raise ValueError(f"Expected state_history dim {PIPER_EE_DIM}, got {past.shape}")
        return state, past

    return state, state


def build_umi_relative_state(current_10: np.ndarray, past_10: np.ndarray) -> np.ndarray:
    """Build 20-D UMI proprio: identity current pose + relative past pose + grippers."""
    current_10 = np.asarray(current_10, dtype=np.float32).reshape(PIPER_EE_DIM)
    past_10 = np.asarray(past_10, dtype=np.float32).reshape(PIPER_EE_DIM)

    out = np.empty(PIPER_UMI_STATE_DIM, dtype=np.float32)
    out[:9] = IDENTITY_POSE_9D
    out[CURRENT_GRIP_IDX] = current_10[9]
    out[_PAST_POSE_SLICE] = _encode_pose_relative_to_current(current_10[:9], past_10[:9])
    out[_PAST_GRIP_IDX] = past_10[9]
    return out


def build_state_with_relative_proprio(current_10: np.ndarray, past_10: np.ndarray) -> np.ndarray:
    """Alias for ``build_umi_relative_state`` (kept for tests and callers)."""
    return build_umi_relative_state(current_10, past_10)


@dataclasses.dataclass(frozen=True)
class RelativeToCurrentEeProprio(_transforms.DataTransformFn):
    """Encode proprio in UMI relative frame; stores absolute anchor in ``state_absolute``."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if data.get(_RELPROPRIO_ENCODED_KEY):
            return data

        current, past = _resolve_current_and_past(data)
        data = dict(data)
        data[_STATE_ABSOLUTE_KEY] = np.asarray(current, dtype=np.float32)
        data["state"] = build_umi_relative_state(current, past)
        data[_RELPROPRIO_ENCODED_KEY] = True
        data.pop("state_history", None)
        return data
