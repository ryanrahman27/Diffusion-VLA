"""UMI PD2.2-style relative-to-current EE proprio for PiperX bimanual policies.

Keeps the current EE pose absolute (20-D) and appends one past timestep per arm as
``inv(T_current) @ T_past`` encoded as [xyz(3), rot6d(6)] with absolute gripper width.
Total state dimension: 40-D.
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms
from openpi.policies.piperx_rel_pose import PIPERX_EE_DIM
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.policies.piperx_rel_pose import se3_inverse

PIPERX_EE_STATE_WITH_REL_PROPRIO_DIM = 40
_RELPROPRIO_ENCODED_KEY = "__relproprio_encoded__"

_LEFT_POSE_SLICE = slice(0, 9)
_RIGHT_POSE_SLICE = slice(10, 19)
_ARM_SLICES = (_LEFT_POSE_SLICE, _RIGHT_POSE_SLICE)
_PAST_LEFT_GRIP_IDX = 29
_PAST_RIGHT_GRIP_IDX = 39

GRIPPER_INDICES_WITH_REL_PROPRIO = (9, 19, _PAST_LEFT_GRIP_IDX, _PAST_RIGHT_GRIP_IDX)


def _encode_arm_relative_to_current(
    current_pose_9: np.ndarray,
    past_pose_9: np.ndarray,
) -> np.ndarray:
    transform_current = pose6d_to_se3(current_pose_9[:3], current_pose_9[3:9])
    transform_past = pose6d_to_se3(past_pose_9[:3], past_pose_9[3:9])
    transform_rel = se3_inverse(transform_current) @ transform_past
    return relative_pose_9d_from_transform(transform_rel)


def _resolve_current_and_past(
    data: dict,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(data["state"], dtype=np.float32)

    if state.ndim == 2:
        if state.shape[0] < 2 or state.shape[-1] != PIPERX_EE_DIM:
            raise ValueError(
                f"Expected state history shape (K, {PIPERX_EE_DIM}) with K>=2, got {state.shape}"
            )
        return state[-1], state[0]

    if state.shape[-1] != PIPERX_EE_DIM:
        raise ValueError(f"Expected current state dim {PIPERX_EE_DIM}, got {state.shape}")

    if "state_history" in data:
        past = np.asarray(data["state_history"], dtype=np.float32)
        if past.shape[-1] != PIPERX_EE_DIM:
            raise ValueError(
                f"Expected state_history dim {PIPERX_EE_DIM}, got {past.shape}"
            )
        return state, past

    # First inference tick / missing history: identity-relative fallback.
    return state, state


def build_state_with_relative_proprio(current_20: np.ndarray, past_20: np.ndarray) -> np.ndarray:
    """Build 40-D state: [current abs 20 | past-rel 20]."""
    current_20 = np.asarray(current_20, dtype=np.float32).reshape(PIPERX_EE_DIM)
    past_20 = np.asarray(past_20, dtype=np.float32).reshape(PIPERX_EE_DIM)

    out = np.empty(PIPERX_EE_STATE_WITH_REL_PROPRIO_DIM, dtype=np.float32)
    out[:PIPERX_EE_DIM] = current_20

    past_block = np.zeros(PIPERX_EE_DIM, dtype=np.float32)
    for pose_slice in _ARM_SLICES:
        past_block[pose_slice] = _encode_arm_relative_to_current(
            current_20[pose_slice],
            past_20[pose_slice],
        )
    past_block[9] = past_20[9]
    past_block[19] = past_20[19]
    out[PIPERX_EE_DIM:] = past_block
    return out


@dataclasses.dataclass(frozen=True)
class RelativeToCurrentEeProprio(_transforms.DataTransformFn):
    """Append past EE pose relative to current (PD2.2); current frame stays absolute."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if data.get(_RELPROPRIO_ENCODED_KEY):
            return data

        current, past = _resolve_current_and_past(data)
        data = dict(data)
        data["state"] = build_state_with_relative_proprio(current, past)
        data[_RELPROPRIO_ENCODED_KEY] = True
        data.pop("state_history", None)
        return data
