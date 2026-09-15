"""UMI-style relative-to-current SE(3) action encoding for single-arm Piper EE policies.

Each action in a chunk is expressed as ``inv(T_ee_current) @ T_ee_target`` (9-D pose),
anchored to the observation-time absolute EE pose. Grippers stay absolute.

Absolute anchor is read from ``state_absolute`` when proprio has already been UMI-encoded.
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms
from openpi.policies.piper_ee_single_policy import PIPER_EE_DIM
from openpi.policies.piper_ee_single_rel_proprio import _STATE_ABSOLUTE_KEY
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.policies.piperx_rel_pose import se3_inverse

_POSE_SLICE = slice(0, 9)
_T0_ENCODED_KEY = "__t0_encoded__"


def _encode_arm_relative_to_current(state_pose_9: np.ndarray, action_pose_9: np.ndarray) -> np.ndarray:
    transform_current = pose6d_to_se3(state_pose_9[:3], state_pose_9[3:9])
    transform_target = pose6d_to_se3(action_pose_9[:3], action_pose_9[3:9])
    transform_rel = se3_inverse(transform_current) @ transform_target
    return relative_pose_9d_from_transform(transform_rel)


def _decode_arm_relative_to_current(state_pose_9: np.ndarray, rel_pose_9: np.ndarray) -> np.ndarray:
    transform_current = pose6d_to_se3(state_pose_9[:3], state_pose_9[3:9])
    transform_rel = pose6d_to_se3(rel_pose_9[:3], rel_pose_9[3:9])
    transform_abs = transform_current @ transform_rel
    return relative_pose_9d_from_transform(transform_abs)


def _resolve_absolute_state_10(data: dict) -> np.ndarray:
    if _STATE_ABSOLUTE_KEY in data:
        return np.asarray(data[_STATE_ABSOLUTE_KEY], dtype=np.float32).reshape(PIPER_EE_DIM)

    state = np.asarray(data["state"], dtype=np.float32)
    if state.ndim == 2:
        return state[-1].reshape(PIPER_EE_DIM)
    if state.shape[-1] == PIPER_EE_DIM:
        return state.reshape(PIPER_EE_DIM)
    raise ValueError(
        f"Cannot resolve absolute EE anchor from state shape {state.shape}; "
        f"expected {_STATE_ABSOLUTE_KEY} or {PIPER_EE_DIM}-D absolute state."
    )


def _encode_single_timestep(state_10: np.ndarray, action_10: np.ndarray) -> np.ndarray:
    encoded = np.asarray(action_10, dtype=np.float32).copy()
    state_10 = np.asarray(state_10, dtype=np.float32)
    encoded[_POSE_SLICE] = _encode_arm_relative_to_current(state_10[_POSE_SLICE], action_10[_POSE_SLICE])
    return encoded


def _decode_single_timestep(state_10: np.ndarray, encoded_10: np.ndarray) -> np.ndarray:
    decoded = np.asarray(encoded_10, dtype=np.float32).copy()
    state_10 = np.asarray(state_10, dtype=np.float32)
    decoded[_POSE_SLICE] = _decode_arm_relative_to_current(state_10[_POSE_SLICE], encoded_10[_POSE_SLICE])
    return decoded


def encode_ee_actions_relative_to_current(data: dict, actions: np.ndarray) -> np.ndarray:
    """Encode absolute EE actions relative to the current observation pose."""
    state = _resolve_absolute_state_10(data)
    actions = np.asarray(actions, dtype=np.float32)
    encoded = np.empty_like(actions)

    if actions.ndim == 1:
        return _encode_single_timestep(state, actions)

    leading_shape = actions.shape[:-2]
    horizon = actions.shape[-2]

    for idx in np.ndindex(leading_shape):
        state_10 = state if state.ndim == 1 else state[idx]
        for h in range(horizon):
            encoded[idx + (h,)] = _encode_single_timestep(state_10, actions[idx + (h,)])
    return encoded


def decode_ee_actions_relative_to_current(data: dict, actions: np.ndarray) -> np.ndarray:
    """Decode relative-to-current EE actions back to absolute poses."""
    state = _resolve_absolute_state_10(data)
    actions = np.asarray(actions, dtype=np.float32)
    decoded = np.empty_like(actions)

    if actions.ndim == 1:
        return _decode_single_timestep(state, actions)

    leading_shape = actions.shape[:-2]
    horizon = actions.shape[-2]

    for idx in np.ndindex(leading_shape):
        state_10 = state if state.ndim == 1 else state[idx]
        for h in range(horizon):
            decoded[idx + (h,)] = _decode_single_timestep(state_10, actions[idx + (h,)])
    return decoded


# Backward-compatible aliases (same anchor: observation-time current EE).
encode_ee_actions_relative_to_t0 = encode_ee_actions_relative_to_current
decode_ee_actions_relative_to_t0 = decode_ee_actions_relative_to_current


@dataclasses.dataclass(frozen=True)
class RelativeToT0EeActions(_transforms.DataTransformFn):
    """Training input: absolute EE chunk -> relative-to-current SE(3) (gripper absolute)."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data or data.get(_T0_ENCODED_KEY):
            return data
        data = dict(data)
        data["actions"] = encode_ee_actions_relative_to_current(data, data["actions"])
        data[_T0_ENCODED_KEY] = True
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteFromT0EeActions(_transforms.DataTransformFn):
    """Inference output: relative-to-current SE(3) chunk -> absolute EE poses."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data
        data = dict(data)
        data["actions"] = decode_ee_actions_relative_to_current(data, data["actions"])
        return data
