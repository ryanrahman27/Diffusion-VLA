"""UMI PD2.1-style relative-to-t0 SE(3) action encoding for PiperX EE policies.

Each action in a chunk is expressed as ``inv(T_ee_at_t0) @ T_ee_target`` per arm,
anchored to the observation-time EE pose (chunk start). Grippers stay absolute.

Mirrors ``DeltaActions`` / ``AbsoluteActions`` but uses proper SE(3) composition
instead of element-wise subtraction on xyz + rot6d.
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.policies.piperx_rel_pose import se3_inverse

_LEFT_POSE_SLICE = slice(0, 9)
_RIGHT_POSE_SLICE = slice(10, 19)
_ARM_SLICES = (_LEFT_POSE_SLICE, _RIGHT_POSE_SLICE)
_T0_ENCODED_KEY = "__t0_encoded__"


def _encode_arm_relative_to_t0(state_pose_9: np.ndarray, action_pose_9: np.ndarray) -> np.ndarray:
    transform_t0 = pose6d_to_se3(state_pose_9[:3], state_pose_9[3:9])
    transform_target = pose6d_to_se3(action_pose_9[:3], action_pose_9[3:9])
    transform_rel = se3_inverse(transform_t0) @ transform_target
    return relative_pose_9d_from_transform(transform_rel)


def _decode_arm_relative_to_t0(state_pose_9: np.ndarray, rel_pose_9: np.ndarray) -> np.ndarray:
    transform_t0 = pose6d_to_se3(state_pose_9[:3], state_pose_9[3:9])
    transform_rel = pose6d_to_se3(rel_pose_9[:3], rel_pose_9[3:9])
    transform_abs = transform_t0 @ transform_rel
    return relative_pose_9d_from_transform(transform_abs)


def _encode_single_timestep(state_20: np.ndarray, action_20: np.ndarray) -> np.ndarray:
    encoded = np.asarray(action_20, dtype=np.float32).copy()
    state_20 = np.asarray(state_20, dtype=np.float32)
    for pose_slice in _ARM_SLICES:
        encoded[pose_slice] = _encode_arm_relative_to_t0(state_20[pose_slice], action_20[pose_slice])
    return encoded


def _decode_single_timestep(state_20: np.ndarray, encoded_20: np.ndarray) -> np.ndarray:
    decoded = np.asarray(encoded_20, dtype=np.float32).copy()
    state_20 = np.asarray(state_20, dtype=np.float32)
    for pose_slice in _ARM_SLICES:
        decoded[pose_slice] = _decode_arm_relative_to_t0(state_20[pose_slice], encoded_20[pose_slice])
    return decoded


def _anchor_state_20(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] > 20:
        state = state[..., :20]
    return state


def encode_ee_actions_relative_to_t0(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Encode absolute EE actions relative to ``state`` (20-D) for all chunk steps."""
    state = _anchor_state_20(state)
    actions = np.asarray(actions, dtype=np.float32)
    encoded = np.empty_like(actions)

    if actions.ndim == 1:
        return _encode_single_timestep(state, actions)

    leading_shape = actions.shape[:-2]
    horizon = actions.shape[-2]
    state_leading = state.shape[:-1]

    if state_leading != leading_shape:
        state = np.broadcast_to(state, (*leading_shape, state.shape[-1]))

    for idx in np.ndindex(leading_shape):
        state_20 = state[idx]
        for h in range(horizon):
            encoded[idx + (h,)] = _encode_single_timestep(state_20, actions[idx + (h,)])
    return encoded


def decode_ee_actions_relative_to_t0(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Decode relative-to-t0 EE actions back to absolute poses using ``state`` as anchor."""
    state = _anchor_state_20(state)
    actions = np.asarray(actions, dtype=np.float32)
    decoded = np.empty_like(actions)

    if actions.ndim == 1:
        return _decode_single_timestep(state, actions)

    leading_shape = actions.shape[:-2]
    horizon = actions.shape[-2]
    state_leading = state.shape[:-1]

    if state_leading != leading_shape:
        state = np.broadcast_to(state, (*leading_shape, state.shape[-1]))

    for idx in np.ndindex(leading_shape):
        state_20 = state[idx]
        for h in range(horizon):
            decoded[idx + (h,)] = _decode_single_timestep(state_20, actions[idx + (h,)])
    return decoded


@dataclasses.dataclass(frozen=True)
class RelativeToT0EeActions(_transforms.DataTransformFn):
    """Training input: absolute EE chunk -> relative-to-t0 SE(3) per arm (grippers absolute)."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data or data.get(_T0_ENCODED_KEY):
            return data
        data = dict(data)
        data["actions"] = encode_ee_actions_relative_to_t0(data["state"], data["actions"])
        data[_T0_ENCODED_KEY] = True
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteFromT0EeActions(_transforms.DataTransformFn):
    """Inference output: relative-to-t0 SE(3) chunk -> absolute EE poses (grippers unchanged)."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data
        data = dict(data)
        data["actions"] = decode_ee_actions_relative_to_t0(data["state"], data["actions"])
        return data
