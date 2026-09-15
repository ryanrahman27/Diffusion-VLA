"""Inter-gripper relative EE pose helpers shared by joint and EE PiperX policies."""

import math
from typing import Sequence

import numpy as np

PIPERX_EE_DIM = 20
PIPERX_REL_POSE_DIM = 9
PIPERX_EE_STATE_WITH_REL_DIM = PIPERX_EE_DIM + PIPERX_REL_POSE_DIM

# Right-arm base origin expressed in the left-arm base frame (Piper SDK: X fwd, Y left).
# Parallel mounts, 50 cm center-to-center along lateral axis.
PIPERX_RIGHT_BASE_IN_LEFT_BASE = np.array([0.0, -0.5, 0.0], dtype=np.float64)

LEFT_POSE_SLICE = slice(0, 9)
RIGHT_POSE_SLICE = slice(10, 19)


def rpy_to_rot6d(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """ZYX Euler (rad) -> Zhou 6D, matching ``episode_recorder.rpy_to_rot6d``."""
    r, p, y = (float(v) for v in rpy[:3])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            cy * cp,
            sy * cp,
            -sp,
            cy * sp * sr - sy * cr,
            sy * sp * sr + cy * cr,
            cp * sr,
        ],
        dtype=np.float32,
    )


def rot6d_to_rotation_matrix(rot6d: Sequence[float] | np.ndarray) -> np.ndarray:
    """Recover 3x3 rotation from 6D rep (Gram-Schmidt on the two stored columns)."""
    r00, r10, r20, r01, r11, r21 = (float(v) for v in rot6d)
    c1 = np.array([r00, r10, r20], dtype=np.float64)
    c2 = np.array([r01, r11, r21], dtype=np.float64)
    n1 = float(np.linalg.norm(c1))
    if n1 < 1e-8:
        return np.eye(3)
    c1 /= n1
    c2 = c2 - float(np.dot(c2, c1)) * c1
    n2 = float(np.linalg.norm(c2))
    if n2 < 1e-8:
        return np.eye(3)
    c2 /= n2
    c3 = np.cross(c1, c2)
    return np.column_stack([c1, c2, c3])


def rotation_matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """First two columns of ``rotation``, flattened (Zhou et al. convention).

    Layout matches ``episode_recorder.rpy_to_rot6d``: concatenate column vectors
    ``[r00,r10,r20,r01,r11,r21]``, not row-major ``reshape`` of the 3x2 block.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def pose6d_to_se3(xyz: Sequence[float] | np.ndarray, rot6d: Sequence[float] | np.ndarray) -> np.ndarray:
    """Build 4x4 SE(3) for EE pose in arm base frame (point in EE -> point in base)."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot6d_to_rotation_matrix(rot6d)
    transform[:3, 3] = np.asarray(xyz[:3], dtype=np.float64)
    return transform


def se3_inverse(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def base_right_to_base_left(
    right_base_in_left_base: Sequence[float] | np.ndarray = PIPERX_RIGHT_BASE_IN_LEFT_BASE,
) -> np.ndarray:
    """Rigid offset from right-arm base frame to left-arm base frame (parallel mounts)."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(right_base_in_left_base[:3], dtype=np.float64)
    return transform


def relative_pose_9d_from_transform(transform: np.ndarray) -> np.ndarray:
    """Encode SE(3) as 9-D [xyz(3), rot6d(6)] — same component order as per-arm EE state."""
    transform = np.asarray(transform, dtype=np.float64)
    translation = transform[:3, 3].astype(np.float32)
    rot6d = rotation_matrix_to_rot6d(transform[:3, :3])
    return np.concatenate([translation, rot6d], axis=-1).astype(np.float32)


def rotation_angle_rad(rotation: np.ndarray) -> float:
    """Geodesic angle between ``rotation`` and identity."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(math.acos(cosine))


def verify_rot6d_roundtrip(
    rpy: Sequence[float] | np.ndarray,
    *,
    angle_tol_rad: float = 1e-4,
) -> tuple[bool, float]:
    """Check Gram-Schmidt recovery is a valid rotation and idempotent."""
    rot6d = rpy_to_rot6d(rpy)
    recovered = rot6d_to_rotation_matrix(rot6d)
    reencoded = rotation_matrix_to_rot6d(recovered)
    re_recovered = rot6d_to_rotation_matrix(reencoded)

    orthonormal_err = float(np.linalg.norm(recovered.T @ recovered - np.eye(3)))
    det_err = abs(float(np.linalg.det(recovered)) - 1.0)
    col0_err = float(np.linalg.norm(recovered[:, 0] - rot6d[:3]))
    angle_err = rotation_angle_rad(recovered.T @ re_recovered)
    ok = (
        orthonormal_err <= angle_tol_rad
        and det_err <= angle_tol_rad
        and col0_err <= angle_tol_rad
        and angle_err <= angle_tol_rad
    )
    return ok, max(angle_err, orthonormal_err, det_err, col0_err)


def implied_base_right_in_left(
    left_xyz: Sequence[float],
    left_rot6d: Sequence[float],
    right_xyz: Sequence[float],
    right_rot6d: Sequence[float],
) -> np.ndarray:
    """Estimate right-base-in-left-base when both EEs share the same world pose.

    With coincident gripper frames, ``T_L @ inv(T_R)`` equals the fixed base offset.
    """
    left_transform = pose6d_to_se3(left_xyz, left_rot6d)
    right_transform = pose6d_to_se3(right_xyz, right_rot6d)
    return left_transform @ se3_inverse(right_transform)


def compute_inter_gripper_rel(
    eef_6d: np.ndarray,
    *,
    right_base_in_left_base: Sequence[float] | np.ndarray = PIPERX_RIGHT_BASE_IN_LEFT_BASE,
) -> np.ndarray:
    """Left-gripper-to-right-gripper relative pose as 9-D [xyz(3), rot6d(6)]."""
    eef_6d = np.asarray(eef_6d, dtype=np.float64)
    if eef_6d.shape[-1] not in (PIPERX_EE_DIM, PIPERX_EE_STATE_WITH_REL_DIM):
        raise ValueError(
            f"Expected eef_6d dim {PIPERX_EE_DIM} or {PIPERX_EE_STATE_WITH_REL_DIM}, got {eef_6d.shape[-1]}"
        )

    left = eef_6d[..., LEFT_POSE_SLICE]
    right = eef_6d[..., RIGHT_POSE_SLICE]
    left_transform = pose6d_to_se3(left[..., :3], left[..., 3:9])
    right_transform = pose6d_to_se3(right[..., :3], right[..., 3:9])
    base_transform = base_right_to_base_left(right_base_in_left_base)

    if left_transform.ndim == 2:
        right_in_left = base_transform @ right_transform
        relative = se3_inverse(left_transform) @ right_in_left
        return relative_pose_9d_from_transform(relative)

    batch = np.empty(eef_6d.shape[:-1] + (PIPERX_REL_POSE_DIM,), dtype=np.float32)
    for index in np.ndindex(eef_6d.shape[:-1]):
        left_transform = pose6d_to_se3(left[index, :3], left[index, 3:9])
        right_transform = pose6d_to_se3(right[index, :3], right[index, 3:9])
        right_in_left = base_transform @ right_transform
        relative = se3_inverse(left_transform) @ right_in_left
        batch[index] = relative_pose_9d_from_transform(relative)
    return batch


def append_inter_gripper_rel_to_ee_state(
    state: np.ndarray,
    *,
    right_base_in_left_base: Sequence[float] | np.ndarray = PIPERX_RIGHT_BASE_IN_LEFT_BASE,
) -> np.ndarray:
    """Return 29-D EE state with inter-gripper relative pose appended."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == PIPERX_EE_STATE_WITH_REL_DIM:
        return state
    if state.shape[-1] != PIPERX_EE_DIM:
        raise ValueError(
            f"Expected EE state dim {PIPERX_EE_DIM} or {PIPERX_EE_STATE_WITH_REL_DIM}, got {state.shape[-1]}"
        )
    relative = compute_inter_gripper_rel(state, right_base_in_left_base=right_base_in_left_base)
    return np.concatenate([state, relative], axis=-1).astype(np.float32)
