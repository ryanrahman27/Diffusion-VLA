"""Input/output transforms for the PiperX bimanual robot.

Conceptually mirrors `aloha_policy.py`, with these differences:

  * 3 cameras (cam_front, cam_left_wrist, cam_right_wrist) instead of ALOHA's 4.
    Each maps cleanly onto pi0's image slots: base_0_rgb, left_wrist_0_rgb,
    right_wrist_0_rgb. No padding or masking needed for any camera.

  * Gripper is recorded as linear position in METERS (0.0 = closed, ~0.07 = open
    on the PiperX puppet) -- not normalized 0-1 like ALOHA. We normalize to 0-1
    here in the transform.

  * No "angular" gripper conversion. We do NOT try to match pi0's internal
    angular gripper convention -- that would require reverse-engineering
    PiperX's gripper kinematics. Instead we normalize to 0-1 and let
    fine-tuning adapt the model to the new gripper distribution.

  * Joint flip mask starts as all-ones (no flips). PiperX URDF sign conventions
    vs pi-internal conventions are unknown until we observe real-robot
    behavior. Update _joint_flip_mask() empirically based on Phase G testing.

State / action layout (14-dim, both):
    [L_arm_6 (rad), L_grip (m), R_arm_6 (rad), R_grip (m)]

With ``append_inter_gripper_rel=True``, state is extended to 23-D by appending a
9-D left-gripper-to-right-gripper relative EE pose [xyz(3), rot6d(6)]. Requires
``eef_6d`` (20-D) in the input dict alongside joint ``state``; actions stay 14-D joint.

After PiperX-specific decoding:
    [L_arm_6 (rad, optionally sign-flipped), L_grip (0-1),
     R_arm_6 (rad, optionally sign-flipped), R_grip (0-1)]
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies.piperx_rel_pose import PIPERX_REL_POSE_DIM
from openpi.policies.piperx_rel_pose import PIPERX_RIGHT_BASE_IN_LEFT_BASE
from openpi.policies.piperx_rel_pose import compute_inter_gripper_rel


# Gripper open/close calibration in meters, as recorded by the Piper teleop
# stack. Derived from min/max across episodes:
#   inspected episode #3 had L_grip min=+0.0013, max=+0.0685.
#   action range went 0.0 .. 0.0688.
# We round outward slightly for safety so normalization stays inside [0, 1]
# for typical values without clipping edge cases hard.
PIPERX_GRIPPER_CLOSED_M = 0.0
PIPERX_GRIPPER_OPEN_M = 0.07

PIPERX_JOINT_DIM = 14
PIPERX_JOINT_STATE_WITH_REL_DIM = PIPERX_JOINT_DIM + PIPERX_REL_POSE_DIM


def make_piperx_example() -> dict:
    """A fake input example matching what env.py would produce at inference.

    Useful for unit tests and to sanity-check the transform pipeline without
    needing a real robot or dataset attached. Mirrors `make_aloha_example()`.
    """
    return {
        "state": np.ones((14,), dtype=np.float32) * 0.5,
        "images": {
            "cam_front":       np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "fold towel",
    }


def _joint_flip_mask() -> np.ndarray:
    """Per-dim sign flip applied to joints (not grippers).

    PLACEHOLDER: starts as all ones (no flips). After Phase G smoke deployment,
    if specific joints move the wrong way, flip their signs here.

    Indices: [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]
    """
    return np.array([1, 1, 1, 1, 1, 1, 1,    # left arm + gripper (gripper sign is irrelevant; always 1)
                     1, 1, 1, 1, 1, 1, 1])   # right arm + gripper


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_model(value):
    """Convert recorded gripper (meters) -> normalized 0..1 for the model.

    0.0 = fully closed, 1.0 = fully open. Values slightly outside this range
    can occur near limits; we don't clip so the model can still react to them.
    """
    return _normalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _gripper_from_model(value):
    """Convert model output (0..1) -> raw meters for the follower."""
    return _unnormalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Convert from on-disk / on-wire convention to model-input convention."""
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        # Gripper at indices 6 (left) and 13 (right).
        state[..., [6, 13]] = _gripper_to_model(state[..., [6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Convert from model-output convention to on-wire convention (for robot)."""
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[..., [6, 13]] = _gripper_from_model(actions[..., [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Inverse of _encode_actions, applied to training-data actions.

    The training-data action goes from on-disk units (radians + meters) into
    the same convention the model is supposed to predict. This is symmetric
    to _decode_state and is required so that the model's training target lives
    in the same numerical space as its output predictions.
    """
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[..., [6, 13]] = _gripper_to_model(actions[..., [6, 13]])
    return actions


_WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")
_ALL_CAMERAS = ("cam_front", *_WRIST_CAMERAS)

_SLOT_TO_CAMERA: dict[str, str] = {
    "base_0_rgb": "cam_front",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}


def _reference_image(in_images: dict[str, np.ndarray]) -> np.ndarray:
    for cam in (*_WRIST_CAMERAS, "cam_front"):
        if cam in in_images:
            return in_images[cam]
    raise ValueError(
        f"Need at least one camera image. Expected subset of {_ALL_CAMERAS}, got {tuple(in_images)}."
    )


def build_piperx_image_inputs(
    in_images: dict[str, np.ndarray],
    *,
    use_front_camera: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.bool_]]:
    """Map PiperX camera dict to pi0 image slots + masks."""
    ref = _reference_image(in_images)

    if use_front_camera:
        front_cam = _SLOT_TO_CAMERA["base_0_rgb"]
        if front_cam not in in_images:
            raise ValueError(f"Missing required camera '{front_cam}' (-> base_0_rgb).")
        images: dict[str, np.ndarray] = {"base_0_rgb": in_images[front_cam]}
        image_masks: dict[str, np.bool_] = {"base_0_rgb": np.True_}
    else:
        if not any(cam in in_images for cam in _WRIST_CAMERAS):
            raise ValueError(
                f"Wrist-only mode requires at least one of {_WRIST_CAMERAS}, got {tuple(in_images)}."
            )
        images = {"base_0_rgb": np.zeros_like(ref)}
        image_masks = {"base_0_rgb": np.False_}

    for slot in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        cam = _SLOT_TO_CAMERA[slot]
        if cam in in_images:
            images[slot] = in_images[cam]
            image_masks[slot] = np.True_
        else:
            images[slot] = np.zeros_like(ref)
            image_masks[slot] = np.False_

    return images, image_masks


def _decode_piperx(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """Apply state/image decoding shared between training and inference."""
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # If floats in [0, 1] (LeRobot returns CHW float32 in [0,1] when
        # loading video), promote to uint8 first.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # CHW -> HWC. The PaliGemma vision tower expects HWC; openpi's
        # wire format is CHW; we transpose here at the boundary.
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def append_inter_gripper_rel_to_joint_state(
    joint_state: np.ndarray,
    eef_6d: np.ndarray,
    *,
    right_base_in_left_base: tuple[float, float, float] | np.ndarray = tuple(
        PIPERX_RIGHT_BASE_IN_LEFT_BASE.tolist()
    ),
) -> np.ndarray:
    """Append 9-D inter-gripper relative EE pose to a 14-D joint state vector."""
    joint_state = np.asarray(joint_state, dtype=np.float32)
    if joint_state.shape[-1] == PIPERX_JOINT_STATE_WITH_REL_DIM:
        return joint_state
    if joint_state.shape[-1] != PIPERX_JOINT_DIM:
        raise ValueError(
            f"Expected joint state dim {PIPERX_JOINT_DIM} or {PIPERX_JOINT_STATE_WITH_REL_DIM}, "
            f"got {joint_state.shape[-1]}"
        )
    relative = compute_inter_gripper_rel(
        eef_6d,
        right_base_in_left_base=right_base_in_left_base,
    )
    return np.concatenate([joint_state, relative], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    """Inputs for the PiperX bimanual policy.

    Expected inputs (post-repack):
      images: dict[str, ndarray]  -- each [C, H, W] uint8 (CHW); the names
                                     must be a subset of EXPECTED_CAMERAS.
      state:  [14]                -- joints + grippers; raw radians + meters
      actions: [action_horizon, 14] (training only) -- same units as state.
      prompt: str (optional)
    """

    # If true, perform PiperX -> model-convention conversion (joint flip mask
    # if set, gripper meters -> 0-1). Set True for both training and inference
    # so the convention is consistent. We keep the parameter explicit for
    # symmetry with AlohaInputs and easy A/B testing later.
    adapt_to_pi: bool = True
    # If False, only wrist cameras are used; base_0_rgb is zero-padded and masked off.
    use_front_camera: bool = True
    # Append 9-D inter-gripper relative EE pose to joint proprio (14-D -> 23-D state).
    append_inter_gripper_rel: bool = False
    right_base_in_left_base: tuple[float, float, float] = tuple(PIPERX_RIGHT_BASE_IN_LEFT_BASE.tolist())
    # Norm-stats only: process state/actions without loading camera frames.
    skip_images: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = _ALL_CAMERAS

    def __call__(self, data: dict) -> dict:
        if self.skip_images:
            state = _decode_state(np.asarray(data["state"]), adapt_to_pi=self.adapt_to_pi)
            if self.append_inter_gripper_rel:
                if state.shape[-1] != PIPERX_JOINT_STATE_WITH_REL_DIM:
                    if "eef_6d" not in data:
                        raise ValueError(
                            "append_inter_gripper_rel requires 'eef_6d' (20-D) in the input data."
                        )
                    state = append_inter_gripper_rel_to_joint_state(
                        state,
                        np.asarray(data["eef_6d"]),
                        right_base_in_left_base=self.right_base_in_left_base,
                    )
            inputs: dict = {"state": state}
            if "actions" in data:
                actions = np.asarray(data["actions"])
                inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return inputs

        data = _decode_piperx(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(
                f"Unexpected camera(s) in input: {unknown}. "
                f"Expected subset of {self.EXPECTED_CAMERAS}."
            )

        images, image_masks = build_piperx_image_inputs(
            in_images, use_front_camera=self.use_front_camera
        )

        state = data["state"]
        if self.append_inter_gripper_rel:
            if state.shape[-1] != PIPERX_JOINT_STATE_WITH_REL_DIM:
                if "eef_6d" not in data:
                    raise ValueError(
                        "append_inter_gripper_rel requires 'eef_6d' (20-D) in the input data."
                    )
                state = append_inter_gripper_rel_to_joint_state(
                    state,
                    np.asarray(data["eef_6d"]),
                    right_base_in_left_base=self.right_base_in_left_base,
                )

        inputs = {
            "image":      images,
            "image_mask": image_masks,
            "state":      state,
        }

        # Actions are only present during training. Convert them into the
        # same convention as state (joint flip + gripper normalize) so the
        # delta later computed is in a sensible space.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    """Outputs for the PiperX bimanual policy.

    The model emits a (action_horizon, padded_action_dim) chunk. We:
      * slice to the real 14 dims,
      * invert joint flip + gripper normalization to convert back to robot units.
    """

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Slice to the real 14 dims. pi0.5's internal action vector is
        # padded to 32 (cross-embodiment); only the first 14 are PiperX.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}