"""Input/output transforms for a bimanual humanoid upper body.

Mirrors ``piperx_policy.py`` but with 7 DoF per arm plus hand pinch distance:

State / action layout (16-dim) — matches HDF5 ``record_joint_names`` order:
    [L_arm_7 (rad), R_arm_7 (rad), L_pinch (m), R_pinch (m)]

Hand pinch is thumb-index distance in metres (larger = more open), from
``observations/hand_pinch/pinch_dist`` in v2 recordings. Only pinch distance
is used — not individual OmniHand motor joints.

Cameras use the same three-view layout as PiperX (front + both wrists).
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies.piperx_policy import build_piperx_image_inputs

# Pinch distance calibration (metres). v2 demos span roughly 0.01–0.13 m.
HAND_PINCH_CLOSED_M = 0.0
HAND_PINCH_OPEN_M = 0.14

# Pinch indices in the 16-dim vector (after both 7-DoF arms).
_LEFT_PINCH_IDX = 14
_RIGHT_PINCH_IDX = 15

_WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")
_ALL_CAMERAS = ("cam_front", *_WRIST_CAMERAS)


def make_humanoid_upper_body_example() -> dict:
    """Fake inference example matching the robot wire format."""
    return {
        "state": np.ones((16,), dtype=np.float32) * 0.5,
        "images": {
            "cam_front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick cube and place in bin",
    }


def _joint_flip_mask() -> np.ndarray:
    """Per-dim sign flip for joints (not grippers). Placeholder: all ones."""
    return np.ones((16,), dtype=np.float64)


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_model(value):
    return _normalize(value, min_val=HAND_PINCH_CLOSED_M, max_val=HAND_PINCH_OPEN_M)


def _gripper_from_model(value):
    return _unnormalize(value, min_val=HAND_PINCH_CLOSED_M, max_val=HAND_PINCH_OPEN_M)


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        state[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]] = _gripper_to_model(
            state[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]]
        )
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]] = _gripper_from_model(
            actions[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]]
        )
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]] = _gripper_to_model(
            actions[..., [_LEFT_PINCH_IDX, _RIGHT_PINCH_IDX]]
        )
    return actions


def _decode_humanoid(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    data["images"] = {name: convert_image(img) for name, img in images.items()}
    data["state"] = state
    return data


@dataclasses.dataclass(frozen=True)
class HumanoidUpperBodyInputs(transforms.DataTransformFn):
    """Inputs for the humanoid upper-body policy."""

    adapt_to_pi: bool = True
    use_front_camera: bool = True
    # Norm-stats only: process state/actions without loading camera frames.
    skip_images: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = _ALL_CAMERAS

    def __call__(self, data: dict) -> dict:
        if self.skip_images:
            state = _decode_state(np.asarray(data["state"]), adapt_to_pi=self.adapt_to_pi)
            inputs: dict = {"state": state}
            if "actions" in data:
                actions = np.asarray(data["actions"])
                inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return inputs

        data = _decode_humanoid(data, adapt_to_pi=self.adapt_to_pi)

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

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class HumanoidUpperBodyOutputs(transforms.DataTransformFn):
    """Outputs for the humanoid upper-body policy."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
