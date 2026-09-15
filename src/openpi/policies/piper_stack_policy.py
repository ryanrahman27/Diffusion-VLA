"""Input/output transforms for single-arm Piper stack (7-D joint + gripper).

Input camera names after repack:
    cam_left_wrist
    cam_right_wrist
    cam_front

State / action (7-D):
    [q1..q6 (rad), gripper (m)]
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies.piperx_policy import PIPERX_GRIPPER_CLOSED_M
from openpi.policies.piperx_policy import PIPERX_GRIPPER_OPEN_M
from openpi.policies.piperx_policy import build_piperx_image_inputs

PIPER_STACK_DIM = 7
_GRIPPER_IDX = 6

_WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")
_ALL_CAMERAS = ("cam_front", *_WRIST_CAMERAS)


def make_piper_stack_example() -> dict:
    return {
        "state": np.ones((PIPER_STACK_DIM,), dtype=np.float32) * 0.5,
        "images": {
            "cam_front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "stack towel",
    }


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_model(value):
    return _normalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _gripper_from_model(value):
    return _unnormalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if adapt_to_pi:
        state = state.copy()
        state[..., _GRIPPER_IDX] = _gripper_to_model(state[..., _GRIPPER_IDX])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if adapt_to_pi:
        actions = actions.copy()
        actions[..., _GRIPPER_IDX] = _gripper_from_model(actions[..., _GRIPPER_IDX])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    return _decode_state(actions, adapt_to_pi=adapt_to_pi)


@dataclasses.dataclass(frozen=True)
class PiperStackInputs(transforms.DataTransformFn):
    """Single-arm Piper stack inputs for pi0 / pi0.5."""

    adapt_to_pi: bool = True
    use_front_camera: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = _ALL_CAMERAS

    def __call__(self, data: dict) -> dict:
        state = _decode_state(data["state"], adapt_to_pi=self.adapt_to_pi)

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] == 3:
                return einops.rearrange(img, "c h w -> h w c")
            return img

        in_images = {name: convert_image(img) for name, img in data["images"].items()}
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(f"Unexpected camera(s): {unknown}. Expected subset of {self.EXPECTED_CAMERAS}.")

        images, image_masks = build_piperx_image_inputs(
            in_images, use_front_camera=self.use_front_camera
        )

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PiperStackOutputs(transforms.DataTransformFn):
    """Slice model output back to 7-D Piper stack actions."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :PIPER_STACK_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
