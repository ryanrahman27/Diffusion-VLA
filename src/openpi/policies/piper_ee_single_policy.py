"""Input/output transforms for single-arm Piper end-effector (Cartesian) control.

Raw dataset state / action (10-dim):
    [x, y, z, rot6d_0..5, gripper (m)]

With UMI relative proprio (``use_relative_proprio``), model state becomes 20-D:
    [identity_pose(9), current_gripper(1), past_pose_rel(9), past_gripper(1)]

Uses the same pi0 image slot mapping as PiperX wrist-only mode: ``cam_left_wrist``
-> ``left_wrist_0_rgb``; base and right wrist slots are zeroed and masked.
"""

import dataclasses
from typing import ClassVar, Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.policies.piperx_policy import build_piperx_image_inputs

_STATE_ABSOLUTE_KEY = "state_absolute"

PIPER_EE_DIM = 10
PIPER_GRIPPER_CLOSED_M = 0.0
PIPER_GRIPPER_OPEN_M = 0.105

GRIPPER_IDX = 9
EXPECTED_CAMERAS: tuple[str, ...] = ("cam_left_wrist",)


def make_piper_ee_single_example(*, umi_state: bool = False) -> dict:
    """Fake observation matching single-arm EE inference wire format (CHW + state)."""
    if umi_state:
        state = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.05,
                -0.02,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.04,
            ],
            dtype=np.float32,
        )
    else:
        state = np.array(
            [
                0.03,
                -0.01,
                0.29,
                0.40,
                -0.05,
                0.91,
                -0.36,
                -0.93,
                0.11,
                0.05,
            ],
            dtype=np.float32,
        )
    return {
        "state": state,
        "images": {
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick cube and place in bin",
    }


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_model(value):
    return _normalize(value, min_val=PIPER_GRIPPER_CLOSED_M, max_val=PIPER_GRIPPER_OPEN_M)


def _gripper_from_model(value):
    return _unnormalize(value, min_val=PIPER_GRIPPER_CLOSED_M, max_val=PIPER_GRIPPER_OPEN_M)


def _decode_state(
    state: np.ndarray,
    *,
    adapt_to_pi: bool = False,
    gripper_indices: Sequence[int] = (GRIPPER_IDX,),
) -> np.ndarray:
    if adapt_to_pi:
        state = np.asarray(state, dtype=np.float32).copy()
        state[..., list(gripper_indices)] = _gripper_to_model(state[..., list(gripper_indices)])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[..., GRIPPER_IDX] = _gripper_from_model(actions[..., GRIPPER_IDX])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[..., GRIPPER_IDX] = _gripper_to_model(actions[..., GRIPPER_IDX])
    return actions


def _decode_piper_ee_single(
    data: dict,
    *,
    adapt_to_pi: bool = False,
    gripper_indices: Sequence[int] = (GRIPPER_IDX,),
) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, gripper_indices=gripper_indices)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    data["images"] = {name: convert_image(img) for name, img in images.items()}
    data["state"] = state
    return data


def _passthrough_anchor_state(data: dict, inputs: dict) -> dict:
    if _STATE_ABSOLUTE_KEY in data:
        inputs[_STATE_ABSOLUTE_KEY] = data[_STATE_ABSOLUTE_KEY]
    return inputs


@dataclasses.dataclass(frozen=True)
class PiperEeSingleInputs(transforms.DataTransformFn):
    """Single-arm EE 6D inputs (training + inference)."""

    adapt_to_pi: bool = True
    gripper_indices: Sequence[int] = (GRIPPER_IDX,)
    skip_images: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = EXPECTED_CAMERAS

    def __call__(self, data: dict) -> dict:
        if self.skip_images:
            state = _decode_state(
                np.asarray(data["state"]),
                adapt_to_pi=self.adapt_to_pi,
                gripper_indices=self.gripper_indices,
            )
            inputs: dict = {"state": state}
            if "actions" in data:
                inputs["actions"] = _encode_actions_inv(np.asarray(data["actions"]), adapt_to_pi=self.adapt_to_pi)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return _passthrough_anchor_state(data, inputs)

        data = _decode_piper_ee_single(
            data,
            adapt_to_pi=self.adapt_to_pi,
            gripper_indices=self.gripper_indices,
        )

        in_images = data["images"]
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(
                f"Unexpected camera(s) in input: {unknown}. "
                f"Expected subset of {self.EXPECTED_CAMERAS}."
            )

        images, image_masks = build_piperx_image_inputs(in_images, use_front_camera=False)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            inputs["actions"] = _encode_actions_inv(np.asarray(data["actions"]), adapt_to_pi=self.adapt_to_pi)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return _passthrough_anchor_state(data, inputs)


@dataclasses.dataclass(frozen=True)
class PiperEeSingleOutputs(transforms.DataTransformFn):
    """Single-arm EE outputs: slice to 10-D and denormalize gripper to meters."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :PIPER_EE_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
