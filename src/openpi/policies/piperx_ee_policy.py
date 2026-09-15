"""Input/output transforms for PiperX bimanual end-effector (Cartesian) control.

Separate from ``piperx_policy.py`` (joint space). Uses ``/observations/eef_6d`` from
``episode_recorder.py`` (continuous 6D rotation; avoids RPY wrap / gimbal issues).

State / action layout (20-dim, both):
    [L_x, L_y, L_z, L_rot6d_0..5, L_grip (m),
     R_x, R_y, R_z, R_rot6d_0..5, R_grip (m)]

With ``append_inter_gripper_rel=True``, state is extended to 29-D by appending the
left-gripper-to-right-gripper relative pose (9-D: xyz + rot6d, matching per-arm layout).

``rot6d`` is the first two columns of the rotation matrix (rows r00,r10,r20,r01,r11,r21).
Grippers are normalized to [0, 1] for the model; deployment converts back to meters.

``DeltaActions`` uses component-wise deltas on xyz + rot6d (anchored to chunk-start
state, not chained across steps); grippers stay absolute. For UMI PD2.1 SE(3)
relative-to-t0 encoding, use ``RelativeToT0EeActions`` via
``use_relative_to_t0_ee_actions`` on ``LeRobotPiperXEEBimanualDataConfig``. For UMI PD2.2
relative past proprio, use ``RelativeToCurrentEeProprio`` via ``use_relative_proprio``
(40-D state: current absolute + past relative).
"""

import dataclasses
from typing import ClassVar, Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.policies.piperx_policy import _ALL_CAMERAS
from openpi.policies.piperx_policy import build_piperx_image_inputs
from openpi.policies.piperx_rel_pose import PIPERX_EE_DIM
from openpi.policies.piperx_rel_pose import PIPERX_EE_STATE_WITH_REL_DIM
from openpi.policies.piperx_rel_pose import PIPERX_RIGHT_BASE_IN_LEFT_BASE
from openpi.policies.piperx_rel_pose import append_inter_gripper_rel_to_ee_state

# Shared gripper calibration with joint-space PiperX policy.
PIPERX_GRIPPER_CLOSED_M = 0.0
PIPERX_GRIPPER_OPEN_M = 0.07

GRIPPER_INDICES = (9, 19)


def make_piperx_ee_example() -> dict:
    """Fake observation matching EE 6D inference wire format (CHW images + 20-D state)."""
    return {
        "state": np.array(
            [
                0.03, -0.01, 0.29,
                0.40, -0.05, 0.91, -0.36, -0.93, 0.11,
                0.05,
                0.02, 0.00, 0.30,
                0.41, 0.12, 0.90, 0.14, -0.99, 0.06,
                0.05,
            ],
            dtype=np.float32,
        ),
        "images": {
            "cam_front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "fold towel",
    }


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_model(value):
    return _normalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _gripper_from_model(value):
    return _unnormalize(value, min_val=PIPERX_GRIPPER_CLOSED_M, max_val=PIPERX_GRIPPER_OPEN_M)


def _decode_state(
    state: np.ndarray,
    *,
    adapt_to_pi: bool = False,
    gripper_indices: Sequence[int] = GRIPPER_INDICES,
) -> np.ndarray:
    if adapt_to_pi:
        state = np.asarray(state, dtype=np.float32).copy()
        state[..., list(gripper_indices)] = _gripper_to_model(state[..., list(gripper_indices)])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[..., list(GRIPPER_INDICES)] = _gripper_from_model(actions[..., list(GRIPPER_INDICES)])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[..., list(GRIPPER_INDICES)] = _gripper_to_model(actions[..., list(GRIPPER_INDICES)])
    return actions


def _decode_piperx_ee(
    data: dict,
    *,
    adapt_to_pi: bool = False,
    gripper_indices: Sequence[int] = GRIPPER_INDICES,
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


@dataclasses.dataclass(frozen=True)
class PiperXEeInputs(transforms.DataTransformFn):
    """EE 6D-space inputs for PiperX bimanual (training + inference)."""

    adapt_to_pi: bool = True
    use_front_camera: bool = True
    append_inter_gripper_rel: bool = False
    right_base_in_left_base: tuple[float, float, float] = tuple(PIPERX_RIGHT_BASE_IN_LEFT_BASE.tolist())
    gripper_indices: Sequence[int] = GRIPPER_INDICES
    # Norm-stats only: process state/actions without loading camera frames.
    skip_images: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = _ALL_CAMERAS

    def __call__(self, data: dict) -> dict:
        if self.skip_images:
            state = _decode_state(
                np.asarray(data["state"]),
                adapt_to_pi=self.adapt_to_pi,
                gripper_indices=self.gripper_indices,
            )
            if self.append_inter_gripper_rel:
                state = append_inter_gripper_rel_to_ee_state(
                    state,
                    right_base_in_left_base=self.right_base_in_left_base,
                )
            inputs: dict = {"state": state}
            if "actions" in data:
                actions = np.asarray(data["actions"])
                inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return inputs

        data = _decode_piperx_ee(
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

        images, image_masks = build_piperx_image_inputs(
            in_images, use_front_camera=self.use_front_camera
        )

        state = data["state"]
        if self.append_inter_gripper_rel:
            state = append_inter_gripper_rel_to_ee_state(
                state,
                right_base_in_left_base=self.right_base_in_left_base,
            )

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperXEeOutputs(transforms.DataTransformFn):
    """EE 6D outputs: slice to 20-D and denormalize grippers to meters."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :PIPERX_EE_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
