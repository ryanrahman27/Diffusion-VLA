"""Data pipeline config for PiperX + metric depth (standalone extension)."""

import dataclasses
import pathlib

import tyro
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
from openpi.depth.constants import DEFAULT_DEPTH_RANGE_M_PER_CAMERA
from openpi.depth.constants import DEFAULT_DEPTH_SCALE_M
from openpi.depth.piperx_depth_policy import PiperXDepthInputs
from openpi.depth.pi0_depth_config import Pi0DepthConfig
from openpi.policies import piperx_policy
from openpi.training.config import DataConfig
from openpi.training.config import GroupFactory
from openpi.training.config import LeRobotPiperXBimanualDataConfig
import openpi.transforms as transforms


def _piperx_depth_repack_transforms(*, use_front_camera: bool) -> _transforms.Group:
    image_map = {
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    depth_map = {
        "cam_left_wrist": "observation.depth.cam_left_wrist",
        "cam_right_wrist": "observation.depth.cam_right_wrist",
    }
    if use_front_camera:
        image_map = {"cam_front": "observation.images.cam_front", **image_map}
        depth_map = {"cam_front": "observation.depth.cam_front", **depth_map}

    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": image_map,
                    "depth": depth_map,
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "task",
                }
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class DepthModelTransformFactory(GroupFactory):
    """Model transforms for Pi0DepthConfig (π₀.₅ tokenization + resize all image/depth slots)."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not isinstance(model_config, Pi0DepthConfig):
            raise TypeError(f"DepthModelTransformFactory expects Pi0DepthConfig, got {type(model_config)}")
        if not model_config.pi05:
            raise ValueError("Pi0DepthConfig is intended for pi05=True fine-tuning.")

        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXDepthDataConfig(LeRobotPiperXBimanualDataConfig):
    """PiperX bimanual LeRobot data with RGB + metric depth keys."""

    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M
    depth_range_per_camera: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_DEPTH_RANGE_M_PER_CAMERA)
    )

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _piperx_depth_repack_transforms(use_front_camera=True)
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = (
            _piperx_depth_repack_transforms(use_front_camera=False)
            if not self.use_front_camera
            else self.repack_transforms
        )
        data_transforms = _transforms.Group(
            inputs=[
                PiperXDepthInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_front_camera=self.use_front_camera,
                    depth_scale_m=self.depth_scale_m,
                    depth_range_per_camera=dict(self.depth_range_per_camera),
                )
            ],
            outputs=[piperx_policy.PiperXOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_action_mask)],
                outputs=[transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = DepthModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
