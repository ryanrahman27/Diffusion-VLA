"""Data pipeline for PiperX + JEPA future frames."""

import dataclasses
import pathlib

import tyro
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
from openpi.jepa.constants import DEFAULT_JEPA_HORIZON_STEPS
from openpi.jepa.piperx_jepa_inputs import PiperXJepaInputs
from openpi.jepa.pi0_jepa_config import Pi0JepaConfig
from openpi.policies import piperx_policy
from openpi.training.config import DataConfig
from openpi.training.config import GroupFactory
from openpi.training.config import LeRobotPiperXBimanualDataConfig


def _piperx_jepa_repack_transforms(*, use_front_camera: bool) -> _transforms.Group:
    image_map = {
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    if use_front_camera:
        image_map = {"cam_front": "observation.images.cam_front", **image_map}

    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": image_map,
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "task",
                }
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class JepaModelTransformFactory(GroupFactory):
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not isinstance(model_config, Pi0JepaConfig):
            raise TypeError(f"JepaModelTransformFactory expects Pi0JepaConfig, got {type(model_config)}")
        if not model_config.pi05:
            raise ValueError("Pi0JepaConfig is intended for pi05=True fine-tuning.")

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
class LeRobotPiperXJepaDataConfig(LeRobotPiperXBimanualDataConfig):
    """PiperX LeRobot data with multi-horizon future frames for JEPA."""

    jepa_horizon_steps: tuple[int, ...] = DEFAULT_JEPA_HORIZON_STEPS

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _piperx_jepa_repack_transforms(use_front_camera=True)
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = (
            _piperx_jepa_repack_transforms(use_front_camera=False)
            if not self.use_front_camera
            else self.repack_transforms
        )
        data_transforms = _transforms.Group(
            inputs=[
                PiperXJepaInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_front_camera=self.use_front_camera,
                    jepa_horizon_steps=(
                        model_config.jepa_horizon_steps
                        if isinstance(model_config, Pi0JepaConfig)
                        else self.jepa_horizon_steps
                    ),
                )
            ],
            outputs=[piperx_policy.PiperXOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = JepaModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
