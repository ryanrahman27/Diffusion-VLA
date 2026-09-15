"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.humanoid_upper_body_policy as humanoid_upper_body_policy
import openpi.policies.piper_ee_single_policy as piper_ee_single_policy
import openpi.policies.piper_ee_single_rel_proprio as piper_ee_single_rel_proprio
import openpi.policies.piper_ee_single_t0_actions as piper_ee_single_t0_actions
import openpi.policies.piper_stack_policy as piper_stack_policy
import openpi.policies.piperx_ee_policy as piperx_ee_policy
import openpi.policies.piperx_ee_rel_proprio as piperx_ee_rel_proprio
import openpi.policies.piperx_ee_t0_actions as piperx_ee_t0_actions
import openpi.policies.piperx_policy as piperx_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # Frames to shift the action target forward (latency-aware training):
    # obs[t] -> action[t+shift : t+shift+H]. shift = round(exec_latency*fps). 0 = off (default).
    action_target_shift: int = 0

    # LeRobot keys for proprio history (e.g. ``observation.state`` with offsets ``[-dt, 0]``).
    state_sequence_keys: Sequence[str] = ()
    # Number of past proprio frames before the current frame (obs horizon = 1 + this value).
    proprio_history_steps: int = 0
    # When set, load exactly two proprio frames at ``[-stride / fps, 0]`` (UMI re-query interval).
    # If None, past frames are spaced by single timesteps via ``proprio_history_steps``.
    proprio_history_stride: int | None = None

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
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
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def _piper_stack_act_recap_repack_transforms(*, use_front_camera: bool) -> _transforms.Group:
    """Repack axiboai/piper-stack-act-recap image keys into Piper-style camera names."""
    image_map = {
        "cam_left_wrist": "observation.images.wrist_left",
        "cam_right_wrist": "observation.images.wrist_right",
    }
    if use_front_camera:
        image_map = {"cam_front": "observation.images.scene_top", **image_map}
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


def _piperx_bimanual_repack_transforms(
    *,
    use_front_camera: bool,
    include_eef_6d: bool = False,
) -> _transforms.Group:
    image_map = {
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    if use_front_camera:
        image_map = {
            "cam_front": "observation.images.cam_front",
            **image_map,
        }
    repack_map = {
        "images": image_map,
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
    }
    if include_eef_6d:
        repack_map["eef_6d"] = "observation.eef_6d"
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(repack_map)
        ]
    )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXBimanualDataConfig(DataConfigFactory):
    """Data config factory for the PiperX bimanual robot.

    Mirrors LeRobotAlohaDataConfig: delta arm actions + absolute gripper,
    PiperX-specific input/output transforms, repack that adapts the
    LeRobot dataset key layout to what the transforms expect.
    """

    # Arms: delta-encoded relative to current state. Grippers: absolute.
    # Same mask shape as ALOHA because the layout is identical.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    # Toggles the joint flip / gripper normalize inside PiperXInputs /
    # PiperXOutputs. True is the only sensible value during fine-tuning
    # from pi0.5 base; left exposed for experiments.
    adapt_to_pi: bool = True
    # If False, training/inference use only wrist cameras (front is not loaded).
    use_front_camera: bool = True
    # Append 9-D left-to-right relative EE pose to joint proprio (14-D -> 23-D state).
    append_inter_gripper_rel: bool = False
    # Right-arm base origin in left-arm base frame (parallel mounts, 50 cm lateral).
    right_base_in_left_base: tuple[float, float, float] = (0.0, -0.5, 0.0)

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _piperx_bimanual_repack_transforms(use_front_camera=True)
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = _piperx_bimanual_repack_transforms(
            use_front_camera=self.use_front_camera,
            include_eef_6d=self.append_inter_gripper_rel,
        )
        data_transforms = _transforms.Group(
            inputs=[
                piperx_policy.PiperXInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_front_camera=self.use_front_camera,
                    append_inter_gripper_rel=self.append_inter_gripper_rel,
                    right_base_in_left_base=self.right_base_in_left_base,
                )
            ],
            outputs=[piperx_policy.PiperXOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            # Arms (6) delta-encoded, gripper (-1) absolute, repeated per arm.
            # Exactly the same mask ALOHA uses.
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def _humanoid_upper_body_repack_transforms(*, use_front_camera: bool) -> _transforms.Group:
    image_map = {
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    if use_front_camera:
        image_map = {
            "cam_front": "observation.images.cam_front",
            **image_map,
        }
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
class LeRobotHumanoidUpperBodyDataConfig(DataConfigFactory):
    """Data config for a bimanual humanoid upper body (7+1 DoF per arm, 16-D total)."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    use_front_camera: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _humanoid_upper_body_repack_transforms(use_front_camera=True)
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = (
            _humanoid_upper_body_repack_transforms(use_front_camera=False)
            if not self.use_front_camera
            else self.repack_transforms
        )
        data_transforms = _transforms.Group(
            inputs=[
                humanoid_upper_body_policy.HumanoidUpperBodyInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_front_camera=self.use_front_camera,
                )
            ],
            outputs=[
                humanoid_upper_body_policy.HumanoidUpperBodyOutputs(adapt_to_pi=self.adapt_to_pi)
            ],
        )

        if self.use_delta_joint_actions:
            # 7 left + 7 right arm joints delta-encoded; 2 hand pinches absolute.
            delta_action_mask = _transforms.make_bool_mask(7, 7, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperStackActRecapDataConfig(DataConfigFactory):
    """Single-arm Piper stack RECAP dataset (7-D joint, 3 cameras).

    Expects HF LeRobot keys:
      - observation.images.wrist_left
      - observation.images.wrist_right
      - observation.images.scene_top
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    use_front_camera: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _piper_stack_act_recap_repack_transforms(use_front_camera=True)
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = (
            _piper_stack_act_recap_repack_transforms(use_front_camera=False)
            if not self.use_front_camera
            else self.repack_transforms
        )
        data_transforms = _transforms.Group(
            inputs=[
                piper_stack_policy.PiperStackInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_front_camera=self.use_front_camera,
                )
            ],
            outputs=[piper_stack_policy.PiperStackOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXEEBimanualDataConfig(DataConfigFactory):
    """Data config for PiperX bimanual EE pose (HDF5 ``observations/eef_6d``, 20-D rot6d)."""

    # xyz (3) + rot6d (6) delta-encoded per arm; grippers absolute.
    use_delta_ee_actions: bool = True
    # UMI PD2.1: SE(3) relative-to-t0 per arm (mutually exclusive with use_delta_ee_actions).
    use_relative_to_t0_ee_actions: bool = False
    # UMI PD2.2: append past EE pose relative to current (20-D -> 40-D state).
    use_relative_proprio: bool = False
    # Past proprio frames loaded before the current frame (1 => obs horizon 2).
    proprio_history_steps: int = 1
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    use_front_camera: bool = True
    # Append 9-D left-to-right relative EE pose to proprio (20-D -> 29-D state).
    append_inter_gripper_rel: bool = False
    # Right-arm base origin in left-arm base frame (parallel mounts, 50 cm lateral).
    right_base_in_left_base: tuple[float, float, float] = (0.0, -0.5, 0.0)

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _piperx_bimanual_repack_transforms(use_front_camera=True)
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_relative_to_t0_ee_actions and self.use_delta_ee_actions:
            raise ValueError(
                "use_relative_to_t0_ee_actions and use_delta_ee_actions are mutually exclusive."
            )
        if self.use_relative_proprio and self.append_inter_gripper_rel:
            raise ValueError(
                "use_relative_proprio and append_inter_gripper_rel cannot both be enabled."
            )

        repack_transforms = (
            _piperx_bimanual_repack_transforms(use_front_camera=False)
            if not self.use_front_camera
            else self.repack_transforms
        )
        gripper_indices = piperx_ee_policy.GRIPPER_INDICES
        input_transforms: list[_transforms.DataTransformFn] = []
        if self.use_relative_proprio:
            input_transforms.append(piperx_ee_rel_proprio.RelativeToCurrentEeProprio())
            gripper_indices = piperx_ee_rel_proprio.GRIPPER_INDICES_WITH_REL_PROPRIO

        input_transforms.append(
            piperx_ee_policy.PiperXEeInputs(
                adapt_to_pi=self.adapt_to_pi,
                use_front_camera=self.use_front_camera,
                append_inter_gripper_rel=self.append_inter_gripper_rel,
                right_base_in_left_base=self.right_base_in_left_base,
                gripper_indices=gripper_indices,
            )
        )
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[piperx_ee_policy.PiperXEeOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        state_sequence_keys: Sequence[str] = ()
        proprio_history_steps = 0
        if self.use_relative_proprio:
            state_sequence_keys = ("observation.state",)
            proprio_history_steps = self.proprio_history_steps

        if self.use_relative_to_t0_ee_actions:
            data_transforms = data_transforms.push(
                inputs=[piperx_ee_t0_actions.RelativeToT0EeActions()],
                outputs=[piperx_ee_t0_actions.AbsoluteFromT0EeActions()],
            )
        elif self.use_delta_ee_actions:
            # Per arm: position (3) + rot6d (6) as deltas; gripper (1) absolute.
            delta_action_mask = _transforms.make_bool_mask(3, 6, -1, 3, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            state_sequence_keys=state_sequence_keys,
            proprio_history_steps=proprio_history_steps,
        )


# Left/right arm z position in PiperX bimanual EE layout: xyz(3)+rot6d(6)+gripper per arm.
PIPERX_EE_Z_AXIS_INDICES: tuple[int, ...] = (2, 12)


def _apply_piperx_ee_z_loss_weight(
    norm_stats: dict[str, _transforms.NormStats] | None,
    *,
    z_axis_weight: float,
    z_indices: tuple[int, ...] = PIPERX_EE_Z_AXIS_INDICES,
) -> dict[str, _transforms.NormStats] | None:
    """Upweight z-axis flow-matching loss by shrinking action norm scales on z dims.

    Dividing std / quantile range by ``sqrt(z_axis_weight)`` makes normalized z errors
    larger, so MSE loss emphasizes vertical motion without changing other configs.
    """
    if norm_stats is None or z_axis_weight == 1.0:
        return norm_stats
    if z_axis_weight <= 0:
        raise ValueError(f"z_axis_weight must be positive, got {z_axis_weight}.")

    actions = norm_stats.get("actions")
    if actions is None:
        return norm_stats

    scale = 1.0 / np.sqrt(z_axis_weight)
    std = np.array(actions.std, dtype=np.float64, copy=True)
    std[list(z_indices)] *= scale

    q01 = actions.q01
    q99 = actions.q99
    if q01 is not None and q99 is not None:
        q01 = np.array(q01, dtype=np.float64, copy=True)
        q99 = np.array(q99, dtype=np.float64, copy=True)
        mid = (q01 + q99) / 2.0
        half_range = (q99 - q01) / 2.0
        half_range[list(z_indices)] *= scale
        q01[list(z_indices)] = mid[list(z_indices)] - half_range[list(z_indices)]
        q99[list(z_indices)] = mid[list(z_indices)] + half_range[list(z_indices)]

    weighted_actions = _normalize.NormStats(
        mean=actions.mean,
        std=std.astype(np.float32),
        q01=None if q01 is None else q01.astype(np.float32),
        q99=None if q99 is None else q99.astype(np.float32),
    )
    return {**norm_stats, "actions": weighted_actions}


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXEEZWeightedBimanualDataConfig(LeRobotPiperXEEBimanualDataConfig):
    """PiperX EE data config that upweights z-axis in the training loss via action norm stats."""

    z_axis_weight: float = 2.0
    z_axis_indices: tuple[int, ...] = PIPERX_EE_Z_AXIS_INDICES

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_config = super().create(assets_dirs, model_config)
        return dataclasses.replace(
            data_config,
            norm_stats=_apply_piperx_ee_z_loss_weight(
                data_config.norm_stats,
                z_axis_weight=self.z_axis_weight,
                z_indices=self.z_axis_indices,
            ),
        )


def _piper_ee_single_repack_transforms() -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {"cam_left_wrist": "observation.images.cam_left_wrist"},
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "task",
                }
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperEESingleArmDataConfig(DataConfigFactory):
    """Single-arm Piper EE pose (10-D rot6d) with one wrist camera."""

    use_delta_ee_actions: bool = True
    use_relative_to_t0_ee_actions: bool = False
    use_relative_proprio: bool = False
    proprio_history_steps: int = 1
    # UMI: past proprio frame is this many dataset steps before current (default: action_horizon).
    proprio_history_stride: int | None = None
    default_prompt: str | None = None
    adapt_to_pi: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=_piper_ee_single_repack_transforms
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_relative_to_t0_ee_actions and self.use_delta_ee_actions:
            raise ValueError(
                "use_relative_to_t0_ee_actions and use_delta_ee_actions are mutually exclusive."
            )

        gripper_indices = (piper_ee_single_policy.GRIPPER_IDX,)
        input_transforms: list[_transforms.DataTransformFn] = []
        output_transforms: list[_transforms.DataTransformFn] = [
            piper_ee_single_policy.PiperEeSingleOutputs(adapt_to_pi=self.adapt_to_pi),
        ]

        if self.use_relative_to_t0_ee_actions:
            input_transforms.append(piper_ee_single_t0_actions.RelativeToT0EeActions())
            output_transforms.insert(0, piper_ee_single_t0_actions.AbsoluteFromT0EeActions())

        if self.use_relative_proprio:
            input_transforms.append(piper_ee_single_rel_proprio.RelativeToCurrentEeProprio())
            gripper_indices = piper_ee_single_rel_proprio.GRIPPER_INDICES_WITH_REL_PROPRIO

        input_transforms.append(
            piper_ee_single_policy.PiperEeSingleInputs(
                adapt_to_pi=self.adapt_to_pi,
                gripper_indices=gripper_indices,
            )
        )
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=output_transforms,
        )

        state_sequence_keys: Sequence[str] = ()
        proprio_history_steps = 0
        proprio_history_stride: int | None = None
        if self.use_relative_proprio:
            state_sequence_keys = ("observation.state",)
            proprio_history_steps = self.proprio_history_steps
            proprio_history_stride = (
                self.proprio_history_stride
                if self.proprio_history_stride is not None
                else model_config.action_horizon
            )

        if self.use_delta_ee_actions and not self.use_relative_to_t0_ee_actions:
            delta_action_mask = _transforms.make_bool_mask(3, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            state_sequence_keys=state_sequence_keys,
            proprio_history_steps=proprio_history_steps,
            proprio_history_stride=proprio_history_stride,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),


    #
    # Fine-tuning PiperX bimanual configs.
    #
    TrainConfig(
        name="pi05_piperx_fold",
        # Use pi0.5 (newer, better, default for ALOHA/DROID/Libero).
        model=pi0_config.Pi0Config(pi05=True),
        # Fine-tune from the pi0.5 base checkpoint on GCS.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_fold",
            # Pull the per-episode task string from the LeRobot dataset
            # (every frame has "task" = "fold towel" for this dataset).
            base_config=DataConfig(prompt_from_task=True),
            # Where to read / write the norm stats. Local path under
            # ./assets/<config name>/piperx_bimanual/. compute_norm_stats.py
            # will populate it; train.py reads it and copies it into the
            # checkpoint's assets/ subdir.
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="fold towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=5e-5,
            decay_steps=4700,
            decay_lr=5e-6,
        ),

        num_train_steps=5000,
        batch_size=32,

        log_interval=100,
        save_interval=1000,
        keep_period=2000,
       
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_flatten_depth_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_flatten_depth",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_depth"),
            default_prompt="flatten towel",
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=9500,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_fold_wrist_only",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_fold",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="fold towel",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "use_front_camera": False,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_stack",
        # Use pi0.5 (newer, better, default for ALOHA/DROID/Libero).
        model=pi0_config.Pi0Config(pi05=True),
        # Fine-tune from the pi0.5 base checkpoint on GCS.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_stack",
            # Pull the per-episode task string from the LeRobot dataset
            # (every frame has "task" = "fold towel" for this dataset).
            base_config=DataConfig(prompt_from_task=True),
            # Where to read / write the norm stats. Local path under
            # ./assets/<config name>/piperx_bimanual/. compute_norm_stats.py
            # will populate it; train.py reads it and copies it into the
            # checkpoint's assets/ subdir.
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="fold towel and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),

        num_train_steps=10000,
        batch_size=32,

        log_interval=100,
        save_interval=2000,
        keep_period=2000,
       
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),


    # === axiboai/piper_stacking (bimanual cube stacking) BC fine-tunes ===
    # Same embodiment as Ishan-Axibo/piperx_stack (14-D bimanual; cam_front/cam_left_wrist/cam_right_wrist);
    # cloned from pi05_piperx_stack with repo_id retargeted to axiboai/piper_stacking.
    # Trained checkpoints: https://huggingface.co/axiboai/pi0_piper_stacking and .../pi05_piper_stacking
    TrainConfig(
        name="pi05_piper_stacking",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=2500,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    # === piper_stacking_v2 (more demos + gripper/start-pose corrections) — pi0.5 baseline ===
    # Same recipe as pi05_piper_stacking (h50) for a clean v1-vs-v2 comparison.
    TrainConfig(
        name="pi05_piper_stacking_v2",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking_v2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600, peak_lr=5e-5, decay_steps=9400, decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=2500,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),


    # === piper_laundry_calibrated_v2 (towel pick-fold-stack, 277 eps / 468k frames) — pi0.5 h50 full fine-tune ===

    # === piper_stacking_v3 (v2 + 50 new eps: top-left corner, bottom-between-robots, varied cube orientation) — pi0.5 h50 ===
    TrainConfig(
        name="pi05_piper_stacking_v3",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking_v3",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600, peak_lr=5e-5, decay_steps=9400, decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=2500,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),


    # === piper_stacking_v3 + Chef-style latency-aware ACTION-TARGET-SHIFT (obs[t] -> action[t+5:t+5+H]; 176ms exec @30fps ~= 5 frames) ===
    TrainConfig(
        name="pi05_piper_stacking_v3_latency",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking_v3",
            base_config=DataConfig(prompt_from_task=True, action_target_shift=5),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600, peak_lr=5e-5, decay_steps=9400, decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=2500,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    # === piper_stacking + ABC-130k YAM subset (axiboai/piper_stacking_abc_mix) — pi0.5 h50 ===
    # Build data: examples/piper_real/convert_abc_mcap_to_lerobot.py + merge_lerobot_datasets.py
    # HF model: axiboai/pi05_piper_stacking_abc_mix
    TrainConfig(
        name="pi05_piper_stacking_abc_mix",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking_abc_mix",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=19000,
            decay_lr=5e-6,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piper_folding_v2",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_laundry_calibrated_v2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-5, decay_steps=19000, decay_lr=5e-6,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),


    # === piper_folding IWR: base folding (277 eps) + 34 interventions x2 (pause-trimmed); warm-start from folding-20k, short LR-2e-5 corrective pass ===
    TrainConfig(
        name="pi05_piper_folding_iwr",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/sagar_recap/piperx-openpi/checkpoints/pi05_piper_folding_v2/pi05_folding_v2_v1/19999/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_folding_iwr_v1",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=2e-5, decay_steps=4800, decay_lr=2e-6,
        ),
        num_train_steps=5000,
        batch_size=32,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    # === Co-train stacking_v3 + folding_iwr_v1 (merge with merge_lerobot_datasets.py) ===
    TrainConfig(
        name="pi05_piper_stacking_folding_v1",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking_folding_v1",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-5, decay_steps=19000, decay_lr=5e-6,
        ),
        num_train_steps=20000,
        batch_size=32,
        num_workers=8,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    # === piper_folding IWR v2: base(277, pause-trimmed) + interventions1(34) + interventions2(21), all pause-trimmed; warm-start from folding-20k, 10k steps ===
    # Two variants differ ONLY by how much the correction episodes are oversampled (weighting).
    TrainConfig(
        name="pi05_piper_folding_iwr_v2_x2",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/sagar_recap/folding_iwr_v2/base_ckpt/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_folding_iwr_v2_x2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=2e-5, decay_steps=9800, decay_lr=2e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        num_workers=10,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        name="pi05_piper_folding_iwr_v2_x3",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/sagar_recap/folding_iwr_v2/base_ckpt/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_folding_iwr_v2_x3",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=2e-5, decay_steps=9800, decay_lr=2e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        num_workers=10,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi0_piper_stacking",
        model=pi0_config.Pi0Config(pi05=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=2500,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),


    # === action_horizon=25 variants (more reactive closed-loop; 0.83s/chunk @30fps vs 1.67s for h50) ===
    TrainConfig(
        name="pi05_piper_stacking_h25",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=25),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600, peak_lr=5e-5, decay_steps=9400, decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        name="pi0_piper_stacking_h25",
        model=pi0_config.Pi0Config(pi05=False, action_horizon=25),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_stacking",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="stack red cube on blue cube",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600, peak_lr=5e-5, decay_steps=9400, decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_flatten",
        # Use pi0.5 (newer, better, default for ALOHA/DROID/Libero).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        # Fine-tune from the pi0.5 base checkpoint on GCS.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_flatten_merged",
            # Pull the per-episode task string from the LeRobot dataset
            # (every frame has "task" = "fold towel" for this dataset).
            base_config=DataConfig(prompt_from_task=True),
            # Where to read / write the norm stats. Local path under
            # ./assets/<config name>/piperx_bimanual/. compute_norm_stats.py
            # will populate it; train.py reads it and copies it into the
            # checkpoint's assets/ subdir.
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=900,
            peak_lr=5e-5,
            decay_steps=14100,
            decay_lr=5e-6,
        ),

        num_train_steps=15000,
        batch_size=32,

        log_interval=100,
        save_interval=5000,
        keep_period=5000,
       
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_flatten_rollouts",
        # Incremental fine-tune of the laundry teleop flatten ckpt on successful
        # HIL rollouts only (~56 episodes). Use a rollout-only raw dir (HIL
        # schema); do not mix in teleop HDF5s. Convert with:
        #   python scripts/convert_rollouts_to_lerobot.py \
        #     --raw-dir /path/to/piperx_lerobot_setup/episodes/laundry_rollouts \
        #     --repo-id Ishan-Axibo/piperx_laundry_rollouts_success \
        #     --task "pick towel from pile, fold and stack"
        # (filter to success episodes before or after conversion; failures are
        # skipped by RECAP but still inflate conversion time if left in raw-dir.)
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_piperx_flatten/from_hf/9999/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_rollouts_success",
            base_config=DataConfig(prompt_from_task=True),
            # Reuse norm stats from the parent flatten run (same robot + task).
            assets=AssetsConfig(
                asset_id="piperx_bimanual",
            ),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=4800,
            decay_lr=2e-6,
        ),
        num_train_steps=5000,
        batch_size=32,
        log_interval=100,
        save_interval=1000,
        keep_period=2000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        name="pi05_piperx_fold_ee",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_fold_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee"),
            default_prompt="fold towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=5e-5,
            decay_steps=4700,
            decay_lr=5e-6,
        ),
        num_train_steps=5000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_humanoid_pick_and_place",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotHumanoidUpperBodyDataConfig(
            repo_id="axiboai/fabric_pick_v2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="humanoid_upper_body"),
            default_prompt="pick up cube and place in bin",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=900,
            peak_lr=5e-5,
            decay_steps=14100,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "action_dim": 16,
            "schema": "humanoid_arms_v2_omnihand_state",
            "action_source": "next_state_t_plus_1",
        },
    ),

    TrainConfig(
        name="pi05_humanoid_laundry_fold",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotHumanoidUpperBodyDataConfig(
            repo_id="axiboai/humanoid_laundry",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="humanoid_upper_body"),
            default_prompt="fold towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=900,
            peak_lr=5e-5,
            decay_steps=14100,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "action_dim": 16,
            "schema": "humanoid_async_v1",
            "action_source": "next_state_t_plus_1",
        },
    ),

    TrainConfig(
        name="pi05_humanoid_pick",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotHumanoidUpperBodyDataConfig(
            repo_id="local/humanoid_pick",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="humanoid_upper_body"),
            default_prompt="pick cube and place in bin",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_dim": 16,
            "schema": "humanoid_arms_v1",
        },
    ),

    TrainConfig(
        name="pi05_piperx_fold_ee_wrist_only",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_fold_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee"),
            default_prompt="fold towel",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "use_front_camera": False,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_stack_ee",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_stack_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee"),
            default_prompt="fold towel and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_piperx_flatten/from_hf/9999/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="fold towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "joint",
            "state_dim": 14,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piper_laundry_calibrated_v2",
        # axiboai/piper_laundry_calibrated_v2 — 277 episodes, ~468k frames, joint-space
        # PiperX bimanual with calibrated extrinsics (cam_front + both wrists).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_laundry_calibrated_v2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=900,
            peak_lr=5e-5,
            decay_steps=14100,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=100,
        save_interval=2500,
        keep_period=5000,
        policy_metadata={
            "action_space": "joint",
            "state_dim": 14,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piper_stack_act_recap",
        # Single-arm Piper stack RECAP (axiboai/piper-stack-act-recap).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperStackActRecapDataConfig(
            repo_id="axiboai/piper-stack-act-recap",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piper_stack_act_recap"),
            default_prompt="stack towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "joint",
            "state_dim": 7,
            "schema": "piper_stack_act_recap",
        },
    ),

    TrainConfig(
        name="pi05_piper_stack_act_recap_success",
        # Success-only filter of axiboai/piper-stack-act-recap (103 episodes).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperStackActRecapDataConfig(
            repo_id="axiboai/piper-stack-act-recap-success",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piper_stack_act_recap_success"),
            default_prompt="stack towel",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "joint",
            "state_dim": 7,
            "schema": "piper_stack_act_recap_success",
        },
    ),

    TrainConfig(
        name="pi05_piper_umi_ee",
        # Baseline delta actions with absolute 10-D state (not UMI — use pi05_piper_umi_ee_rel_traj).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperEESingleArmDataConfig(
            repo_id="axiboai/piper-umi-ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piper_umi_ee"),
            default_prompt="pick cube and place in bin",
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 10,
            "state_dim": 10,
            "schema": "umi_ee_v1",
            "camera": "cam_left_wrist",
        },
    ),

    TrainConfig(
        name="pi05_piper_umi_ee_t0",
        # Relative SE(3) actions only; state is still absolute 10-D (use pi05_piper_umi_ee_rel_traj for UMI proprio).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperEESingleArmDataConfig(
            repo_id="axiboai/piper-umi-ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piper_umi_ee_t0"),
            default_prompt="pick cube and place in bin",
            use_relative_to_t0_ee_actions=True,
            use_delta_ee_actions=False,
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "action_encoding": "relative_to_t0_se3",
            "ee_dim": 10,
            "state_dim": 10,
            "schema": "umi_ee_v1",
            "camera": "cam_left_wrist",
        },
    ),

    TrainConfig(
        name="pi05_piper_umi_ee_rel_traj",
        # UMI: relative-to-current actions + relative proprio history (20-D state, no absolute pose).
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperEESingleArmDataConfig(
            repo_id="axiboai/piper-umi-ee-pick-calibrated",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piper_umi_ee_rel_traj"),
            default_prompt="pick cube and place in bin",
            use_relative_to_t0_ee_actions=True,
            use_delta_ee_actions=False,
            use_relative_proprio=True,
            proprio_history_steps=1,
            # proprio_history_stride defaults to action_horizon (50 frames @ 30 Hz ≈ 1.67 s).
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "eef_6d",
            "action_encoding": "relative_to_current_se3",
            "proprio_encoding": "umi_relative_history",
            "ee_dim": 10,
            "state_dim": 20,
            "use_relative_proprio": True,
            "proprio_history_stride": 50,
            "schema": "umi_ee_v1",
            "camera": "cam_left_wrist",
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_joint_rel",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_rel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_joint_rel"),
            default_prompt="fold towel",
            append_inter_gripper_rel=True,
            right_base_in_left_base=(0.0, -0.5, 0.0),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "joint",
            "state_dim": 23,
            "append_inter_gripper_rel": True,
            "right_base_in_left_base": [0.0, -0.5, 0.0],
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_ee",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_softfold_ee_269ep",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=900,
            peak_lr=5e-5,
            decay_steps=14100,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "state_dim": 20,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_softfold_ee_merged",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_softfold_ee_269ep",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee"),
            default_prompt="pick towel from pile, fold and stack",
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "state_dim": 20,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_ee_z_weighted",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEZWeightedBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_ee",
            base_config=DataConfig(prompt_from_task=True),
            # Reuse unweighted norm stats; z scaling is applied in-memory for this config only.
            assets=AssetsConfig(
                asset_id="piperx_bimanual_ee",
            ),
            default_prompt="fold towel",
            z_axis_weight=2.0,
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "state_dim": 20,
            "z_axis_weight": 2.0,
            "z_axis_indices": list(PIPERX_EE_Z_AXIS_INDICES),
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_ee_t0",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee_t0"),
            default_prompt="fold towel",
            use_relative_to_t0_ee_actions=True,
            use_delta_ee_actions=False,
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "action_encoding": "relative_to_t0_se3",
            "ee_dim": 20,
            "state_dim": 20,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_ee_rel_traj",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee_rel_traj"),
            default_prompt="fold towel",
            use_relative_to_t0_ee_actions=True,
            use_delta_ee_actions=False,
            use_relative_proprio=True,
            proprio_history_steps=1,
        ),
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "action_encoding": "relative_to_t0_se3",
            "proprio_encoding": "relative_to_current_se3",
            "ee_dim": 20,
            "state_dim": 40,
            "use_relative_proprio": True,
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        name="pi05_piperx_laundry_ee_rel",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXEEBimanualDataConfig(
            repo_id="Ishan-Axibo/piperx_laundry_ee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_bimanual_ee_rel"),
            default_prompt="fold towel",
            append_inter_gripper_rel=True,
            right_base_in_left_base=(0.0, -0.5, 0.0),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "action_space": "eef_6d",
            "ee_dim": 20,
            "state_dim": 29,
            "append_inter_gripper_rel": True,
            "right_base_in_left_base": [0.0, -0.5, 0.0],
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_place_mug_joint -- a bimanual
        # "place mug into machine" task recorded as JOINT ANGLES, not EE trajectories
        # (robot_type piperx_bimanual_joint; 50 eps / 39645 frames @ 30fps; 14-D
        # state/action = [L joint1-6, L gripper, R joint1-6, R gripper]). Uses the
        # LeRobotPiperXBimanualDataConfig joint recipe (same as pi05_piperx_stack):
        # PiperXInputs/Outputs, delta arm joints + absolute grippers (mask 6,-1,6,-1).
        # Wrist-only: the dataset has only cam_left_wrist + cam_right_wrist (no front),
        # so use_front_camera=False is REQUIRED (the repack KeyErrors on a missing
        # cam_front otherwise). No geometric wrist-cam augmentation is applied here --
        # the cameras stay fixed/consistent between training data and inference, so the
        # crop+tilt aug from the umi EE runs is deliberately omitted. Distinct asset_id
        # so norm stats do not collide with the other bimanual joint runs.
        # Run `compute_norm_stats.py --config-name pi05_piperx_place_mug_joint` first.
        name="pi05_piperx_place_mug_joint",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_place_mug_joint",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_place_mug_joint_bimanual"),
            default_prompt="place mug into machine",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        num_workers=2,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_pour_beans_joint -- a bimanual
        # "pour the beans into the coffee machine" task recorded as JOINT ANGLES
        # (robot_type piperx_bimanual_joint; 45 eps / 41191 frames @ 30fps; 14-D
        # state/action = [L joint1-6, L gripper, R joint1-6, R gripper]). Regular pi0.5
        # (NO MoH). Same recipe as pi05_piperx_place_mug_joint: LeRobotPiperXBimanualDataConfig
        # joint recipe (PiperXInputs/Outputs, delta arm joints + absolute grippers,
        # mask 6,-1,6,-1), fine-tune from pi05_base. Wrist-only dataset (cam_left_wrist +
        # cam_right_wrist, no front) so use_front_camera=False is REQUIRED. No wrist-cam
        # augmentation -- cameras stay consistent between training data and inference.
        # Distinct asset_id so norm stats do not collide with the other bimanual joint runs.
        # Run `compute_norm_stats.py --config-name pi05_piperx_pour_beans_joint` first.
        name="pi05_piperx_pour_beans_joint",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_pour_beans_joint",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_pour_beans_joint_bimanual"),
            default_prompt="pour the beans into the coffee machine",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        # 8x H200 machine -- crank data loading so it does not bottleneck the GPUs.
        num_workers=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_remove_mug_joint -- a bimanual
        # "remove mug from coffee machine" task recorded as JOINT ANGLES (robot_type
        # piperx_bimanual_joint; 40 eps / 24014 frames @ 30fps; 14-D state/action =
        # [L joint1-6, L gripper, R joint1-6, R gripper]). Regular pi0.5 (NO MoH). Same
        # recipe as pi05_piperx_place_mug_joint / pi05_piperx_pour_beans_joint:
        # LeRobotPiperXBimanualDataConfig joint recipe (PiperXInputs/Outputs, delta arm
        # joints + absolute grippers, mask 6,-1,6,-1), fine-tune from pi05_base. Wrist-only
        # dataset (cam_left_wrist + cam_right_wrist, no front) so use_front_camera=False is
        # REQUIRED. No wrist-cam augmentation -- cameras stay consistent between training
        # data and inference. Distinct asset_id so norm stats do not collide with the other
        # bimanual joint runs.
        # Run `compute_norm_stats.py --config-name pi05_piperx_remove_mug_joint` first.
        name="pi05_piperx_remove_mug_joint",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_remove_mug_joint",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_remove_mug_joint_bimanual"),
            default_prompt="remove mug from coffee machine",
            use_front_camera=False,
        ),
        # Extended from 10k to 15k: the first 10k run's loss was not fully converged.
        # decay_steps bumped 9400 -> 14400 (warmup 600 + 14400 = 15000) so the cosine
        # anneals over the full 15k instead of hitting the 5e-6 floor at 10k. Resuming
        # from the 10k checkpoint (--resume) gives a warm restart (~1.7e-5 at step 10k)
        # decaying back to 5e-6 by 15k. A fresh-from-base run of this config now trains a
        # clean 15k cosine.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=14400,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        # 8x H200 machine -- crank data loading so it does not bottleneck the GPUs.
        num_workers=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_place_mug_joint_v2 -- a larger,
        # already-combined recording of the "place mug into machine" task (robot_type
        # piperx_bimanual_joint; 68 eps / 54520 frames @ 30fps; 14-D state/action =
        # [L joint1-6, L gripper, R joint1-6, R gripper]). Regular pi0.5 (NO MoH),
        # fine-tuned fresh from pi05_base (not warm-started from the v1 place_mug policy).
        # Same recipe as pi05_piperx_place_mug_joint: LeRobotPiperXBimanualDataConfig joint
        # recipe (PiperXInputs/Outputs, delta arm joints + absolute grippers, mask
        # 6,-1,6,-1). Wrist-only dataset (cam_left_wrist + cam_right_wrist, no front) so
        # use_front_camera=False is REQUIRED. No wrist-cam augmentation -- cameras stay
        # consistent between training data and inference. Distinct asset_id so norm stats
        # are computed over the v2 set and do not collide with the v1 run.
        # Run `compute_norm_stats.py --config-name pi05_piperx_place_mug_joint_v2` first.
        name="pi05_piperx_place_mug_joint_v2",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_place_mug_joint_v2",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_place_mug_joint_v2_bimanual"),
            default_prompt="place mug into machine",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        # 8x H200 machine -- crank data loading so it does not bottleneck the GPUs.
        num_workers=32,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_tamping_joint -- a bimanual
        # "grind coffee beans and then tamp" task recorded as JOINT ANGLES (robot_type
        # piperx_bimanual_joint; 41 eps / 52151 frames @ 30fps; 14-D state/action =
        # [L joint1-6, L gripper, R joint1-6, R gripper]). Regular pi0.5 (NO MoH). Same
        # recipe as the other single-task joint runs (pi05_piperx_place_mug_joint etc.):
        # LeRobotPiperXBimanualDataConfig joint recipe (PiperXInputs/Outputs, delta arm
        # joints + absolute grippers, mask 6,-1,6,-1), fine-tune from pi05_base, 10k steps.
        # Wrist-only dataset (cam_left_wrist + cam_right_wrist, no front) so
        # use_front_camera=False is REQUIRED. No wrist-cam augmentation. Distinct asset_id
        # so norm stats do not collide with the other bimanual joint runs.
        # Run `compute_norm_stats.py --config-name pi05_piperx_tamping_joint` first.
        name="pi05_piperx_tamping_joint",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_tamping_joint",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_tamping_joint_bimanual"),
            default_prompt="grind coffee beans and then tamp",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=9400,
            decay_lr=5e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        num_workers=2,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),

    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_4_coffee -- a large MULTI-TASK
        # combined coffee dataset (robot_type piperx_bimanual_joint; 194 eps / 171876
        # frames @ 30fps; 14-D state/action). FOUR tasks in one dataset: "place mug into
        # machine", "remove mug from coffee machine", "pour the beans into the coffee
        # machine", "grind coffee beans and then tamp" -- so prompt_from_task=True is
        # essential (each episode carries its own task string) and there is deliberately
        # NO single default_prompt. Regular pi0.5 (NO MoH), fine-tune from pi05_base.
        # Same LeRobotPiperXBimanualDataConfig joint recipe as the single-task runs, but
        # trained LONGER (30k steps) because this dataset is ~4x larger and multi-task:
        # 10k would be < 2 epochs. Wrist-only (cam_left_wrist + cam_right_wrist, no front)
        # so use_front_camera=False is REQUIRED. No wrist-cam augmentation. Distinct
        # asset_id so norm stats are computed over the full multi-task set.
        # Run `compute_norm_stats.py --config-name pi05_piperx_4_coffee` first.
        name="pi05_piperx_4_coffee",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_4_coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_4_coffee_bimanual"),
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=29000,
            decay_lr=5e-6,
        ),
        num_train_steps=30000,
        batch_size=32,
        num_workers=2,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_filter_joint -- a bimanual
        # "put the handle in the coffee machine" (portafilter) task recorded as JOINT
        # ANGLES (robot_type piperx_bimanual_joint; 80 eps / 120519 frames @ 30fps; 14-D
        # state/action = [L joint1-6, L gripper, R joint1-6, R gripper]). Regular pi0.5
        # (NO MoH), fine-tune from pi05_base. Same LeRobotPiperXBimanualDataConfig joint
        # recipe as the other single-task joint runs (PiperXInputs/Outputs, delta arm
        # joints + absolute grippers, mask 6,-1,6,-1). Wrist-only (cam_left_wrist +
        # cam_right_wrist, no front) so use_front_camera=False is REQUIRED. No wrist-cam
        # augmentation -- cameras stay consistent between training data and inference.
        # Trained 15k (not the usual 10k): at 120k frames, 10k steps is only ~2.6 epochs,
        # so decay_steps is 14400 (warmup 600 + 14400 = 15000) for a full cosine over 15k.
        # Distinct asset_id so norm stats do not collide with the other joint runs.
        # Run `compute_norm_stats.py --config-name pi05_piperx_filter_joint` first.
        name="pi05_piperx_filter_joint",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_filter_joint",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_filter_joint_bimanual"),
            default_prompt="put the handle in the coffee machine",
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=5e-5,
            decay_steps=14400,
            decay_lr=5e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        num_workers=2,
        log_interval=100,
        save_interval=2000,
        keep_period=2000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    TrainConfig(
        # Bimanual PiperX, JOINT space, on axiboai/piper_5_coffee -- a large MULTI-TASK
        # combined coffee dataset (robot_type piperx_bimanual_joint; 274 eps / 292395
        # frames @ 30fps; 14-D state/action). FIVE tasks in one dataset: "put the handle
        # in the coffee machine", "place mug into machine", "remove mug from coffee
        # machine", "grind coffee beans and then tamp", "pour the beans into the coffee
        # machine" -- i.e. the four piper_4_coffee tasks plus the filter/handle task. So
        # prompt_from_task=True is essential (each episode carries its own task string) and
        # there is deliberately NO single default_prompt. Regular pi0.5 (NO MoH), fine-tune
        # from pi05_base. Same LeRobotPiperXBimanualDataConfig joint recipe as the other
        # runs. Trained 50k steps: ~1.7x more data than piper_4_coffee (which ran 30k), so
        # 50k keeps a comparable ~5.5 epochs (10k would be < 1.1 epochs). Wrist-only
        # (cam_left_wrist + cam_right_wrist, no front) so use_front_camera=False is
        # REQUIRED. No wrist-cam augmentation. Distinct asset_id so norm stats are computed
        # over the full 5-task set.
        # Run `compute_norm_stats.py --config-name pi05_piperx_5_coffee` first.
        name="pi05_piperx_5_coffee",
        model=pi0_config.Pi0Config(pi05=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=LeRobotPiperXBimanualDataConfig(
            repo_id="axiboai/piper_5_coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="piperx_5_coffee_bimanual"),
            use_front_camera=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=49000,
            decay_lr=5e-6,
        ),
        num_train_steps=50000,
        batch_size=32,
        num_workers=2,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "action_space": "joint",
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
        },
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]


# --- Diffusion-backbone study configs -------------------------------------
# Registered here, after get_config/_CONFIGS_DICT are defined, because these
# configs are built by forking `pi05_libero` via get_config (which is not
# available while the _CONFIGS literal above is still being constructed).
# Only implemented variants are returned (Model A today; B/C/D as they land).
from openpi.diffusion_backbone import train_config as _db_train_config  # noqa: E402

for _db_cfg in _db_train_config.get_diffusion_backbone_configs():
    if _db_cfg.name in _CONFIGS_DICT:
        raise ValueError(f"Duplicate config name from diffusion_backbone: {_db_cfg.name}")
    _CONFIGS.append(_db_cfg)
    _CONFIGS_DICT[_db_cfg.name] = _db_cfg
