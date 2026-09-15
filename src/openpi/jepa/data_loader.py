"""LeRobot loader with multi-horizon image stacks for JEPA."""

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


def _jepa_image_delta_timestamps(
    *,
    fps: float,
    horizon_steps: tuple[int, ...],
    use_front_camera: bool,
) -> dict[str, list[float]]:
    keys = [
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    if use_front_camera:
        keys = ["observation.images.cam_front", *keys]
    deltas = [0.0] + [h / fps for h in horizon_steps]
    return {key: deltas for key in keys}


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    jepa_horizon_steps: tuple[int, ...],
    use_front_camera: bool = True,
) -> _data_loader.Dataset:
    """Like ``openpi.training.data_loader.create_torch_dataset`` with multi-horizon RGB."""
    from openpi.jepa.pi0_jepa_config import Pi0JepaConfig

    if not isinstance(model_config, Pi0JepaConfig):
        raise TypeError("JEPA dataset helper expects Pi0JepaConfig")

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    delta_timestamps.update(
        _jepa_image_delta_timestamps(
            fps=dataset_meta.fps,
            horizon_steps=jepa_horizon_steps,
            use_front_camera=use_front_camera,
        )
    )

    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_timestamps)

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset


def patch_create_torch_dataset() -> None:
    """Install JEPA-aware dataset creation for ``train_pi05_jepa.py``."""
    original = _data_loader.create_torch_dataset

    def wrapped(
        data_config: _config.DataConfig,
        action_horizon: int,
        model_config: _model.BaseModelConfig,
    ) -> _data_loader.Dataset:
        from openpi.jepa.pi0_jepa_config import Pi0JepaConfig

        if isinstance(model_config, Pi0JepaConfig):
            return create_torch_dataset(
                data_config,
                action_horizon,
                model_config,
                jepa_horizon_steps=model_config.jepa_horizon_steps,
                use_front_camera=model_config.use_front_camera,
            )
        return original(data_config, action_horizon, model_config)

    _data_loader.create_torch_dataset = wrapped
