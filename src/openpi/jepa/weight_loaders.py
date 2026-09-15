"""Weight loaders for π₀.₅ + JEPA (merge pi05_base, keep new jepa_pred)."""

import dataclasses

import numpy as np

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as weight_loaders

_JEPA_MISSING_REGEX = ".*(?:lora|jepa_pred|jepa_mlp|jepa_pool_proj).*"


@dataclasses.dataclass(frozen=True)
class Pi05JepaCheckpointWeightLoader:
    """Load ``pi05_base`` and keep ``jepa_pred`` from init when missing."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        return weight_loaders._merge_params(loaded_params, params, missing_regex=_JEPA_MISSING_REGEX)
