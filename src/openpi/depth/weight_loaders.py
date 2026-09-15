"""Weight loaders for π₀.₅ + depth (merge pi05_base, keep new depth_enc)."""

import dataclasses

import numpy as np

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as weight_loaders

# Keys in the train model that are not in pi05_base (random init, trained from scratch).
_DEPTH_MISSING_REGEX = ".*(?:lora|depth_enc|depth_proj).*"


@dataclasses.dataclass(frozen=True)
class Pi05DepthCheckpointWeightLoader:
    """Load ``pi05_base`` and keep ``depth_enc`` / ``depth_proj`` from init when missing."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        return weight_loaders._merge_params(loaded_params, params, missing_regex=_DEPTH_MISSING_REGEX)
