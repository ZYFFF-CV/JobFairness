"""FairJob representation adapters for native FuxiCTR CTR models.

Training deliberately uses the inherited model implementations. The additional
method only exposes named intermediate tensors for offline Stage1.1 diagnosis,
which keeps optimizer, loss, initialization, and checkpoint behavior identical
to the corresponding FuxiCTR model-zoo class.
"""

from __future__ import annotations

import torch
from torch import nn

from model_zoo.DCNv2.src.DCNv2 import DCNv2
from model_zoo.DeepFM.DeepFM_torch.src.DeepFM import DeepFM


def _mlp_linear_states(block: nn.Module, inputs: torch.Tensor, prefix: str) -> tuple:
    """Run an FuxiCTR MLP block and capture outputs of its linear layers."""

    value = inputs
    states = {}
    linear_index = 0
    for layer in block.mlp:
        value = layer(value)
        if isinstance(layer, nn.Linear):
            states[f"{prefix}_{linear_index}"] = value
            linear_index += 1
    return value, states


class FairJobDeepFM(DeepFM):
    """Native DeepFM with a side-effect-free representation export method."""

    def forward_with_representations(self, inputs):
        """Return click probability and named DeepFM path representations."""

        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        embedding_flat = feature_emb.flatten(start_dim=1)
        fm_logit = self.fm(X, feature_emb)
        dnn_logit, dnn_states = _mlp_linear_states(
            self.mlp, embedding_flat, "dnn_linear"
        )
        logit = fm_logit + dnn_logit
        y_pred = self.output_activation(logit)
        representations = {
            "embedding_flat": embedding_flat,
            "fm_logit": fm_logit,
            **dnn_states,
            "deepfm_logit": logit,
        }
        return {"y_pred": y_pred, "representations": representations}


class FairJobDCNv2(DCNv2):
    """Native DCNv2 with cross-layer and DNN-path representation export."""

    def _cross_states(self, feature_emb: torch.Tensor) -> tuple:
        """Expose CrossNetV2 layer states while preserving native equations."""

        if not hasattr(self.crossnet, "cross_layers"):
            return self.crossnet(feature_emb), {}
        X_0 = feature_emb
        X_i = X_0
        states = {}
        for index, layer in enumerate(self.crossnet.cross_layers):
            X_i = X_i + X_0 * layer(X_i)
            states[f"cross_layer_{index}"] = X_i
        return X_i, states

    def forward_with_representations(self, inputs):
        """Return click probability and named DCNv2 path representations."""

        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out, cross_states = self._cross_states(feature_emb)
        representations = {"embedding_flat": feature_emb, **cross_states}

        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            stacked_out, stacked_states = _mlp_linear_states(
                self.stacked_dnn, cross_out, "stacked_dnn_linear"
            )
            representations.update(stacked_states)
            final_out = stacked_out
        elif self.model_structure == "parallel":
            parallel_out, parallel_states = _mlp_linear_states(
                self.parallel_dnn, feature_emb, "parallel_dnn_linear"
            )
            representations.update(parallel_states)
            final_out = torch.cat([cross_out, parallel_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            stacked_out, stacked_states = _mlp_linear_states(
                self.stacked_dnn, cross_out, "stacked_dnn_linear"
            )
            parallel_out, parallel_states = _mlp_linear_states(
                self.parallel_dnn, feature_emb, "parallel_dnn_linear"
            )
            representations.update(stacked_states)
            representations.update(parallel_states)
            final_out = torch.cat([stacked_out, parallel_out], dim=-1)
        else:
            raise ValueError(f"Unsupported DCNv2 structure: {self.model_structure}")

        logit = self.fc(final_out)
        y_pred = self.output_activation(logit)
        representations["dcnv2_final"] = final_out
        representations["dcnv2_logit"] = logit
        return {"y_pred": y_pred, "representations": representations}


FAIRJOB_MODELS = {
    "FairJobDeepFM": FairJobDeepFM,
    "FairJobDCNv2": FairJobDCNv2,
}


def resolve_model_class(model_name: str):
    """Resolve a FairJob adapter without changing ``model_zoo.__init__``."""

    if model_name not in FAIRJOB_MODELS:
        raise KeyError(f"Unknown FairJob model adapter: {model_name}")
    return FAIRJOB_MODELS[model_name]
