"""FairJob representation adapters for native FuxiCTR CTR models.

Training deliberately uses the inherited model implementations. The additional
method only exposes named intermediate tensors for offline Stage1.1 diagnosis,
which keeps optimizer, loss, initialization, and checkpoint behavior identical
to the corresponding FuxiCTR model-zoo class.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

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


class _GradientReverse(torch.autograd.Function):
    """Pass values unchanged and reverse gradients flowing to the encoder."""

    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def _gradient_reverse(inputs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(inputs, scale)


class FairJobMitigatedDCNv2(FairJobDCNv2):
    """DCNv2 adapter for the frozen M5A suppression and fairness baselines.

    All methods retain the native DCNv2 encoder and click loss. Suppression
    methods intervene only on ``dcnv2_final`` before the native output layer.
    DP and adversarial penalties consume raw meta aliases that are excluded by
    ``BaseModel.get_inputs()``, preserving the proxy-excluded input contract.
    """

    METHODS = {
        "baseline",
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    }

    def __init__(
        self,
        feature_map,
        mitigation_method="baseline",
        suppression_fraction=0.1,
        suppression_seed_offset=100003,
        selective_indices_config="configs/fairjob/m5_selective_indices.yaml",
        fairness_weight=0.1,
        adversarial_weight=0.1,
        adversary_hidden_units=64,
        gradient_reversal_scale=1.0,
        learning_rate=1e-3,
        seed=2019,
        **kwargs,
    ):
        if mitigation_method not in self.METHODS:
            raise ValueError(f"Unsupported M5 mitigation method: {mitigation_method}")
        self.mitigation_method = mitigation_method
        self.suppression_fraction = float(suppression_fraction)
        self.training_seed = int(seed)
        self.fairness_weight = float(fairness_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.gradient_reversal_scale = float(gradient_reversal_scale)
        super().__init__(
            feature_map,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )

        final_dim = self.fc.in_features
        mask = torch.ones(final_dim, dtype=torch.float32)
        self.suppressed_indices = []
        if mitigation_method == "global_suppression":
            if self.model_structure != "parallel":
                raise ValueError("M5 global suppression requires parallel DCNv2.")
            # The parallel final vector starts with the full cross path. Removing
            # that path is the declared global interaction-suppression control.
            cross_dim = feature_map.sum_emb_out_dim()
            self.suppressed_indices = list(range(cross_dim))
        elif mitigation_method == "matched_random_suppression":
            count = max(1, int(round(final_dim * self.suppression_fraction)))
            generator = torch.Generator().manual_seed(
                self.training_seed + int(suppression_seed_offset)
            )
            self.suppressed_indices = torch.randperm(
                final_dim, generator=generator
            )[:count].tolist()
        elif mitigation_method == "selective_suppression":
            config_path = Path(selective_indices_config)
            if not config_path.is_absolute():
                config_path = Path(__file__).resolve().parents[2] / config_path
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if int(payload["dimensions"]) != final_dim:
                raise ValueError(
                    f"Selective mask dimension {payload['dimensions']} != {final_dim}."
                )
            try:
                self.suppressed_indices = payload["indices_by_seed"][str(self.training_seed)]
            except KeyError as error:
                raise ValueError(
                    f"No frozen selective indices for seed {self.training_seed}."
                ) from error
        if self.suppressed_indices:
            if min(self.suppressed_indices) < 0 or max(self.suppressed_indices) >= final_dim:
                raise ValueError("Suppression indices exceed the final representation.")
            mask[self.suppressed_indices] = 0.0
        self.register_buffer("suppression_mask", mask)

        if mitigation_method == "adversarial":
            self.adversary = nn.Sequential(
                nn.Linear(final_dim, int(adversary_hidden_units)),
                nn.ReLU(),
                nn.Linear(int(adversary_hidden_units), 1),
            )
            for layer in self.adversary:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            # DCNv2 compiles before subclass modules exist. Recompile once so
            # the adversary parameters join the same optimizer transaction.
            self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.model_to_device()

    def _fairness_meta(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw protected proxy and senior-scope tensors on model device."""

        required = {"protected_attribute_meta", "senior_meta"}
        missing = sorted(required.difference(inputs))
        if missing:
            raise ValueError(f"M5 dataset is missing fairness meta columns: {missing}")
        protected = inputs["protected_attribute_meta"].to(self.device).float().view(-1, 1)
        senior = inputs["senior_meta"].to(self.device).float().view(-1, 1) > 0.5
        return protected, senior

    def forward_with_representations(self, inputs):
        """Apply the frozen intervention and expose pre/post final states."""

        diagnostic = super().forward_with_representations(inputs)
        pre_mitigation = diagnostic["representations"]["dcnv2_final"]
        final_out = pre_mitigation * self.suppression_mask
        logit = self.fc(final_out)
        diagnostic["y_pred"] = self.output_activation(logit)
        diagnostic["representations"]["dcnv2_final_pre_mitigation"] = pre_mitigation
        diagnostic["representations"]["dcnv2_final"] = final_out
        diagnostic["representations"]["dcnv2_logit"] = logit
        return diagnostic

    def forward(self, inputs):
        """Return native click prediction plus method-specific training terms."""

        diagnostic = self.forward_with_representations(inputs)
        result = {"y_pred": diagnostic["y_pred"]}
        if self.mitigation_method == "dp_regularization":
            protected, senior = self._fairness_meta(inputs)
            group_0 = senior & (protected < 0.5)
            group_1 = senior & (protected >= 0.5)
            if group_0.any() and group_1.any():
                difference = (
                    result["y_pred"][group_1].mean()
                    - result["y_pred"][group_0].mean()
                )
                result["fairness_penalty"] = difference.square()
            else:
                result["fairness_penalty"] = result["y_pred"].sum() * 0.0
        elif self.mitigation_method == "adversarial":
            protected, senior = self._fairness_meta(inputs)
            final_out = diagnostic["representations"]["dcnv2_final"]
            result["adversary_logit"] = self.adversary(
                _gradient_reverse(final_out, self.gradient_reversal_scale)
            )
            result["adversary_target"] = protected
            result["adversary_scope"] = senior
        return result

    def compute_loss(self, return_dict, y_true):
        """Add the declared mitigation penalty to FuxiCTR's native CTR loss."""

        loss = super().compute_loss(return_dict, y_true)
        if self.mitigation_method == "dp_regularization":
            loss = loss + self.fairness_weight * return_dict["fairness_penalty"]
        elif self.mitigation_method == "adversarial":
            scope = return_dict["adversary_scope"].view(-1)
            if scope.any():
                adversary_loss = F.binary_cross_entropy_with_logits(
                    return_dict["adversary_logit"].view(-1)[scope],
                    return_dict["adversary_target"].view(-1)[scope],
                )
                loss = loss + self.adversarial_weight * adversary_loss
        return loss


FAIRJOB_MODELS = {
    "FairJobDeepFM": FairJobDeepFM,
    "FairJobDCNv2": FairJobDCNv2,
    "FairJobMitigatedDCNv2": FairJobMitigatedDCNv2,
}


def resolve_model_class(model_name: str):
    """Resolve a FairJob adapter without changing ``model_zoo.__init__``."""

    if model_name not in FAIRJOB_MODELS:
        raise KeyError(f"Unknown FairJob model adapter: {model_name}")
    return FAIRJOB_MODELS[model_name]
