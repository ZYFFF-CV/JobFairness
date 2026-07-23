"""FairJob representation adapters for native FuxiCTR CTR models.

Training deliberately uses the inherited model implementations. The additional
method only exposes named intermediate tensors for offline Stage1.1 diagnosis,
which keeps optimizer, loss, initialization, and checkpoint behavior identical
to the corresponding FuxiCTR model-zoo class.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model_zoo.DCNv2.src.DCNv2 import DCNv2
from model_zoo.DeepFM.DeepFM_torch.src.DeepFM import DeepFM
from fuxictr_ext.fairjob.adaptive_constraints import GraphRiskAggregator
from fuxictr_ext.fairjob.multinode_adversary import MultiNodeAdversary
from fuxictr_ext.fairjob.structured_path_gates import StructuredPathGates


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
            increment = X_0 * layer(X_i)
            X_i = X_i + increment
            states[f"cross_increment_{index}"] = increment
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
            representations["cross_path_output"] = cross_out
            representations["parallel_dnn_path_output"] = parallel_out
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
        representations["fusion_pre_logit"] = final_out
        representations["dcnv2_final"] = final_out
        representations["dcnv2_logit"] = logit
        representations["dcnv2_probability"] = y_pred
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
        diagnostic["representations"]["dcnv2_suppressed_component"] = (
            pre_mitigation - final_out
        )
        diagnostic["representations"]["dcnv2_residual_component"] = final_out
        diagnostic["representations"]["dcnv2_logit"] = logit
        diagnostic["representations"]["dcnv2_probability"] = diagnostic["y_pred"]
        return diagnostic

    def representation_export_metadata(self) -> dict:
        """Describe the fixed M5 intervention without repeating masks per row."""

        mask = self.suppression_mask.detach().cpu().numpy()
        return {
            "mitigation_method": self.mitigation_method,
            "training_seed": self.training_seed,
            "final_dimensions": int(mask.size),
            "suppressed_dimensions": int((mask == 0).sum()),
            "suppressed_indices": [int(index) for index in self.suppressed_indices],
            "suppression_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        }

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


class FairJobGraphContainmentDCNv2(FairJobDCNv2):
    """Graph-aware Stage2 containment model over native parallel DCNv2 paths.

    The CTR backbone, click loss, optimizer lifecycle, and checkpoint format
    remain those of FuxiCTR's DCNv2. Stage2 adds train-only proxy adversaries
    and scalar gates at architecture-named blocks. Raw proxy metadata is never
    passed to ``get_inputs`` and is not required during evaluation or inference.
    """

    METHODS = {
        "baseline",
        "final_only_no_gate",
        "multi_layer_no_gate",
        "final_only_path_gate",
        "multi_layer_path_gate_full",
        "full_without_joint_adversary",
        "layer_risk_sum",
        "structured_random_gate",
        "matched_capacity_regularization",
    }
    GATED_METHODS = {
        "final_only_path_gate": "learned",
        "multi_layer_path_gate_full": "learned",
        "full_without_joint_adversary": "learned",
        "layer_risk_sum": "learned",
        "structured_random_gate": "frozen_random",
    }
    JOINT_METHODS = {
        "multi_layer_path_gate_full",
        "layer_risk_sum",
        "structured_random_gate",
        "matched_capacity_regularization",
    }

    def __init__(
        self,
        feature_map,
        stage2_method="baseline",
        stage2_adversary_nodes=None,
        stage2_joint_paths=None,
        stage2_adversary_hidden_units=(64,),
        stage2_adversarial_weight=0.1,
        stage2_gradient_reversal_scale=1.0,
        stage2_graph_risk_mode="smooth_max_proxy_advantage",
        stage2_graph_risk_temperature=0.05,
        stage2_cvar_fraction=0.25,
        stage2_gate_budget=0.1,
        stage2_gate_minimum_keep=0.5,
        stage2_gate_temperature=1.0,
        stage2_gate_seed_offset=200003,
        stage2_centered_logit_weight=0.05,
        stage2_pairwise_ranking_weight=0.05,
        stage2_pairwise_margin=0.05,
        stage2_backbone_checkpoint=None,
        stage2_require_baseline_teacher=False,
        stage2_log_interval=50,
        learning_rate=1e-3,
        seed=2019,
        **kwargs,
    ):
        if stage2_method not in self.METHODS:
            raise ValueError(f"Unsupported Stage2 method: {stage2_method}")
        self.stage2_method = stage2_method
        self.stage2_adversarial_weight = float(stage2_adversarial_weight)
        self.stage2_centered_logit_weight = float(stage2_centered_logit_weight)
        self.stage2_pairwise_ranking_weight = float(
            stage2_pairwise_ranking_weight
        )
        self.stage2_pairwise_margin = float(stage2_pairwise_margin)
        self.stage2_log_interval = int(stage2_log_interval)
        self.training_seed = int(seed)
        super().__init__(
            feature_map,
            learning_rate=learning_rate,
            seed=seed,
            **kwargs,
        )
        if self.model_structure != "parallel":
            raise ValueError("Stage2 graph containment requires parallel DCNv2.")
        if not hasattr(self.crossnet, "cross_layers"):
            raise ValueError("Stage2 requires the explicit CrossNetV2 layer graph.")

        self.stage2_backbone_checkpoint = stage2_backbone_checkpoint
        self.teacher_backbone = None
        if (
            stage2_require_baseline_teacher
            and stage2_method != "baseline"
            and not stage2_backbone_checkpoint
        ):
            raise ValueError(
                "Formal Stage2 runs require --backbone_checkpoint from the "
                "same seed baseline."
            )
        if stage2_backbone_checkpoint:
            checkpoint = Path(stage2_backbone_checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Stage2 baseline checkpoint not found: {checkpoint}"
                )
            state_dict = torch.load(checkpoint, map_location="cpu")
            current_state = self.state_dict()
            missing = sorted(set(current_state).difference(state_dict))
            mismatched = sorted(
                name
                for name in current_state.keys() & state_dict.keys()
                if current_state[name].shape != state_dict[name].shape
            )
            if missing or mismatched:
                raise ValueError(
                    "Stage2 baseline checkpoint is incompatible: "
                    f"missing={missing}, mismatched={mismatched}"
                )
            # A Stage1.1 baseline may contain intervention metadata buffers.
            # Only exact native-backbone keys are loaded into the Stage2 model.
            self.load_state_dict(
                {name: state_dict[name] for name in current_state},
                strict=True,
            )
            # The teacher is a frozen snapshot of the same-seed trained
            # baseline. Keeping only its forward modules avoids a second
            # optimizer/checkpoint lifecycle.
            self.teacher_backbone = nn.ModuleDict(
                {
                    "embedding_layer": copy.deepcopy(self.embedding_layer),
                    "crossnet": copy.deepcopy(self.crossnet),
                    "parallel_dnn": copy.deepcopy(self.parallel_dnn),
                    "fc": copy.deepcopy(self.fc),
                }
            )
            self.teacher_backbone.requires_grad_(False)
            self.teacher_backbone.eval()

        cross_count = len(self.crossnet.cross_layers)
        dnn_linear_dims = [
            layer.out_features
            for layer in self.parallel_dnn.mlp
            if isinstance(layer, nn.Linear)
        ]
        if not dnn_linear_dims:
            raise ValueError("Stage2 requires at least one parallel DNN block.")
        input_dim = feature_map.sum_emb_out_dim()
        self._stage2_node_dims = {
            **{
                f"cross_layer_{index}": input_dim
                for index in range(cross_count)
            },
            "parallel_dnn_path_output": dnn_linear_dims[-1],
            "fusion_pre_logit": self.fc.in_features,
        }
        configured_nodes = list(
            stage2_adversary_nodes
            or (
                [f"cross_layer_{index}" for index in range(cross_count)]
                + ["parallel_dnn_path_output", "fusion_pre_logit"]
            )
        )
        unknown_nodes = sorted(
            set(configured_nodes).difference(self._stage2_node_dims)
        )
        if unknown_nodes:
            raise ValueError(f"Unknown Stage2 adversary nodes: {unknown_nodes}")
        self.stage2_adversary_nodes = configured_nodes

        self.stage2_joint_paths = dict(
            stage2_joint_paths
            or {
                "cross_dnn_outputs": [
                    f"cross_layer_{cross_count - 1}",
                    "parallel_dnn_path_output",
                ]
            }
        )
        for name, members in self.stage2_joint_paths.items():
            unknown = sorted(set(members).difference(self._stage2_node_dims))
            if unknown:
                raise ValueError(
                    f"Stage2 joint path {name} has unknown members: {unknown}"
                )

        gate_names = [
            *[f"cross_increment_{index}" for index in range(cross_count)],
            *[
                f"parallel_dnn_linear_{index}"
                for index in range(len(dnn_linear_dims))
            ],
            "cross_path_output",
            "parallel_dnn_path_output",
        ]
        gate_mode = self.GATED_METHODS.get(stage2_method, "identity")
        self.path_gates = StructuredPathGates(
            gate_names,
            mode=gate_mode,
            suppression_budget_per_gate=float(stage2_gate_budget),
            minimum_keep=float(stage2_gate_minimum_keep),
            temperature=float(stage2_gate_temperature),
            seed=int(seed) + int(stage2_gate_seed_offset),
        )

        adversary_dims = self._selected_adversary_dims()
        self.proxy_adversary = None
        self.graph_risk = None
        if adversary_dims:
            reversal_scale = (
                0.0
                if stage2_method == "matched_capacity_regularization"
                else float(stage2_gradient_reversal_scale)
            )
            self.proxy_adversary = MultiNodeAdversary(
                adversary_dims,
                hidden_units=tuple(stage2_adversary_hidden_units),
                gradient_reversal_scale=reversal_scale,
            )
            risk_mode = (
                "mean_bce"
                if stage2_method == "layer_risk_sum"
                else stage2_graph_risk_mode
            )
            self.graph_risk = GraphRiskAggregator(
                mode=risk_mode,
                temperature=float(stage2_graph_risk_temperature),
                cvar_fraction=float(stage2_cvar_fraction),
            )
            # Native DCNv2 compiles before Stage2 modules exist. Recompiling
            # enrolls gates and adversaries in the same optimizer transaction.
            self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        elif gate_mode == "learned":
            self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self._stage2_last_telemetry = {}
        self.model_to_device()

    def _selected_adversary_dims(self) -> dict[str, int]:
        """Resolve method-specific graph targets before optimizer compilation."""

        if self.stage2_method == "baseline":
            return {}
        if self.stage2_method in {
            "final_only_no_gate",
            "final_only_path_gate",
        }:
            return {"fusion_pre_logit": self._stage2_node_dims["fusion_pre_logit"]}
        dimensions = {
            name: self._stage2_node_dims[name]
            for name in self.stage2_adversary_nodes
        }
        if self.stage2_method in self.JOINT_METHODS:
            for name, members in self.stage2_joint_paths.items():
                dimensions[f"joint_{name}"] = sum(
                    self._stage2_node_dims[member] for member in members
                )
        return dimensions

    def _cross_path(self, feature_emb: torch.Tensor) -> tuple:
        """Run CrossNetV2 while gating whole residual increments."""

        X_0 = feature_emb
        X_i = X_0
        states = {}
        for index, layer in enumerate(self.crossnet.cross_layers):
            raw_increment = X_0 * layer(X_i)
            increment = self.path_gates.apply(
                f"cross_increment_{index}", raw_increment
            )
            X_i = X_i + increment
            states[f"cross_increment_{index}"] = increment
            states[f"cross_layer_{index}"] = X_i
        output = self.path_gates.apply("cross_path_output", X_i)
        states["cross_path_output"] = output
        return output, states

    def _parallel_path(self, feature_emb: torch.Tensor) -> tuple:
        """Run the native MLP and gate outputs of architecture-named linears."""

        value = feature_emb
        states = {}
        linear_index = 0
        for layer in self.parallel_dnn.mlp:
            value = layer(value)
            if isinstance(layer, nn.Linear):
                name = f"parallel_dnn_linear_{linear_index}"
                value = self.path_gates.apply(name, value)
                states[name] = value
                linear_index += 1
        value = self.path_gates.apply("parallel_dnn_path_output", value)
        states["parallel_dnn_path_output"] = value
        return value, states

    def _joint_representations(self, representations: dict) -> dict:
        """Build preregistered aligned joint paths for train-only adversaries."""

        return {
            f"joint_{name}": torch.cat(
                [representations[member] for member in members], dim=-1
            )
            for name, members in self.stage2_joint_paths.items()
        }

    def forward_with_representations(self, inputs):
        """Return gated predictions and the real named DCNv2 computation graph."""

        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out, cross_states = self._cross_path(feature_emb)
        dnn_out, dnn_states = self._parallel_path(feature_emb)
        final_out = torch.cat([cross_out, dnn_out], dim=-1)
        logit = self.fc(final_out)
        y_pred = self.output_activation(logit)
        representations = {
            "embedding_flat": feature_emb,
            **cross_states,
            **dnn_states,
            "fusion_pre_logit": final_out,
            "dcnv2_final": final_out,
            "dcnv2_logit": logit,
            "dcnv2_probability": y_pred,
        }
        return {"y_pred": y_pred, "representations": representations}

    def _frozen_teacher_logit(self, inputs) -> torch.Tensor:
        """Return logits from the same-seed frozen baseline checkpoint."""

        if self.teacher_backbone is None:
            raise RuntimeError("No frozen Stage2 baseline teacher is available.")
        with torch.no_grad():
            X = self.get_inputs(inputs)
            feature_emb = self.teacher_backbone["embedding_layer"](
                X, flatten_emb=True
            )
            cross_out = self.teacher_backbone["crossnet"](feature_emb)
            dnn_out = self.teacher_backbone["parallel_dnn"](feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
            return self.teacher_backbone["fc"](final_out).detach()

    def forward(self, inputs):
        """Return click prediction and train-only containment objective inputs."""

        diagnostic = self.forward_with_representations(inputs)
        result = {
            "y_pred": diagnostic["y_pred"],
            "stage2_representations": diagnostic["representations"],
        }
        if not self.training:
            return result

        if self.proxy_adversary is not None:
            if "protected_attribute_meta" not in inputs:
                raise ValueError(
                    "Stage2 training requires raw protected_attribute_meta."
                )
            target = (
                inputs["protected_attribute_meta"]
                .to(self.device)
                .float()
                .view(-1, 1)
            )
            adversary_inputs = {
                name: diagnostic["representations"][name]
                for name in self.stage2_adversary_nodes
                if name in self.proxy_adversary.input_dims
            }
            adversary_inputs.update(
                {
                    name: value
                    for name, value in self._joint_representations(
                        diagnostic["representations"]
                    ).items()
                    if name in self.proxy_adversary.input_dims
                }
            )
            result["proxy_logits"] = self.proxy_adversary(adversary_inputs)
            result["proxy_target"] = target

        if self.teacher_backbone is not None:
            result["frozen_teacher_logit"] = self._frozen_teacher_logit(inputs)
        return result

    def _preservation_losses(self, return_dict: dict) -> tuple:
        """Penalize score collapse on a teacher-normalized logit scale."""

        logit = return_dict["stage2_representations"]["dcnv2_logit"]
        teacher = return_dict.get("frozen_teacher_logit")
        connected_zero = logit.sum() * 0.0
        if teacher is None:
            return connected_zero, connected_zero, None
        centered = logit - logit.mean()
        teacher_centered = teacher - teacher.mean()
        # CrossNet logits can span hundreds or thousands even when sigmoid
        # probabilities remain finite. Scaling by the frozen teacher's batch
        # standard deviation keeps the preservation weight comparable across
        # batches and seeds; SmoothL1 prevents a few extreme rows from taking
        # over the graph-containment objective.
        teacher_scale = teacher_centered.detach().std(unbiased=False).clamp_min(
            0.1
        )
        centered_loss = F.smooth_l1_loss(
            centered / teacher_scale,
            teacher_centered / teacher_scale,
        )

        if len(logit) < 2:
            return centered_loss, connected_zero, teacher_scale
        pair_rows = (len(logit) // 2) * 2
        student_pairs = logit[:pair_rows].reshape(-1, 2)
        teacher_pairs = teacher[:pair_rows].reshape(-1, 2)
        student_difference = (
            student_pairs[:, 1] - student_pairs[:, 0]
        ) / teacher_scale
        teacher_difference = (
            teacher_pairs[:, 1] - teacher_pairs[:, 0]
        ) / teacher_scale
        direction = torch.sign(teacher_difference)
        informative = direction != 0
        if informative.any():
            ranking_loss = F.relu(
                self.stage2_pairwise_margin
                - student_difference[informative] * direction[informative]
            ).mean()
        else:
            ranking_loss = connected_zero
        return centered_loss, ranking_loss, teacher_scale

    def compute_loss(self, return_dict, y_true):
        """Combine native CTR, graph containment, and anti-collapse objectives."""

        click_loss = super().compute_loss(return_dict, y_true)
        total_loss = click_loss
        telemetry = {"click_loss": float(click_loss.detach().cpu())}
        if self.proxy_adversary is not None:
            node_losses = self.proxy_adversary.losses(
                return_dict["proxy_logits"], return_dict["proxy_target"]
            )
            graph_loss, risk_telemetry = self.graph_risk(node_losses)
            total_loss = total_loss + self.stage2_adversarial_weight * graph_loss
            telemetry.update(
                {
                    "graph_loss": float(graph_loss.detach().cpu()),
                    "graph_risk": risk_telemetry,
                }
            )

        centered_loss, ranking_loss, teacher_scale = self._preservation_losses(
            return_dict
        )
        total_loss = (
            total_loss
            + self.stage2_centered_logit_weight * centered_loss
            + self.stage2_pairwise_ranking_weight * ranking_loss
        )
        probability = return_dict["y_pred"].detach().view(-1)
        logit = return_dict["stage2_representations"]["dcnv2_logit"].detach().view(-1)
        label = y_true.detach().view(-1)
        positive = label > 0.5
        negative = ~positive
        class_gap = None
        if positive.any() and negative.any():
            class_gap = float(
                (probability[positive].mean() - probability[negative].mean())
                .cpu()
            )
        telemetry.update(
            {
                "centered_logit_loss": float(centered_loss.detach().cpu()),
                "pairwise_ranking_loss": float(ranking_loss.detach().cpu()),
                "teacher_logit_scale": (
                    float(teacher_scale.detach().cpu())
                    if teacher_scale is not None
                    else None
                ),
                "score_mean": float(probability.mean().cpu()),
                "score_std": float(probability.std(unbiased=False).cpu()),
                "logit_dynamic_range": float(
                    (logit.max() - logit.min()).cpu()
                ),
                "class_score_gap": class_gap,
                "gates": self.path_gates.telemetry(),
                "total_loss": float(total_loss.detach().cpu()),
            }
        )
        self._stage2_last_telemetry = telemetry
        return total_loss

    def train_step(self, batch_data):
        """Run one native optimizer step and emit bounded live Stage2 telemetry."""

        loss = super().train_step(batch_data)
        step = int(getattr(self, "_total_steps", 0))
        if (
            self.stage2_log_interval > 0
            and step > 0
            and step % self.stage2_log_interval == 0
        ):
            logging.info(
                "Stage2 telemetry step=%d %s",
                step,
                json.dumps(
                    self._stage2_last_telemetry,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        return loss

    def train(self, mode: bool = True):
        """Keep the frozen baseline teacher deterministic during student fit."""

        super().train(mode)
        if self.teacher_backbone is not None:
            self.teacher_backbone.eval()
        return self

    def representation_export_metadata(self) -> dict:
        """Describe graph targets and gates without exposing proxy values."""

        return {
            "stage2_method": self.stage2_method,
            "training_seed": self.training_seed,
            "adversary_nodes": list(self.stage2_adversary_nodes),
            "joint_paths": self.stage2_joint_paths,
            "gate_definition_sha256": self.path_gates.definition_hash(),
            "gate_telemetry": self.path_gates.telemetry(),
            "frozen_baseline_teacher": self.teacher_backbone is not None,
            "baseline_checkpoint": (
                str(self.stage2_backbone_checkpoint)
                if self.stage2_backbone_checkpoint
                else None
            ),
            "inference_proxy_dependency": False,
        }


FAIRJOB_MODELS = {
    "FairJobDeepFM": FairJobDeepFM,
    "FairJobDCNv2": FairJobDCNv2,
    "FairJobMitigatedDCNv2": FairJobMitigatedDCNv2,
    "FairJobGraphContainmentDCNv2": FairJobGraphContainmentDCNv2,
}


def resolve_model_class(model_name: str):
    """Resolve a FairJob adapter without changing ``model_zoo.__init__``."""

    if model_name not in FAIRJOB_MODELS:
        raise KeyError(f"Unknown FairJob model adapter: {model_name}")
    return FAIRJOB_MODELS[model_name]
