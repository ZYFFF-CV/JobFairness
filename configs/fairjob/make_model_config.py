"""Generate Stage1.1 DeepFM/DCNv2 entries in the FairJob model config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


GENERATED_PREFIXES = ("DeepFM_fairjob_", "DCNv2_fairjob_", "M5DCNv2_")
PRIMARY_PROTOCOLS = ("pre_ranking", "post_display")
IDENTITY_CONTROL_PROTOCOLS = (
    "pre_ranking_no_user_id",
    "pre_ranking_no_product_id",
    "pre_ranking_no_identity_ids",
)
M5_METHODS = (
    "baseline",
    "global_suppression",
    "matched_random_suppression",
    "selective_suppression",
    "dp_regularization",
    "adversarial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="configs/fairjob/model_config.yaml")
    return parser.parse_args()


def common_config(model: str, dataset_id: str, mode: str) -> dict:
    """Return settings shared by the two diagnostic backbones."""

    return {
        "model": f"FairJob{model}",
        "dataset_id": dataset_id,
        "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"],
        "task": "binary_classification",
        "optimizer": "adam",
        "learning_rate": 1.0e-3,
        "embedding_regularizer": 1.0e-8,
        "net_regularizer": 0,
        "batch_size": 1024 if mode == "smoke" else 4096,
        "embedding_dim": 16,
        "epochs": 1 if mode == "smoke" else 20,
        "shuffle": True,
        "seed": 2019,
        "monitor": "AUC",
        "monitor_mode": "max",
    }


def model_config(model: str, dataset_id: str, mode: str) -> dict:
    """Build a model-zoo-aligned configuration for one experiment."""

    config = common_config(model, dataset_id, mode)
    if model == "DeepFM":
        config.update(
            {
                "hidden_units": [256, 128, 64],
                "hidden_activations": "relu",
                "batch_norm": False,
                "net_dropout": 0.1,
            }
        )
    elif model == "DCNv2":
        config.update(
            {
                "model_structure": "parallel",
                "use_low_rank_mixture": False,
                "low_rank": 32,
                "num_experts": 4,
                "stacked_dnn_hidden_units": [256, 128],
                "parallel_dnn_hidden_units": [256, 128],
                "dnn_activations": "relu",
                "num_cross_layers": 3,
                "net_dropout": 0.1,
                "batch_norm": False,
            }
        )
    else:
        raise KeyError(f"Unsupported Stage1.1 model: {model}")
    return config


def generate_entries() -> dict[str, dict]:
    """Generate the complete task/proxy/mode matrix for both backbones."""

    entries = {}
    for model in ("DeepFM", "DCNv2"):
        for protocol in PRIMARY_PROTOCOLS:
            for regime in ("proxy_excluded", "proxy_included"):
                for mode in ("smoke", "full"):
                    suffix = f"fairjob_{protocol}_{regime}_{mode}"
                    entries[f"{model}_{suffix}"] = model_config(model, suffix, mode)
        # Identity interventions answer the proxy-excluded M3 mechanism
        # question; proxy-included variants would be redundant positive controls.
        for protocol in IDENTITY_CONTROL_PROTOCOLS:
            for mode in ("smoke", "full"):
                suffix = f"fairjob_{protocol}_proxy_excluded_{mode}"
                entries[f"{model}_{suffix}"] = model_config(model, suffix, mode)
    m5_dataset_prefix = "fairjob_m5_pre_ranking_no_user_id_proxy_excluded"
    for method in M5_METHODS:
        for mode in ("smoke", "full"):
            dataset_id = f"{m5_dataset_prefix}_{mode}"
            expid = f"M5DCNv2_{method}_{mode}"
            config = model_config("DCNv2", dataset_id, mode)
            config.update(
                {
                    "model": "FairJobMitigatedDCNv2",
                    "mitigation_method": method,
                    "suppression_fraction": 0.1,
                    "suppression_seed_offset": 100003,
                    "selective_indices_config": "configs/fairjob/m5_selective_indices.yaml",
                    "fairness_weight": 0.1,
                    "adversarial_weight": 0.1,
                    "adversary_hidden_units": 64,
                    "gradient_reversal_scale": 1.0,
                }
            )
            entries[expid] = config
    return entries


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    config = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    config = {
        key: value
        for key, value in config.items()
        if not key.startswith(GENERATED_PREFIXES)
    }
    config.update(generate_entries())
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out} experiments={len(config) - 1}")


if __name__ == "__main__":
    main()
