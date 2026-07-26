"""Audit one Stage2 checkpoint without persisting large representations.

The P3 audit reconstructs representations from the exact P2 checkpoint on the
frozen row sample, keeps them in memory for one model, and writes only compact
probe results. This preserves checkpoint reproducibility while avoiding a
multi-model representation export that would exceed the server storage budget.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_config

from fuxictr_ext.fairjob.joint_path_probe import fit_stage2_probes
from fuxictr_ext.fairjob.models import resolve_model_class
from fuxictr_ext.fairjob.prepare_probe_samples import load_probe_row_ids
from fuxictr_ext.fairjob.representation_graph import (
    graph_hash,
    load_representation_graph,
)
from fuxictr_ext.fairjob.representation_io import selection_positions, sha256_file
from fuxictr_ext.fairjob.run_manifest import (
    git_commit,
    runtime_versions,
    stable_hash,
    write_run_manifest,
)


SPLITS = ("train", "valid", "test")
PROTOCOL_NAMES = ("row_disjoint", "user_disjoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--graph", default="configs/fairjob/stage2_representation_graph.yaml"
    )
    parser.add_argument(
        "--protocol", default="configs/fairjob/stage2_probe_protocol.yaml"
    )
    parser.add_argument("--row_sample", default=None)
    parser.add_argument("--user_sample", default=None)
    parser.add_argument("--reference_prediction", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_probe_protocol(path: str | Path) -> dict:
    """Load the frozen P3 protocol and reject unsupported revisions."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage2 probe protocol version.")
    if payload.get("selection", {}).get("capacity_and_orientation") != (
        "validation_only"
    ):
        raise ValueError("P3 requires validation-only probe selection.")
    return payload


def build_target_plan(graph: dict, protocol: dict) -> dict[str, dict]:
    """Return all declared node and joint targets in preregistered order.

    Exact graph aliases remain visible in the result schema but reuse their
    canonical probe result. Refitting an identical matrix would add runtime
    without adding an independent scientific observation.
    """

    plan = {}
    for name, spec in graph["nodes"].items():
        plan[name] = {
            "kind": spec["kind"],
            "members": [spec.get("alias_of", name)],
            "alias_of": spec.get("alias_of"),
        }
    for name in protocol["joint_paths"]:
        if name not in graph["joint_paths"]:
            raise ValueError(f"Probe protocol references unknown joint path: {name}")
        spec = graph["joint_paths"][name]
        if spec.get("kind") == "legacy_control":
            raise ValueError(f"P3 cannot probe unavailable legacy path: {name}")
        plan[f"joint_{name}"] = {
            "kind": "joint",
            "members": list(spec["members"]),
            "alias_of": None,
        }
    return plan


def required_representation_names(plan: dict[str, dict]) -> list[str]:
    """Return unique source tensors required by the node and joint plan."""

    names = []
    for spec in plan.values():
        for member in spec["members"]:
            if member not in names:
                names.append(member)
    return names


def load_protocol_rows(
    row_sample: str | Path, user_sample: str | Path
) -> dict[str, dict[str, np.ndarray]]:
    """Load and validate frozen row-disjoint and raw-user-disjoint samples."""

    samples = {
        "row_disjoint": {
            split: load_probe_row_ids(row_sample, split) for split in SPLITS
        },
        "user_disjoint": {
            split: load_probe_row_ids(user_sample, split) for split in SPLITS
        },
    }
    for split in SPLITS:
        row_ids = samples["row_disjoint"][split]
        user_ids = samples["user_disjoint"][split]
        positions = np.searchsorted(row_ids, user_ids)
        valid = positions < len(row_ids)
        if not np.all(valid) or not np.array_equal(
            row_ids[positions[valid]], user_ids[valid]
        ):
            raise ValueError(
                f"User-disjoint rows are not a subset of row-disjoint {split} rows."
            )
    return samples


def _selected_metadata(
    meta_path: str | Path, selected_row_ids: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read raw split metadata and align labels to frozen row IDs."""

    frame = pd.read_csv(
        meta_path,
        usecols=["row_id", "click", "protected_attribute"],
    )
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Split metadata contains duplicate row IDs: {meta_path}")
    positions = selection_positions(
        frame["row_id"].to_numpy(dtype=np.int64), selected_row_ids
    )
    selected = frame.iloc[positions].reset_index(drop=True)
    if not np.array_equal(
        selected["row_id"].to_numpy(dtype=np.int64), selected_row_ids
    ):
        raise ValueError(f"Frozen rows are misaligned with metadata: {meta_path}")
    return selected, frame["row_id"].to_numpy(dtype=np.int64)


def collect_split_representations(
    model,
    data_generator,
    meta_path: str | Path,
    selected_row_ids: np.ndarray,
    required_names: list[str],
) -> dict:
    """Collect selected representations from one ordered FuxiCTR split.

    Batch positions are aligned against raw split metadata. No tokenizer ID is
    used for the protected-proxy target or for row selection.
    """

    selected_meta, source_row_ids = _selected_metadata(
        meta_path, selected_row_ids
    )
    selected_positions = selection_positions(
        source_row_ids, selected_row_ids
    )
    buffers = {name: [] for name in required_names}
    prediction_parts = []
    offset = 0
    selected_count = 0
    shapes = None

    model.eval()
    with torch.no_grad():
        for batch_data in data_generator:
            output = model.forward_with_representations(batch_data)
            y_pred = output["y_pred"].reshape(-1)
            rows = int(y_pred.shape[0])
            representations = output["representations"]
            missing = sorted(set(required_names).difference(representations))
            if missing:
                raise ValueError(f"Checkpoint output is missing representations: {missing}")
            current_shapes = {
                name: list(representations[name].shape[1:])
                for name in required_names
            }
            if shapes is None:
                shapes = current_shapes
            elif shapes != current_shapes:
                raise ValueError("Representation shapes changed between batches.")

            left = np.searchsorted(selected_positions, offset, side="left")
            right = np.searchsorted(
                selected_positions, offset + rows, side="left"
            )
            local_positions = selected_positions[left:right] - offset
            if len(local_positions):
                indices = torch.as_tensor(
                    local_positions, dtype=torch.long, device=y_pred.device
                )
                prediction_parts.append(
                    y_pred.index_select(0, indices)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                for name in required_names:
                    value = representations[name].reshape(rows, -1)
                    buffers[name].append(
                        value.index_select(0, indices)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                selected_count += len(local_positions)
            offset += rows

    if offset != len(source_row_ids):
        raise ValueError(
            f"Model rows {offset} do not match metadata rows {len(source_row_ids)}."
        )
    if selected_count != len(selected_row_ids):
        raise ValueError("Not all frozen probe rows were reconstructed.")

    arrays = {
        name: np.concatenate(parts, axis=0)
        for name, parts in buffers.items()
    }
    if any(len(value) != selected_count for value in arrays.values()):
        raise ValueError("Reconstructed representations are not row-aligned.")
    return {
        "row_id": selected_row_ids,
        "y_true": selected_meta["click"].to_numpy(dtype=np.int8),
        "protected_attribute": selected_meta["protected_attribute"].to_numpy(
            dtype=np.int8
        ),
        "y_pred": np.concatenate(prediction_parts).reshape(-1),
        "representations": arrays,
        "representation_shapes": shapes or {},
        "source_rows": offset,
    }


def protocol_view(
    collected: dict, selected_row_ids: np.ndarray
) -> dict:
    """Select one frozen protocol view from the shared in-memory row sample."""

    positions = selection_positions(collected["row_id"], selected_row_ids)
    return {
        "row_id": collected["row_id"][positions],
        "y_true": collected["y_true"][positions],
        "protected_attribute": collected["protected_attribute"][positions],
        "y_pred": collected["y_pred"][positions],
        "representations": {
            name: value[positions]
            for name, value in collected["representations"].items()
        },
    }


def target_matrix(view: dict, members: list[str]) -> np.ndarray:
    """Build one node or joint-path matrix in frozen member order."""

    arrays = [view["representations"][name] for name in members]
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=1)


def validate_reference_prediction(
    collected_test: dict,
    reference_prediction: str | Path,
    tolerance: float = 1e-6,
) -> dict:
    """Require reconstructed positive-class probabilities to match P2 output."""

    reference = pd.read_csv(
        reference_prediction, usecols=["row_id", "y_true", "y_pred"]
    )
    if reference["row_id"].duplicated().any():
        raise ValueError("Reference prediction contains duplicate row IDs.")
    aligned = reference.set_index("row_id").reindex(collected_test["row_id"])
    if aligned.isna().any().any():
        raise ValueError("Frozen test rows are missing from the P2 prediction.")
    if not np.array_equal(
        aligned["y_true"].to_numpy(dtype=np.int8), collected_test["y_true"]
    ):
        raise ValueError("Reconstructed test labels differ from the P2 prediction.")
    differences = np.abs(
        aligned["y_pred"].to_numpy(dtype=np.float64)
        - collected_test["y_pred"].astype(np.float64)
    )
    max_abs_diff = float(differences.max()) if len(differences) else 0.0
    if max_abs_diff > tolerance:
        raise ValueError(
            "Checkpoint reconstruction changed P2 predictions: "
            f"max_abs_diff={max_abs_diff}"
        )
    return {
        "rows": int(len(differences)),
        "reference_prediction": str(reference_prediction),
        "reference_prediction_sha256": sha256_file(reference_prediction),
        "tolerance": tolerance,
        "max_abs_prediction_diff": max_abs_diff,
    }


def load_resume_payload(path: str | Path, identity: dict) -> dict:
    """Resume target-level work only when every immutable input still matches."""

    path = Path(path)
    if not path.exists():
        return {
            "version": 1,
            "stage": "S2-P3",
            "status": "running",
            **identity,
            "targets": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"Cannot resume P3 result with changed {key}.")
    payload["status"] = "running"
    payload.setdefault("targets", {})
    return payload


def load_model_and_data(args, hparams: dict):
    """Rebuild the exact P2 model/data stack and load its best checkpoint."""

    params = load_config(args.config_dir, args.expid)
    if params["dataset_id"] != args.dataset_id:
        raise ValueError(
            f"Config dataset_id={params['dataset_id']} does not match "
            f"--dataset_id={args.dataset_id}."
        )
    params["gpu"] = args.gpu
    params["seed"] = args.seed
    params["model_root"] = str(Path(args.run_dir) / "checkpoints")
    teacher = hparams.get("hparams", {}).get("stage2_backbone_checkpoint")
    if teacher:
        params["stage2_backbone_checkpoint"] = teacher

    seed_everything(seed=params["seed"])
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
        feature_encoder, **params
    )
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    model = resolve_model_class(params["model"])(feature_map, **params)
    checkpoint = Path(model.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"P2 checkpoint not found: {checkpoint}")
    model.load_weights(str(checkpoint))

    ordered_params = dict(params)
    ordered_params["shuffle"] = False
    generators = RankDataLoader(
        feature_map, stage="both", **ordered_params
    ).make_iterator()
    return model, params, checkpoint, dict(zip(SPLITS, generators))


def execute(args: argparse.Namespace) -> dict:
    """Run the complete dual-protocol audit for one P2 checkpoint."""

    os.chdir(PROJECT_ROOT)
    run_dir = Path(args.run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    hparams_path = run_dir / "hparams.json"
    if not run_manifest_path.exists() or not hparams_path.exists():
        raise FileNotFoundError(f"P2 run metadata is incomplete: {run_dir}")
    training_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("status") != "complete":
        raise ValueError(f"P2 run is not complete: {run_dir}")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))

    graph = load_representation_graph(args.graph)
    protocol = load_probe_protocol(args.protocol)
    row_sample = Path(args.row_sample or protocol["row_disjoint_sample"])
    user_sample = Path(args.user_sample or protocol["user_disjoint_sample"])
    samples = load_protocol_rows(row_sample, user_sample)
    plan = build_target_plan(graph, protocol)
    required_names = required_representation_names(plan)

    model, params, checkpoint, generators = load_model_and_data(args, hparams)
    reference_prediction = Path(
        args.reference_prediction
        or run_dir / "predictions" / f"{args.expid}.csv"
    )
    identity = {
        "audit_git_commit": git_commit(PROJECT_ROOT),
        "training_git_commit": training_manifest["git_commit"],
        "expid": args.expid,
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "probe_seed": args.probe_seed,
        "graph_hash": graph_hash(graph),
        "protocol_hash": stable_hash(protocol),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "row_sample": str(row_sample),
        "row_sample_sha256": sha256_file(row_sample),
        "user_sample": str(user_sample),
        "user_sample_sha256": sha256_file(user_sample),
        "storage_mode": "in_memory_checkpoint_reconstructable",
    }
    payload = load_resume_payload(args.out, identity)
    payload["runtime"] = runtime_versions()
    payload["claim_boundary"] = protocol["claim_boundary"]
    write_run_manifest(args.out, payload)

    collected = {}
    for split in SPLITS:
        print(f"[P3] reconstruct split={split}", flush=True)
        collected[split] = collect_split_representations(
            model=model,
            data_generator=generators[split],
            meta_path=params["fairjob_split_meta_paths"][split],
            selected_row_ids=samples["row_disjoint"][split],
            required_names=required_names,
        )
    payload["prediction_integrity"] = validate_reference_prediction(
        collected["test"], reference_prediction
    )
    payload["source_rows"] = {
        split: collected[split]["source_rows"] for split in SPLITS
    }
    payload["representation_shapes"] = collected["train"][
        "representation_shapes"
    ]
    payload["protocol_rows"] = {
        protocol_name: {
            split: int(len(samples[protocol_name][split])) for split in SPLITS
        }
        for protocol_name in PROTOCOL_NAMES
    }
    views = {
        protocol_name: {
            split: protocol_view(
                collected[split], samples[protocol_name][split]
            )
            for split in SPLITS
        }
        for protocol_name in PROTOCOL_NAMES
    }
    write_run_manifest(args.out, payload)

    for target_name, spec in plan.items():
        if target_name in payload["targets"]:
            print(f"[P3] skip complete target={target_name}", flush=True)
            continue
        if spec["alias_of"]:
            canonical = spec["alias_of"]
            if canonical not in payload["targets"]:
                raise ValueError(
                    f"Alias {target_name} precedes canonical target {canonical}."
                )
            result = copy.deepcopy(payload["targets"][canonical])
            result["alias_of"] = canonical
            result["members"] = spec["members"]
            payload["targets"][target_name] = result
            write_run_manifest(args.out, payload)
            continue

        print(f"[P3] fit target={target_name}", flush=True)
        result = {
            "kind": spec["kind"],
            "members": spec["members"],
            "alias_of": None,
            "protocols": {},
        }
        for protocol_name in PROTOCOL_NAMES:
            print(
                f"[P3] fit target={target_name} "
                f"protocol={protocol_name}",
                flush=True,
            )
            split_views = views[protocol_name]
            matrices = {
                split: target_matrix(split_views[split], spec["members"])
                for split in SPLITS
            }
            labels = {
                split: split_views[split]["protected_attribute"]
                for split in SPLITS
            }
            probes = fit_stage2_probes(
                matrices["train"],
                labels["train"],
                matrices["valid"],
                labels["valid"],
                matrices["test"],
                labels["test"],
                protocol,
                seed=args.probe_seed,
            )
            result["protocols"][protocol_name] = {
                "rows": {
                    split: int(len(labels[split])) for split in SPLITS
                },
                "group_counts": {
                    split: {
                        str(group): int(np.sum(labels[split] == group))
                        for group in (0, 1)
                    }
                    for split in SPLITS
                },
                "dimensions": int(matrices["train"].shape[1]),
                "probes": probes,
            }
            print(
                f"[P3] complete target={target_name} "
                f"protocol={protocol_name}",
                flush=True,
            )
            del matrices
            gc.collect()
        payload["targets"][target_name] = result
        write_run_manifest(args.out, payload)

    payload["status"] = "complete"
    payload["completed_target_count"] = len(payload["targets"])
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
