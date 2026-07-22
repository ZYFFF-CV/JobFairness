"""Sharded export and validation of FairJob model representations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REQUIRED_REPRESENTATION_META = ["row_id", "click", "protected_attribute"]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_2d_float(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim < 2:
        raise ValueError(f"Representation {name} must have a row dimension.")
    array = array.reshape(array.shape[0], -1).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"Representation {name} contains non-finite values.")
    return array


class RepresentationShardWriter:
    """Buffer aligned arrays and write deterministic NPZ shards."""

    def __init__(self, out_dir: str | Path, shard_rows: int = 10000):
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive.")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = shard_rows
        self.buffers: dict[str, list[np.ndarray]] = {}
        self.buffered_rows = 0
        self.total_rows = 0
        self.shards = []

    def add(self, arrays: dict[str, np.ndarray]) -> None:
        """Append one aligned batch and flush complete shards."""

        normalized = {name: np.asarray(value) for name, value in arrays.items()}
        lengths = {value.shape[0] for value in normalized.values()}
        if len(lengths) != 1:
            raise ValueError("Representation batch arrays have different row counts.")
        rows = lengths.pop()
        if rows == 0:
            return
        if self.buffers and set(normalized) != set(self.buffers):
            raise ValueError("Representation keys changed between batches.")
        for name, value in normalized.items():
            self.buffers.setdefault(name, []).append(value)
        self.buffered_rows += rows
        while self.buffered_rows >= self.shard_rows:
            self._flush(self.shard_rows)

    def _flush(self, rows: int) -> None:
        merged = {name: np.concatenate(parts, axis=0) for name, parts in self.buffers.items()}
        shard = {name: value[:rows] for name, value in merged.items()}
        remainder = {name: value[rows:] for name, value in merged.items()}
        shard_index = len(self.shards)
        path = self.out_dir / f"part-{shard_index:05d}.npz"
        np.savez(path, **shard)
        self.shards.append(
            {
                "path": path.name,
                "rows": rows,
                "row_start": self.total_rows,
                "row_end": self.total_rows + rows,
                "sha256": sha256_file(path),
            }
        )
        self.total_rows += rows
        self.buffered_rows -= rows
        self.buffers = {
            name: [value] for name, value in remainder.items() if len(value) > 0
        }

    def close(self) -> list[dict]:
        """Write the final partial shard and return shard metadata."""

        if self.buffered_rows:
            self._flush(self.buffered_rows)
        return self.shards


def export_representations(
    model,
    data_generator,
    meta_path: str | Path,
    out_dir: str | Path,
    split: str,
    shard_rows: int = 10000,
    max_rows: int | None = None,
    sample_seed: int = 2019,
) -> dict:
    """Export one split while aligning every tensor to raw FairJob metadata."""

    meta = pd.read_csv(meta_path)
    missing = [name for name in REQUIRED_REPRESENTATION_META if name not in meta.columns]
    if missing:
        raise ValueError(f"Representation metadata is missing columns: {missing}")
    writer = RepresentationShardWriter(Path(out_dir) / split, shard_rows=shard_rows)
    representation_shapes = None
    offset = 0
    selected_positions = None
    if max_rows is not None and len(meta) > max_rows:
        rng = np.random.default_rng(sample_seed)
        selected_positions = np.sort(rng.choice(len(meta), size=max_rows, replace=False))
    selected_offset = 0

    model.eval()
    with torch.no_grad():
        for batch_data in data_generator:
            output = model.forward_with_representations(batch_data)
            y_pred = output["y_pred"].detach().cpu().numpy().reshape(-1)
            rows = len(y_pred)
            batch_meta = meta.iloc[offset : offset + rows]
            if len(batch_meta) != rows:
                raise ValueError("Model output has more rows than representation metadata.")

            representations = {
                name: _as_2d_float(name, tensor.detach().cpu().numpy())
                for name, tensor in output["representations"].items()
            }
            current_shapes = {name: list(value.shape[1:]) for name, value in representations.items()}
            if representation_shapes is None:
                representation_shapes = current_shapes
            elif representation_shapes != current_shapes:
                raise ValueError("Representation shapes changed between batches.")

            if selected_positions is None:
                batch_indices = np.arange(rows)
            else:
                left = np.searchsorted(selected_positions, offset, side="left")
                right = np.searchsorted(selected_positions, offset + rows, side="left")
                batch_indices = selected_positions[left:right] - offset
            selected_meta = batch_meta.iloc[batch_indices]
            writer.add(
                {
                    "row_id": selected_meta["row_id"].to_numpy(dtype=np.int64),
                    "y_true": selected_meta["click"].to_numpy(dtype=np.float32),
                    "protected_attribute": selected_meta["protected_attribute"].to_numpy(
                        dtype=np.int8
                    ),
                    "y_pred": y_pred[batch_indices].astype(np.float32),
                    **{
                        name: value[batch_indices]
                        for name, value in representations.items()
                    },
                }
            )
            selected_offset += len(batch_indices)
            offset += rows

    if offset != len(meta):
        raise ValueError(f"Representation rows {offset} do not match metadata rows {len(meta)}.")
    shards = writer.close()
    manifest = {
        "version": 1,
        "split": split,
        "rows": selected_offset,
        "source_rows": offset,
        "meta_path": str(meta_path),
        "sampling": {
            "method": "uniform_without_replacement" if selected_positions is not None else "all_rows",
            "max_rows": max_rows,
            "seed": sample_seed,
        },
        "representation_shapes": representation_shapes or {},
        "shards": shards,
    }
    metadata_fn = getattr(model, "representation_export_metadata", None)
    if metadata_fn is not None:
        manifest["model_representation_metadata"] = metadata_fn()
    manifest_path = Path(out_dir) / split / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_exported_predictions(
    split_dir: str | Path,
    reference_prediction: str | Path,
    tolerance: float = 1e-6,
) -> dict:
    """Compare sampled exported probabilities with a canonical prediction CSV.

    Stage1.2 reuses completed checkpoints. This check makes checkpoint loading,
    feature preprocessing, and representation hooks auditable by requiring the
    exported test probabilities to reproduce the original M5A predictions on
    exactly the exported row IDs.
    """

    if tolerance < 0:
        raise ValueError("Prediction tolerance must be non-negative.")
    split_dir = Path(split_dir)
    manifest = validate_representation_directory(split_dir)
    row_ids = []
    y_true = []
    y_pred = []
    for shard_info in manifest["shards"]:
        with np.load(split_dir / shard_info["path"]) as shard:
            row_ids.append(np.asarray(shard["row_id"], dtype=np.int64))
            y_true.append(np.asarray(shard["y_true"], dtype=np.float64))
            y_pred.append(np.asarray(shard["y_pred"], dtype=np.float64))
    exported_row_ids = np.concatenate(row_ids) if row_ids else np.empty(0, dtype=np.int64)
    exported_y_true = np.concatenate(y_true) if y_true else np.empty(0, dtype=np.float64)
    exported_y_pred = np.concatenate(y_pred) if y_pred else np.empty(0, dtype=np.float64)

    reference = pd.read_csv(reference_prediction, usecols=["row_id", "y_true", "y_pred"])
    if reference["row_id"].duplicated().any():
        raise ValueError("Reference prediction contains duplicate row IDs.")
    aligned = reference.set_index("row_id").reindex(exported_row_ids)
    if aligned.isna().any().any():
        raise ValueError("Exported row IDs are missing from the reference prediction.")
    reference_y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    reference_y_pred = aligned["y_pred"].to_numpy(dtype=np.float64)
    if not np.array_equal(exported_y_true, reference_y_true):
        raise ValueError("Exported labels differ from the reference prediction.")
    differences = np.abs(exported_y_pred - reference_y_pred)
    max_abs_diff = float(differences.max()) if len(differences) else 0.0
    if max_abs_diff > tolerance:
        raise ValueError(
            f"Representation export changed predictions: max_abs_diff={max_abs_diff}"
        )
    return {
        "rows": int(len(exported_row_ids)),
        "reference_prediction": str(reference_prediction),
        "tolerance": tolerance,
        "max_abs_prediction_diff": max_abs_diff,
    }


def validate_representation_directory(path: str | Path) -> dict:
    """Validate shard hashes, schemas, finiteness, and contiguous row IDs."""

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    expected_keys = None
    previous_row_id = None
    total_rows = 0
    for shard_info in manifest["shards"]:
        shard_path = path / shard_info["path"]
        if sha256_file(shard_path) != shard_info["sha256"]:
            raise ValueError(f"Representation shard hash mismatch: {shard_path}")
        with np.load(shard_path) as shard:
            keys = set(shard.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError("Representation shard schemas do not match.")
            rows = len(shard["row_id"])
            row_ids = np.asarray(shard["row_id"], dtype=np.int64)
            if rows > 1 and not np.all(np.diff(row_ids) > 0):
                raise ValueError(f"row_id must be strictly increasing in {shard_path}")
            if previous_row_id is not None and rows and row_ids[0] <= previous_row_id:
                raise ValueError("row_id order overlaps between representation shards.")
            for name in shard.files:
                if not np.isfinite(shard[name]).all():
                    raise ValueError(f"Non-finite values in {shard_path}:{name}")
            if rows:
                previous_row_id = int(row_ids[-1])
            total_rows += rows
    if total_rows != manifest["rows"]:
        raise ValueError("Representation manifest row count does not match shards.")
    return manifest


def load_representation_row_ids(
    path: str | Path, validate_shards: bool = True
) -> tuple[dict, np.ndarray]:
    """Return a representation manifest and its ordered split-local row IDs.

    Full shard validation is deliberately optional here. M3 validates each
    representation set once while creating its frozen probe sample, then layer
    probes reuse the same manifest without repeatedly hashing large NPZ files.
    """

    path = Path(path)
    if validate_shards:
        manifest = validate_representation_directory(path)
    else:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    row_ids = []
    for shard_info in manifest["shards"]:
        with np.load(path / shard_info["path"]) as shard:
            row_ids.append(np.asarray(shard["row_id"], dtype=np.int64))
    combined = np.concatenate(row_ids) if row_ids else np.empty(0, dtype=np.int64)
    if len(combined) != manifest["rows"]:
        raise ValueError(f"Representation row count does not match manifest: {path}")
    if len(combined) > 1 and not np.all(np.diff(combined) > 0):
        raise ValueError(f"Representation row IDs are not strictly increasing: {path}")
    return manifest, combined
