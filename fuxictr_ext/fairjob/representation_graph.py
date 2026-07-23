"""Stage2 representation-graph definitions and forward-output validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


DEFAULT_GRAPH_CONFIG = Path("configs/fairjob/stage2_representation_graph.yaml")


def load_representation_graph(path: str | Path = DEFAULT_GRAPH_CONFIG) -> dict:
    """Load and validate the frozen Stage2 representation graph."""

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage2 representation graph version.")
    if payload.get("model_structure") != "parallel":
        raise ValueError("Stage2 currently supports only parallel DCNv2.")
    nodes = payload.get("nodes", {})
    targets = payload.get("target_nodes", [])
    sentinels = payload.get("sentinel_nodes", [])
    missing = sorted(set(targets + sentinels).difference(nodes))
    if missing:
        raise ValueError(f"Graph roles reference unknown nodes: {missing}")
    overlap = sorted(set(targets).intersection(sentinels))
    if overlap:
        raise ValueError(f"Graph nodes cannot be target and sentinel: {overlap}")
    for name, spec in nodes.items():
        unknown_parents = sorted(
            set(spec.get("parents", [])).difference(nodes).difference({"model_inputs"})
        )
        if unknown_parents:
            raise ValueError(f"Node {name} has unknown parents: {unknown_parents}")
        alias = spec.get("alias_of")
        if alias is not None and alias not in nodes:
            raise ValueError(f"Node {name} aliases unknown node {alias}.")
    for name, spec in payload.get("joint_paths", {}).items():
        members = spec.get("members", [])
        if len(members) < 2:
            raise ValueError(f"Joint path {name} requires at least two members.")
        known = set(nodes).union(payload.get("legacy_nodes", {}))
        unknown = sorted(set(members).difference(known))
        if unknown:
            raise ValueError(f"Joint path {name} has unknown members: {unknown}")
    return payload


def graph_hash(graph: dict) -> str:
    """Return a stable hash for manifests and checkpoint provenance."""

    encoded = json.dumps(
        graph, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_graph_representations(
    representations: dict[str, torch.Tensor],
    graph: dict,
    include_legacy: bool = False,
    tolerance: float = 1e-7,
) -> dict:
    """Validate required nodes, row alignment, finiteness, and declared aliases."""

    required = set(graph["nodes"])
    if include_legacy:
        required.update(graph.get("legacy_nodes", {}))
    missing = sorted(required.difference(representations))
    if missing:
        raise ValueError(f"Forward output is missing graph nodes: {missing}")
    rows = {int(representations[name].shape[0]) for name in required}
    if len(rows) != 1:
        raise ValueError("Representation graph nodes have different row counts.")
    for name in required:
        value = representations[name]
        if value.ndim < 2:
            raise ValueError(f"Graph node {name} must retain a feature dimension.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Graph node {name} contains non-finite values.")
    alias_differences = {}
    for name, spec in graph["nodes"].items():
        alias = spec.get("alias_of")
        if alias is None:
            continue
        difference = float(
            torch.max(torch.abs(representations[name] - representations[alias]))
            .detach()
            .cpu()
        )
        alias_differences[f"{name}={alias}"] = difference
        if difference > tolerance:
            raise ValueError(
                f"Graph alias {name} differs from {alias}: {difference}"
            )
    return {
        "rows": rows.pop(),
        "node_shapes": {
            name: list(representations[name].shape[1:])
            for name in sorted(required)
        },
        "alias_max_abs_differences": alias_differences,
        "graph_hash": graph_hash(graph),
    }


def joint_representation(
    representations: dict[str, torch.Tensor],
    graph: dict,
    joint_name: str,
) -> torch.Tensor:
    """Concatenate one preregistered joint path in its frozen member order."""

    try:
        members = graph["joint_paths"][joint_name]["members"]
    except KeyError as error:
        raise KeyError(f"Unknown Stage2 joint path: {joint_name}") from error
    missing = [name for name in members if name not in representations]
    if missing:
        raise ValueError(
            f"Joint path {joint_name} is missing representations: {missing}"
        )
    rows = {int(representations[name].shape[0]) for name in members}
    if len(rows) != 1:
        raise ValueError(f"Joint path {joint_name} has misaligned row counts.")
    return torch.cat([representations[name] for name in members], dim=-1)


def build_joint_representations(
    representations: dict[str, torch.Tensor],
    graph: dict,
    include_legacy: bool = False,
) -> dict[str, torch.Tensor]:
    """Build all target joint paths, optionally including legacy controls."""

    result = {}
    for name, spec in graph["joint_paths"].items():
        if spec.get("kind") == "legacy_control" and not include_legacy:
            continue
        result[f"joint_{name}"] = joint_representation(
            representations, graph, name
        )
    return result
