"""Load and validate the frozen FairJob Stage1.1 experiment protocols."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_TASK_PROTOCOLS = Path("configs/fairjob/task_protocols.yaml")
DEFAULT_FAIRNESS_PROTOCOLS = Path("configs/fairjob/fairness_protocols.yaml")


def load_yaml(path: str | Path) -> dict:
    """Read a YAML mapping and reject empty or non-mapping documents."""

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol file must contain a mapping: {path}")
    return payload


def load_protocols(
    task_path: str | Path = DEFAULT_TASK_PROTOCOLS,
    fairness_path: str | Path = DEFAULT_FAIRNESS_PROTOCOLS,
) -> tuple[dict, dict]:
    """Return validated task and fairness protocol documents."""

    task = load_yaml(task_path)
    fairness = load_yaml(fairness_path)
    if task.get("version") != 1 or fairness.get("version") != 1:
        raise ValueError("Unsupported FairJob protocol version.")
    if task.get("primary_protocol") not in task.get("protocols", {}):
        raise ValueError("primary_protocol is not defined in protocols.")
    if not fairness.get("proxy_regimes"):
        raise ValueError("At least one proxy regime is required.")
    return task, fairness


def resolve_regime(fairness: dict, regime: str) -> tuple[str, dict]:
    """Resolve either a scientific or legacy regime name.

    The returned key always uses underscores for config identifiers, while the
    associated ``scientific_name`` is the label that belongs in reports.
    """

    normalized = regime.replace("-", "_")
    regimes = fairness["proxy_regimes"]
    if normalized in regimes:
        return normalized, regimes[normalized]
    for key, spec in regimes.items():
        if regime in {spec.get("legacy_name"), spec.get("scientific_name")}:
            return key, spec
    raise KeyError(f"Unknown proxy regime: {regime}")


def features_for_protocol(
    manifest: dict,
    task: dict,
    fairness: dict,
    protocol: str,
    regime: str,
) -> tuple[list[str], str, str]:
    """Derive model features from the prepared manifest and frozen protocols.

    Evaluator metadata is never considered here. In particular, excluding the
    protected proxy from model input does not remove it from metadata used for
    disparity and utility evaluation.
    """

    if protocol not in task["protocols"]:
        raise KeyError(f"Unknown task protocol: {protocol}")
    regime_key, regime_spec = resolve_regime(fairness, regime)
    protocol_spec = task["protocols"][protocol]
    excluded = set(protocol_spec.get("excluded_features", []))

    features = list(manifest["unaware_features"])
    features = [name for name in features if name not in excluded]
    if regime_spec["include_protected_proxy"]:
        features.append(manifest["protected_feature_for_model"])

    if not protocol_spec.get("allow_impression_id", False):
        features = [name for name in features if name != "impression_id"]
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate features in {protocol}/{regime_key}.")

    return features, protocol_spec["scientific_name"], regime_spec["scientific_name"]


def scientific_regime_name(regime: str, fairness: dict | None = None) -> str:
    """Map legacy report labels to the Stage1.1 scientific terminology."""

    if fairness is None:
        _, fairness = load_protocols()
    _, spec = resolve_regime(fairness, regime)
    return spec["scientific_name"]
