"""Frozen probe suites for Stage2.1 AUC and conditional-CE endpoints."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from fuxictr_ext.fairjob.joint_path_probe import (
    FAMILIES,
    _candidate_models,
)


def _oriented_metrics(y_true, probability, orientation: int, epsilon: float):
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if orientation not in (-1, 1):
        raise ValueError("Probe orientation must be -1 or 1.")
    if orientation < 0:
        probability = 1.0 - probability
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "cross_entropy": float(log_loss(y_true, probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, probability >= 0.5)
        ),
    }


def _convergence(model) -> tuple[int | None, bool]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "n_iter_"):
        return None, True
    n_iter = int(np.max(np.atleast_1d(estimator.n_iter_)))
    return n_iter, n_iter < estimator.max_iter


def _matches_fixed(params: dict, fixed: dict) -> bool:
    normalized = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }
    return normalized == fixed


def fit_probe_suite(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    protocol: dict,
    fixed_capacity: dict,
    seed: int,
    epsilon: float = 1e-7,
) -> dict:
    """Fit frozen families and read test only for preregistered candidates.

    Every family produces three endpoints: validation-AUC-selected,
    validation-cross-entropy-selected, and one frozen fixed-capacity candidate.
    Candidate validation metrics are retained to make selection auditable; test
    metrics are never computed for unselected candidates.
    """

    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and 0.5.")
    result = {}
    candidates = list(_candidate_models(protocol, seed))
    for family in FAMILIES:
        family_candidates = []
        for current_family, params, model in candidates:
            if current_family != family:
                continue
            model.fit(X_train, y_train)
            valid_probability = model.predict_proba(X_valid)[:, 1]
            raw_valid_auc = float(roc_auc_score(y_valid, valid_probability))
            orientation = 1 if raw_valid_auc >= 0.5 else -1
            valid_metrics = _oriented_metrics(
                y_valid, valid_probability, orientation, epsilon
            )
            n_iter, converged = _convergence(model)
            family_candidates.append(
                {
                    "params": params,
                    "model": model,
                    "orientation": orientation,
                    "raw_valid_auc": raw_valid_auc,
                    "valid_auc": valid_metrics["auc"],
                    "valid_cross_entropy": valid_metrics["cross_entropy"],
                    "valid_balanced_accuracy": valid_metrics[
                        "balanced_accuracy"
                    ],
                    "n_iter": n_iter,
                    "converged": converged,
                }
            )
        if not family_candidates:
            raise ValueError(f"No frozen candidates exist for family {family}.")

        auc_selected = max(
            family_candidates,
            key=lambda item: (
                item["valid_auc"],
                json.dumps(item["params"], sort_keys=True),
            ),
        )
        ce_selected = min(
            family_candidates,
            key=lambda item: (
                item["valid_cross_entropy"],
                json.dumps(item["params"], sort_keys=True),
            ),
        )
        fixed_spec = fixed_capacity[family]
        fixed_matches = [
            item
            for item in family_candidates
            if _matches_fixed(item["params"], fixed_spec)
        ]
        if len(fixed_matches) != 1:
            raise ValueError(
                f"Fixed capacity for {family} matched {len(fixed_matches)} candidates."
            )
        fixed_selected = fixed_matches[0]

        selected_payloads = {}
        test_cache = {}
        for role, selected in (
            ("auc_selected", auc_selected),
            ("ce_selected", ce_selected),
            ("fixed_capacity", fixed_selected),
        ):
            cache_key = json.dumps(selected["params"], sort_keys=True)
            if cache_key not in test_cache:
                probability = selected["model"].predict_proba(X_test)[:, 1]
                test_cache[cache_key] = _oriented_metrics(
                    y_test,
                    probability,
                    selected["orientation"],
                    epsilon,
                )
            selected_payloads[role] = {
                key: value
                for key, value in selected.items()
                if key != "model"
            }
            selected_payloads[role].update(
                {
                    f"test_{key}": value
                    for key, value in test_cache[cache_key].items()
                }
            )

        result[family] = {
            **selected_payloads,
            "candidate_validation": [
                {key: value for key, value in item.items() if key != "model"}
                for item in family_candidates
            ],
        }
    return {
        "families": result,
        "epsilon": epsilon,
        "selection": {
            "auc": "validation_maximum_oriented_auc",
            "cross_entropy": "validation_minimum_oriented_cross_entropy",
            "fixed_capacity": fixed_capacity,
            "test_candidate_reads_per_family_maximum": 3,
        },
    }


def conditional_gain(parent_suite: dict, combined_suite: dict) -> dict:
    """Compute family-specific held-out CE gain from parent to parent+child."""

    result = {}
    for family in FAMILIES:
        parent = parent_suite["families"][family]
        combined = combined_suite["families"][family]
        result[family] = {}
        for role in ("ce_selected", "fixed_capacity"):
            parent_ce = float(parent[role]["test_cross_entropy"])
            combined_ce = float(combined[role]["test_cross_entropy"])
            result[family][role] = {
                "parent_test_cross_entropy": parent_ce,
                "combined_test_cross_entropy": combined_ce,
                "conditional_predictive_gain": parent_ce - combined_ce,
            }
    return result


def joint_gain(
    left_suite: dict,
    right_suite: dict,
    joint_suite: dict,
) -> dict:
    """Compute gain of a joint representation over its better single branch."""

    result = {}
    for family in FAMILIES:
        result[family] = {}
        for role in ("ce_selected", "fixed_capacity"):
            left_ce = float(
                left_suite["families"][family][role]["test_cross_entropy"]
            )
            right_ce = float(
                right_suite["families"][family][role]["test_cross_entropy"]
            )
            joint_ce = float(
                joint_suite["families"][family][role]["test_cross_entropy"]
            )
            result[family][role] = {
                "left_test_cross_entropy": left_ce,
                "right_test_cross_entropy": right_ce,
                "joint_test_cross_entropy": joint_ce,
                "joint_predictive_gain": min(left_ce, right_ce) - joint_ce,
            }
    return result
