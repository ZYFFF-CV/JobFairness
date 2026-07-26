import json

from fuxictr_ext.fairjob.stage2_1.build_replication_report import build_report


FAMILIES = (
    "linear",
    "matched_capacity_nonlinear",
    "independent_bounded",
)


def _suite(auc):
    return {
        "max_family_endpoint": {"selected": {"test_auc": auc}},
        "families": {
            family: {
                "auc_selected": {"test_auc": auc},
                "fixed_capacity": {"test_auc": auc},
            }
            for family in FAMILIES
        },
    }


def _audit(seed, values):
    return {
        "status": "complete",
        "audit_git_commit": "audit",
        "stage2_1_protocol_sha256": "protocol",
        "row_sample_sha256": "row",
        "user_sample_sha256": "user",
        "probe_seed": 2019,
        "seed": seed,
        "targets": {
            target: {
                "protocols": {
                    protocol: _suite(auc)
                    for protocol, auc in protocol_values.items()
                }
            }
            for target, protocol_values in values.items()
        },
        "conditional_gains": {},
        "joint_gains": {},
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_replication_gate_pairs_confirmatory_seeds(tmp_path):
    workdir = tmp_path / "stage2_1"
    pilot = tmp_path / "pilot"
    protocol = {
        "seed_roles": {"internal_confirmatory": [2020, 2021]},
        "methods": {"baseline": "baseline", "primary": "primary"},
        "audit_targets": {
            "nodes": ["dnn", "cross"],
            "joint_paths": ["path"],
            "local_reduction_targets": ["dnn"],
            "amplification_targets": ["cross", "joint_path"],
        },
        "material_delta": 0.005,
        "confirmatory_gate": {
            "minimum_common_local_reductions": 1,
            "minimum_common_material_amplifications": 1,
            "minimum_common_user_joint_amplifications": 1,
            "auc_delta_collapse_min": -0.01,
        },
        "claim_boundary": ["internal_only"],
    }
    baseline_values = {
        "dnn": {"row_disjoint": 0.65, "user_disjoint": 0.60},
        "cross": {"row_disjoint": 0.60, "user_disjoint": 0.56},
        "joint_path": {"row_disjoint": 0.61, "user_disjoint": 0.56},
    }
    primary_values = {
        "dnn": {"row_disjoint": 0.63, "user_disjoint": 0.59},
        "cross": {"row_disjoint": 0.62, "user_disjoint": 0.58},
        "joint_path": {"row_disjoint": 0.63, "user_disjoint": 0.58},
    }
    for seed in (2020, 2021):
        for method, values in (
            ("baseline", baseline_values),
            ("primary", primary_values),
        ):
            _write(
                workdir
                / "probes"
                / f"seed{seed}"
                / method
                / "checkpoint_audit.json",
                _audit(seed, values),
            )
            _write(
                pilot / f"seed{seed}" / method / "metrics.json",
                {
                    "AUC": 0.75 if method == "baseline" else 0.76,
                    "NLLH": 0.04,
                    "overall_Brier": 0.007,
                    "DP": 0.001,
                    "U": 0.01,
                    "U_TILDE": 0.012,
                },
            )
    report = build_report(protocol, workdir, pilot)
    assert report["decision"] == "replicated"
    assert report["gate"]["common_local_negative"] == ["dnn"]
    assert "cross" in report["gate"]["common_material_amplified"]
    assert report["next_stage"] == "S21-M4_component_attribution"
