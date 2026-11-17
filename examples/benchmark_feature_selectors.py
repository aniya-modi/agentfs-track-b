"""Benchmark feature-selection methods with repeated nested CV."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif

from agentfs.baseline_selectors import LassoSelector, MRMRSelector
from agentfs.evaluation import kuncheva_stability_index, run_repeated_nested_cv
from agentfs.hypergraph_experiments import BioStatSoftRedundancySelector
from agentfs.kegg import build_kegg_incidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLUB_PATH = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"
ANNOTATION_PATH = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"
RESULTS_DIR = PROJECT_ROOT / "data/results"


def _make_experiments(H: sparse.csr_matrix) -> list[tuple[str, object, dict[str, list[object]]]]:
    """Describe estimators and their inner-CV search spaces."""
    return [
        (
            "Univariate F-score baseline",
            lambda: SelectKBest(score_func=f_classif, k=50),
            {"selector__k": [50, 100], "classifier__C": [0.1, 1.0, 10.0]},
        ),
        (
            "mRMR",
            lambda: MRMRSelector(top_k=50, candidate_pool=500),
            {"selector__top_k": [50, 100], "classifier__C": [0.1, 1.0, 10.0]},
        ),
        (
            "L1 logistic selector",
            lambda: LassoSelector(top_k=50, C=1.0),
            {
                "selector__top_k": [50, 100],
                "selector__C": [0.5, 1.0],
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),
        (
            "Biological/statistical soft redundancy",
            lambda: BioStatSoftRedundancySelector(H=H, top_k=50, alpha=0.5, gamma=0.25),
            {
                "selector__top_k": [50, 100],
                "selector__alpha": [0.25, 0.5, 0.75],
                "selector__gamma": [0.0, 0.25],
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),
    ]


def _fixed_k_grid(param_grid: dict[str, list[object]]) -> dict[str, list[object]]:
    return {
        key: values
        for key, values in param_grid.items()
        if not key.endswith("__k") and not key.endswith("__top_k")
    }


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "agentfs-track-b"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def main() -> None:
    frame = pd.read_csv(GOLUB_PATH, compression="gzip")
    labels = (frame["target"].astype(str) == "AML").astype(int).to_numpy()
    expression = frame.drop(columns=["target"])
    probe_ids = list(expression.columns)
    X = expression.to_numpy(dtype=float)

    print(f"Loaded Golub reference: {X.shape[0]} samples × {X.shape[1]} probes")
    print("Building probe-by-KEGG-pathway incidence matrix...")
    H, kegg_hashes = build_kegg_incidence(probe_ids)
    print(f"H: {H.shape[0]} probes × {H.shape[1]} pathways; {H.nnz} memberships")
    for filename, digest in kegg_hashes.items():
        print(f"KEGG snapshot SHA-256 ({filename}): {digest}")

    summary: list[dict[str, str]] = []
    results: list[dict[str, object]] = []
    for name, selector_factory, grid in _make_experiments(H):
        print(f"Running repeated nested CV for {name}...")
        result = run_repeated_nested_cv(
            X,
            labels,
            selector_factory=selector_factory,
            param_grid=grid,
            n_splits=5,
            n_repeats=3,
            inner_splits=3,
        )
        fixed_k_result = run_repeated_nested_cv(
            X,
            labels,
            selector_factory=selector_factory,
            param_grid=_fixed_k_grid(grid),
            n_splits=5,
            n_repeats=3,
            inner_splits=3,
        )
        kuncheva = kuncheva_stability_index(
            fixed_k_result["selected_indices"], n_total_features=X.shape[1]
        )
        summary.append(
            {
                "Method": name,
                "Accuracy": (
                    f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}"
                ),
                "Balanced accuracy": (
                    f"{result['mean_balanced_accuracy']:.4f} ± "
                    f"{result['std_balanced_accuracy']:.4f}"
                ),
                "Macro F1": f"{result['mean_macro_f1']:.4f} ± {result['std_macro_f1']:.4f}",
                "Jaccard (tuned)": f"{result['jaccard_stability']:.4f}",
                "Kuncheva (fixed k)": f"{kuncheva:.4f}",
            }
        )
        results.append(
            {
                "method": name,
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "mean_balanced_accuracy": result["mean_balanced_accuracy"],
                "std_balanced_accuracy": result["std_balanced_accuracy"],
                "mean_macro_f1": result["mean_macro_f1"],
                "std_macro_f1": result["std_macro_f1"],
                "jaccard_tuned": result["jaccard_stability"],
                "kuncheva_fixed_k": kuncheva,
            }
        )

    print("\n========== MONTH 2 REPEATED NESTED-CV SUMMARY ==========")
    print(pd.DataFrame(summary).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / "month2_selector_benchmark.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)

    metadata_path = RESULTS_DIR / "month2_selector_benchmark.json"
    metadata_record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "id": "golub-hu6800",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "evaluation": {
            "outer_splits": 5,
            "outer_repeats": 3,
            "inner_splits": 3,
            "scoring": "balanced_accuracy",
            "random_state": 42,
        },
        "kegg": {
            "snapshot_sha256": kegg_hashes,
            "n_probes": int(H.shape[0]),
            "n_pathways": int(H.shape[1]),
            "n_memberships": int(H.nnz),
        },
        "git": _git_state(),
        "packages": _package_versions(),
        "results_file": results_csv.name,
    }
    metadata_path.write_text(json.dumps(metadata_record, indent=2))
    print(f"Saved benchmark results: {results_csv}")
    print(f"Saved run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
