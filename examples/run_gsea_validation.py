"""Run GSEA and ORA validation for a selected Golub gene subset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from agentfs.baseline_selectors import LassoSelector, MRMRSelector
from agentfs.bio_rag import load_probe_annotation
from agentfs.gsea import (
    DEFAULT_LIBRARY,
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    DEFAULT_PERMUTATIONS,
    load_gene_sets,
    ranked_gene_series,
    run_ora,
    run_prerank,
    signed_cohens_d,
    summarise_ora,
    summarise_prerank,
    symbols_for_probes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLUB_PATH = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"
ANNOTATION_PATH = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data/reference/gene_sets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/results"


def _select_probes(method: str, X: np.ndarray, y: np.ndarray, top_k: int) -> list[int]:
    if method == "fscore":
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
        return np.argsort(-scores, kind="mergesort")[:top_k].tolist()
    if method == "mrmr":
        selector = MRMRSelector(top_k=top_k, candidate_pool=max(500, top_k))
    elif method == "lasso":
        selector = LassoSelector(top_k=top_k)
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unsupported selector: {method}")
    return selector.fit(X, y).get_support(indices=True).tolist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a selected Golub gene subset with gseapy."
    )
    parser.add_argument("--selector", choices=("fscore", "mrmr", "lasso"), default="mrmr")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive.")

    frame = pd.read_csv(GOLUB_PATH, compression="gzip")
    labels = (frame["target"].astype(str) == "AML").astype(int).to_numpy()
    expression = frame.drop(columns=["target"])
    probe_ids = list(expression.columns)
    X = expression.to_numpy(dtype=float)

    probe_to_gene = load_probe_annotation(
        ANNOTATION_PATH, probe_column="Probe Set ID", gene_symbol_column="Gene Symbol"
    )

    selected_indices = _select_probes(args.selector, X, labels, args.top_k)
    selected_probes = [probe_ids[index] for index in selected_indices]
    selected_genes = symbols_for_probes(selected_probes, probe_to_gene)

    effects = signed_cohens_d(X, labels)
    ranked = ranked_gene_series(effects, probe_ids, probe_to_gene)
    background = symbols_for_probes(probe_ids, probe_to_gene)

    print(
        f"Loaded Golub: {X.shape[0]} samples x {X.shape[1]} probes; "
        f"selector={args.selector}, top_k={args.top_k}"
    )
    print(
        f"Selected {len(selected_probes)} probes -> {len(selected_genes)} gene symbols; "
        f"ranked {len(ranked)} unique symbols by signed Cohen's d"
    )

    gene_sets = load_gene_sets(args.library, args.cache_dir)
    print(f"Library {args.library}: {len(gene_sets)} gene sets (cache: {args.cache_dir})")

    prerank_table = run_prerank(
        ranked,
        gene_sets,
        min_size=args.min_size,
        max_size=args.max_size,
        permutations=args.permutations,
        seed=args.seed,
        threads=args.threads,
    )
    ora_table = run_ora(selected_genes, gene_sets, background)
    prerank_summary = summarise_prerank(prerank_table, fdr=args.fdr)
    ora_summary = summarise_ora(ora_table, fdr=args.fdr)

    print(
        f"Prerank: {prerank_summary['n_significant']}/{prerank_summary['n_tested']} "
        f"gene sets at FDR <= {args.fdr}"
    )
    for record in prerank_summary["top_terms"][:5]:
        print(
            f"  {record['term']}: NES={record['nes']:.2f}, FDR={record['fdr']:.3g}, "
            f"{record['direction']}"
        )
    print(
        f"ORA: {ora_summary['n_significant']}/{ora_summary['n_tested']} "
        f"gene sets at adjusted p <= {args.fdr}"
    )
    for record in ora_summary["top_terms"][:5]:
        print(
            f"  {record['term']}: overlap={record['overlap']}, "
            f"adjusted_p={record['adjusted_p']:.3g}"
        )

    sizes = [len(genes) for genes in gene_sets.values()]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "descriptive pathway validation of a selected gene subset",
        "dataset": {
            "id": "golub-hu6800",
            "n_samples": int(X.shape[0]),
            "n_probes": int(X.shape[1]),
            "positive_class": "AML",
        },
        "selection": {
            "method": args.selector,
            "top_k": args.top_k,
            "uses_full_dataset_labels": True,
            "n_selected_probes": len(selected_probes),
            "n_selected_symbols": len(selected_genes),
            "selected_symbols": selected_genes,
        },
        "ranking": {
            "method": "signed_cohens_d",
            "positive_direction": "higher_in_AML",
            "n_ranked_symbols": len(ranked),
        },
        "gene_sets": {
            "library": args.library,
            "n_sets": len(gene_sets),
            "min_set_size": int(min(sizes)) if sizes else 0,
            "max_set_size": int(max(sizes)) if sizes else 0,
            "min_size_used": args.min_size,
            "max_size_used": args.max_size,
        },
        "prerank": {
            **prerank_summary,
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "ora": ora_summary,
        "packages": {"gseapy": version("gseapy")},
        "caveats": [
            "Selection uses all samples, so this is descriptive validation, not held-out evaluation.",
            "Prerank p-values come from gene-set permutations on one cohort; treat them as exploratory.",
            "Multi-symbol probes contribute every annotated symbol; unmapped probes are dropped.",
            "ORA and preranked GSEA use different null hypotheses and answer different questions.",
            "Enrichment does not establish clinical or diagnostic value.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "month5_gsea_validation.json"
    prerank_path = output_dir / "month5_gsea_prerank.csv"
    ora_path = output_dir / "month5_gsea_ora.csv"
    report_path.write_text(json.dumps(report, indent=2))
    prerank_table.to_csv(prerank_path, index=False)
    ora_table.to_csv(ora_path, index=False)
    print(f"Saved report: {report_path}")
    print(f"Saved prerank table: {prerank_path}")
    print(f"Saved ORA table: {ora_path}")


if __name__ == "__main__":
    main()
