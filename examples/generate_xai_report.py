"""Generate a natural-language XAI report for retained Golub biomarkers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agentfs.baseline_selectors import MRMRSelector
from agentfs.gsea import signed_cohens_d
from agentfs.xai_report import (
    BiomarkerEvidence,
    RetentionEvidence,
    generate_xai_report,
    load_contexts,
    load_enrichment_terms,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLUB_PATH = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"
CONTEXTS_PATH = PROJECT_ROOT / "data/results/month5_xai_biomarker_contexts.json"
GSEA_REPORT_PATH = PROJECT_ROOT / "data/results/month5_gsea_validation.json"
PRERANK_CSV_PATH = PROJECT_ROOT / "data/results/month5_gsea_prerank.csv"
ORA_CSV_PATH = PROJECT_ROOT / "data/results/month5_gsea_ora.csv"
OUTPUT_MARKDOWN = PROJECT_ROOT / "data/results/month5_xai_report.md"
OUTPUT_JSON = PROJECT_ROOT / "data/results/month5_xai_report.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=CONTEXTS_PATH)
    parser.add_argument("--golub", type=Path, default=GOLUB_PATH)
    parser.add_argument("--gsea-report", type=Path, default=GSEA_REPORT_PATH)
    parser.add_argument("--prerank-csv", type=Path, default=PRERANK_CSV_PATH)
    parser.add_argument("--ora-csv", type=Path, default=ORA_CSV_PATH)
    parser.add_argument("--enrichment-fdr", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--positive-class", default="AML")
    parser.add_argument("--negative-class", default="ALL")
    parser.add_argument("--dataset-id", default="golub-hu6800")
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive.")

    contexts = load_contexts(args.contexts)
    frame = pd.read_csv(args.golub, compression="gzip")
    labels = (frame["target"].astype(str) == args.positive_class).astype(int).to_numpy()
    expression = frame.drop(columns=["target"])
    probe_ids = list(expression.columns)
    X = expression.to_numpy(dtype=float)

    selector = MRMRSelector(
        top_k=args.top_k, candidate_pool=max(args.candidate_pool, args.top_k)
    ).fit(X, labels)
    ranked_probes = [probe_ids[index] for index in selector.ranking_]
    missing = [context.probe_id for context in contexts if context.probe_id not in set(ranked_probes)]
    if missing:
        probe_flags = " ".join(f"--probe {probe}" for probe in ranked_probes)
        raise SystemExit(
            "Saved contexts do not match the retained probes: "
            f"{', '.join(missing)}. Re-run examples/resolve_golub_context.py with: "
            f"{probe_flags}"
        )
    retention_by_probe = {
        probe: RetentionEvidence(method="mrmr", rank=rank, requested_k=args.top_k)
        for rank, probe in enumerate(ranked_probes, start=1)
    }

    effects = signed_cohens_d(X, labels)
    effect_by_probe = {
        probe: float(effects[index]) for index, probe in enumerate(probe_ids)
    }

    biomarkers = [
        BiomarkerEvidence(
            context=context,
            retention=retention_by_probe.get(context.probe_id),
            signed_effect_size=effect_by_probe.get(context.probe_id),
            effect_metric="signed Cohen's d",
        )
        for context in contexts
    ]
    biomarkers.sort(
        key=lambda item: (
            item.retention.rank if item.retention and item.retention.rank else 10**6
        )
    )

    terms = []
    for csv_path in (args.prerank_csv, args.ora_csv):
        if csv_path.is_file():
            terms.extend(load_enrichment_terms(csv_path, fdr=args.enrichment_fdr))
    if not terms and args.gsea_report.is_file():
        terms = load_enrichment_terms(args.gsea_report, fdr=args.enrichment_fdr)
    unique_terms = {}
    for term in terms:
        key = term.term.casefold()
        current = unique_terms.get(key)
        if current is None or (
            term.fdr is not None and (current.fdr is None or term.fdr < current.fdr)
        ):
            unique_terms[key] = term
    terms = sorted(
        unique_terms.values(),
        key=lambda item: (item.fdr is None, item.fdr or 0.0, -abs(item.nes or 0.0)),
    )
    print(
        f"Loaded {len(terms)} enrichment terms at FDR <= {args.enrichment_fdr}."
    )

    report = generate_xai_report(
        args.dataset_id,
        biomarkers,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        enrichment_terms=terms,
    )
    write_report(report, markdown_path=args.output_markdown, json_path=args.output_json)

    print(report.summary)
    for biomarker in report.biomarkers:
        print(
            f"- {biomarker.gene_symbol or biomarker.probe_id}: "
            f"{biomarker.direction}; evidence={biomarker.evidence_level}"
        )
    print(f"Saved Markdown report: {args.output_markdown}")
    print(f"Saved JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
