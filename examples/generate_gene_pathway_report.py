"""Generate an exploratory gene-selection and biological-context report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agentfs.bio_rag import BioRAGClient, load_probe_annotation
from agentfs.hypergraph_experiments import BioStatSoftRedundancySelector
from agentfs.kegg import build_kegg_incidence
from agentfs.knockoff_selector import TwoStageKnockoffFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLUB_PATH = PROJECT_ROOT / "data/reference/golub_hu6800.csv.gz"
ANNOTATION_PATH = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"


def main() -> None:
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise RuntimeError("Set NCBI_EMAIL before running live NCBI/Bio-RAG lookups.")

    frame = pd.read_csv(GOLUB_PATH, compression="gzip")
    labels = (frame["target"].astype(str) == "AML").astype(int).to_numpy()
    expression = frame.drop(columns=["target"])
    probe_ids = list(expression.columns)
    X = expression.to_numpy(dtype=float)

    print("=== Exploratory supervised screen + knockoff selection ===")
    selector = TwoStageKnockoffFilter(fdr=0.10, stage1_k=250)
    X_candidates = selector.fit_transform(X, labels)
    candidate_indices = selector.get_support(indices=True)
    candidate_probes = [probe_ids[index] for index in candidate_indices]
    print(
        f"Selected {len(candidate_probes)} of {X.shape[1]} probes. "
        "The requested threshold is not a formal FDR guarantee."
    )

    if not candidate_probes:
        print("No probes passed the knockoff selection rule; stopping without forcing genes.")
        return

    H, hashes = build_kegg_incidence(candidate_probes, annotation_path=ANNOTATION_PATH)
    top_k = min(10, len(candidate_probes))
    pathway_selector = BioStatSoftRedundancySelector(
        H=H, top_k=top_k, alpha=0.5, gamma=0.25
    ).fit(X_candidates, labels)
    top_local_indices = pathway_selector.get_support(indices=True)
    top_probes = [candidate_probes[index] for index in top_local_indices]

    annotation = load_probe_annotation(
        ANNOTATION_PATH, probe_column="Probe Set ID", gene_symbol_column="Gene Symbol"
    )
    client = BioRAGClient(probe_to_gene=annotation, email=email)

    resolved_records = []
    for probe in top_probes:
        context = client.resolve_probe(probe, platform="GPL80")
        resolved_records.append(context.model_dump(mode="json"))
        print(
            f"{probe} -> status={context.status} | symbol={context.gene_symbol} | "
            f"pathways={len(context.pathways)}"
        )

    report = {
        "purpose": "exploratory full-dataset feature and pathway review",
        "uses_full_dataset_labels_for_selection": True,
        "formal_fdr_control": False,
        "requested_knockoff_threshold": selector.fdr_target_,
        "screening_method": selector.screening_method_,
        "n_features_screened": selector.n_features_screened_,
        "n_features_selected": selector.n_features_selected_,
        "kegg_snapshot_sha256": hashes,
        "records": resolved_records,
    }
    output_path = PROJECT_ROOT / "data/results/gene_pathway_context_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Exploratory report saved to {output_path}")


if __name__ == "__main__":
    main()
