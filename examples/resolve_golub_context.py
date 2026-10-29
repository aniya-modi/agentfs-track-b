"""Resolve Golub Hu6800 probe IDs using the matching GPL80 annotation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from agentfs.bio_rag import BioRAGClient, BioRAGError, load_probe_annotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/golub_processed.csv"
ANNOTATION_SOURCE = "NCBI GEO GPL80.annot.gz (Hu6800; annotation build 35, 2016-08-09)"


def read_feature_ids(path: Path, label_column: str = "target") -> list[str]:
    """Read feature names from the CSV header without loading expression values."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        headers = next(csv.reader(stream))
    return [header for header in headers if header != label_column]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="append",
        help="Hu6800 probe set ID to resolve; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of leading processed feature IDs to resolve when --probe is omitted.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    probe_ids = args.probe or read_feature_ids(PROCESSED_DATA_PATH)[: args.limit]
    annotation = load_probe_annotation(
        ANNOTATION_PATH,
        probe_column="Probe Set ID",
        gene_symbol_column="Gene Symbol",
    )
    client = BioRAGClient(
        annotation,
        organism="Homo sapiens",
        kegg_organism="hsa",
        annotation_source=ANNOTATION_SOURCE,
    )

    for probe_id in probe_ids:
        try:
            context = client.resolve_probe(probe_id, platform="GPL80")
            print(context.model_dump_json(indent=2))
        except BioRAGError as exc:
            print(
                json.dumps(
                    {
                        "probe_id": probe_id,
                        "platform": "GPL80",
                        "status": "lookup_error",
                        "detail": str(exc),
                        "sources": [ANNOTATION_SOURCE],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    if not os.environ.get("NCBI_EMAIL"):
        raise SystemExit("Set NCBI_EMAIL before running this live lookup example.")
    main()
