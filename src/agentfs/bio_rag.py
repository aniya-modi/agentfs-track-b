"""NCBI Entrez client for gene-symbol lookup and summary retrieval."""

from __future__ import annotations

import os
import time
from typing import Any

from Bio import Entrez


def ncbi_gene_summary(
    gene_symbol: str,
    *,
    organism: str = "Homo sapiens",
    pause_seconds: float = 0.34,
) -> dict[str, Any]:
    """Look up a gene symbol in NCBI Gene and return its summary and identifiers."""
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise RuntimeError("Set NCBI_EMAIL before using NCBI Entrez.")

    Entrez.email = email
    Entrez.api_key = os.environ.get("NCBI_API_KEY")
    query = f'{gene_symbol}[sym] AND "{organism}"[orgn]'

    with Entrez.esearch(db="gene", term=query, retmax=5) as handle:
        search_result = Entrez.read(handle)

    gene_ids = search_result.get("IdList", [])
    if not gene_ids:
        return {
            "query": gene_symbol,
            "organism": organism,
            "status": "not_found",
            "gene_id": None,
            "official_symbol": None,
            "summary": None,
        }

    if pause_seconds > 0:
        time.sleep(pause_seconds)
    with Entrez.esummary(db="gene", id=",".join(gene_ids)) as handle:
        summaries = Entrez.read(handle)

    first = summaries["DocumentSummarySet"]["DocumentSummary"][0]
    return {
        "query": gene_symbol,
        "organism": organism,
        "status": "found",
        "gene_id": str(first.attributes.get("uid", gene_ids[0])),
        "official_symbol": first.get("NomenclatureSymbol"),
        "summary": first.get("Summary"),
    }
