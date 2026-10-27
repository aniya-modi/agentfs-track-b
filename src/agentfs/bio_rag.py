"""NCBI and KEGG context retrieval for platform-annotated microarray probes.

Probe identifiers are platform-specific. This client deliberately requires a
platform annotation mapping before resolving a probe; it never guesses a gene
symbol by searching an opaque probe ID.
"""

from __future__ import annotations

import csv
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from Bio import Entrez

from .schemas import PathwayRecord, ProbeGeneContext

NCBI_EUTILS_SOURCE = "NCBI Gene E-utilities"
KEGG_SOURCE = "KEGG REST API"


class BioRAGError(RuntimeError):
    """Raised when a biological context service cannot complete a lookup."""


def load_probe_annotation(
    path: str | Path,
    *,
    probe_column: str,
    gene_symbol_column: str,
) -> dict[str, list[str]]:
    """Load a platform annotation CSV/TSV into probe-to-symbol candidates.

    Duplicate rows and symbols separated by ``///`` or ``;`` are retained as
    candidates. Ambiguities are reported by :meth:`BioRAGClient.resolve_probe`
    instead of being silently collapsed.
    """
    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    delimiter = "\t" if annotation_path.suffix.lower() in {".tsv", ".txt"} else ","

    mapping: dict[str, list[str]] = {}
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Annotation file must have a header row.")
        missing_columns = {probe_column, gene_symbol_column} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Missing annotation columns: {sorted(missing_columns)}")

        for row in reader:
            probe_id = (row.get(probe_column) or "").strip()
            raw_symbols = (row.get(gene_symbol_column) or "").strip()
            if not probe_id or not raw_symbols:
                continue
            symbols = [item.strip() for item in re.split(r"\s*///\s*|\s*;\s*", raw_symbols)]
            candidates = mapping.setdefault(probe_id, [])
            for symbol in symbols:
                if symbol and symbol not in candidates:
                    candidates.append(symbol)
    return mapping


class BioRAGClient:
    """Resolve annotated probes to NCBI Gene summaries and KEGG pathways.

    Args:
        probe_to_gene: Mapping from the selected microarray platform's probe
            IDs to one or more gene symbols, usually loaded from a platform
            annotation file with :func:`load_probe_annotation`.
        organism: NCBI organism name.
        kegg_organism: KEGG organism code (``hsa`` for human).
        email: NCBI contact email. Defaults to ``NCBI_EMAIL``.
        api_key: Optional NCBI API key. Defaults to ``NCBI_API_KEY``.
    """

    def __init__(
        self,
        probe_to_gene: Mapping[str, str | Sequence[str]] | None = None,
        *,
        organism: str = "Homo sapiens",
        kegg_organism: str = "hsa",
        email: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        kegg_min_interval_seconds: float = 0.34,
    ) -> None:
        contact_email = email or os.environ.get("NCBI_EMAIL")
        if not contact_email:
            raise ValueError("Set NCBI_EMAIL or pass email when creating BioRAGClient.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if retry_backoff_seconds < 0 or kegg_min_interval_seconds < 0:
            raise ValueError("Retry and request-interval values cannot be negative.")
        if not re.fullmatch(r"[A-Za-z0-9_]+", kegg_organism):
            raise ValueError("kegg_organism must be a KEGG organism code, such as 'hsa'.")

        self.organism = organism
        self.kegg_organism = kegg_organism
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.kegg_min_interval_seconds = kegg_min_interval_seconds
        self.probe_to_gene = {
            str(probe).strip(): self._normalize_symbols(symbols)
            for probe, symbols in (probe_to_gene or {}).items()
        }
        self._gene_cache: dict[str, dict[str, Any]] = {}
        self._kegg_gene_cache: dict[str, tuple[str | None, str | None]] = {}
        self._pathway_cache: dict[str, list[dict[str, str]]] = {}
        self._pathway_names: dict[str, dict[str, str]] = {}
        self._http = requests.Session()
        self._last_kegg_request = 0.0
        self._kegg_lock = threading.Lock()

        Entrez.email = contact_email
        Entrez.tool = "agentfs_track_b"
        Entrez.api_key = api_key or os.environ.get("NCBI_API_KEY")

    @staticmethod
    def _normalize_symbols(value: str | Sequence[str]) -> list[str]:
        if isinstance(value, str):
            values = re.split(r"\s*///\s*|\s*;\s*", value)
        else:
            values = list(value)
        result: list[str] = []
        for item in values:
            symbol = str(item).strip()
            if symbol and symbol not in result:
                result.append(symbol)
        return result

    @staticmethod
    def _response_model(
        *,
        probe_id: str,
        platform: str | None,
        organism: str,
        status: str,
        retrieved_at: datetime,
        **values: Any,
    ) -> ProbeGeneContext:
        return ProbeGeneContext(
            probe_id=probe_id,
            platform=platform,
            organism=organism,
            status=status,  # validated by Pydantic
            retrieved_at=retrieved_at,
            **values,
        )

    def _lookup_ncbi_gene(self, gene_symbol: str) -> dict[str, Any]:
        cache_key = gene_symbol.casefold()
        if cache_key in self._gene_cache:
            return self._gene_cache[cache_key]

        term = f'"{gene_symbol}"[sym] AND "{self.organism}"[orgn]'
        try:
            with Entrez.esearch(db="gene", term=term, retmax=20, sort="relevance") as handle:
                search_result = Entrez.read(handle)
            gene_ids = [str(item) for item in search_result.get("IdList", [])]
            if not gene_ids:
                result = {"status": "gene_not_found", "candidates": []}
                self._gene_cache[cache_key] = result
                return result

            with Entrez.esummary(db="gene", id=",".join(gene_ids)) as handle:
                summary_response = Entrez.read(handle)
            documents = summary_response["DocumentSummarySet"]["DocumentSummary"]
        except Exception as exc:
            raise BioRAGError(f"NCBI Gene lookup failed for {gene_symbol!r}: {exc}") from exc

        candidates: list[dict[str, str | None]] = []
        exact_matches: list[dict[str, str | None]] = []
        for document in documents:
            symbol = document.get("NomenclatureSymbol") or document.get("Name")
            candidate = {
                "gene_id": str(document.attributes.get("uid", "")),
                "gene_symbol": str(symbol) if symbol else None,
                "summary": str(document.get("Summary") or "") or None,
            }
            candidates.append(candidate)
            if symbol and str(symbol).casefold() == gene_symbol.casefold():
                exact_matches.append(candidate)

        if len(exact_matches) == 1:
            result = {"status": "resolved", **exact_matches[0]}
        elif len(exact_matches) > 1:
            result = {"status": "ambiguous_gene_mapping", "candidates": exact_matches}
        else:
            result = {"status": "ambiguous_gene_mapping", "candidates": candidates}
        self._gene_cache[cache_key] = result
        return result

    def _get_kegg(self, operation: str, argument: str) -> str:
        url = f"https://rest.kegg.jp/{operation}/{quote(argument, safe='/:')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._kegg_lock:
                    elapsed = time.monotonic() - self._last_kegg_request
                    wait = self.kegg_min_interval_seconds - elapsed
                    if wait > 0:
                        time.sleep(wait)
                    response = self._http.get(url, timeout=self.timeout_seconds)
                    self._last_kegg_request = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise BioRAGError(f"KEGG request failed at {url}: {last_error}") from last_error

    def _find_kegg_gene(self, gene_symbol: str) -> tuple[str | None, str | None]:
        cache_key = gene_symbol.casefold()
        if cache_key in self._kegg_gene_cache:
            return self._kegg_gene_cache[cache_key]

        response = self._get_kegg(
            "find",
            f"{self.kegg_organism}/{gene_symbol}",
        )
        exact: list[tuple[str, str]] = []
        for line in response.splitlines():
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2:
                continue
            kegg_gene_id, names = fields
            primary_name = names.split(",", maxsplit=1)[0].strip()
            if primary_name.casefold() == gene_symbol.casefold():
                exact.append((kegg_gene_id, names))

        result = exact[0] if len(exact) == 1 else (None, None)
        self._kegg_gene_cache[cache_key] = result
        return result

    def _load_kegg_pathway_names(self) -> dict[str, str]:
        if self.kegg_organism not in self._pathway_names:
            response = self._get_kegg("list/pathway", self.kegg_organism)
            names: dict[str, str] = {}
            for line in response.splitlines():
                fields = line.split("\t", maxsplit=1)
                if len(fields) == 2:
                    pathway_id = fields[0].removeprefix("path:")
                    names[pathway_id] = fields[1].strip()
            self._pathway_names[self.kegg_organism] = names
        return self._pathway_names[self.kegg_organism]

    def _get_gene_pathways(self, kegg_gene_id: str) -> list[PathwayRecord]:
        if kegg_gene_id in self._pathway_cache:
            return [PathwayRecord(**item) for item in self._pathway_cache[kegg_gene_id]]

        response = self._get_kegg("link/pathway", kegg_gene_id)
        pathway_ids: list[str] = []
        for line in response.splitlines():
            fields = line.split("\t", maxsplit=1)
            if len(fields) == 2:
                pathway_id = fields[1].removeprefix("path:")
                if pathway_id not in pathway_ids:
                    pathway_ids.append(pathway_id)

        names = self._load_kegg_pathway_names()
        pathways = [
            PathwayRecord(pathway_id=pathway_id, name=names.get(pathway_id, pathway_id))
            for pathway_id in pathway_ids
        ]
        encoded = [item.model_dump(mode="json") for item in pathways]
        self._pathway_cache[kegg_gene_id] = encoded
        return pathways

    def resolve_probe(
        self,
        probe_id: str,
        *,
        platform: str | None = None,
    ) -> ProbeGeneContext:
        """Resolve one probe using the supplied platform annotation mapping.

        An unknown or ambiguous annotation is returned as a typed status rather
        than guessed. Network/API failures raise :class:`BioRAGError` so callers
        can log them in ``AgentState.errors`` and retry deliberately.
        """
        normalized_probe = probe_id.strip()
        if not normalized_probe:
            raise ValueError("probe_id must not be empty.")
        retrieved_at = datetime.now(timezone.utc)
        symbols = self.probe_to_gene.get(normalized_probe)
        if symbols is None:
            lookup = {key.casefold(): value for key, value in self.probe_to_gene.items()}
            symbols = lookup.get(normalized_probe.casefold())

        if not symbols:
            return self._response_model(
                probe_id=normalized_probe,
                platform=platform,
                organism=self.organism,
                status="unmapped_probe",
                retrieved_at=retrieved_at,
                sources=["platform annotation required"],
                detail=(
                    "No platform annotation mapping was supplied for this probe; "
                    "the identifier was not guessed or sent as a gene symbol."
                ),
            )
        if len(symbols) != 1:
            return self._response_model(
                probe_id=normalized_probe,
                platform=platform,
                organism=self.organism,
                status="ambiguous_probe_mapping",
                retrieved_at=retrieved_at,
                gene_symbols=list(symbols),
                sources=["platform annotation"],
            )

        symbol = symbols[0]
        gene = self._lookup_ncbi_gene(symbol)
        if gene["status"] != "resolved":
            return self._response_model(
                probe_id=normalized_probe,
                platform=platform,
                organism=self.organism,
                status=gene["status"],
                retrieved_at=retrieved_at,
                gene_symbols=[symbol],
                candidate_gene_ids=[
                    str(candidate["gene_id"])
                    for candidate in gene.get("candidates", [])
                    if candidate.get("gene_id")
                ],
                candidate_gene_symbols=[
                    str(candidate["gene_symbol"])
                    for candidate in gene.get("candidates", [])
                    if candidate.get("gene_symbol")
                ],
                sources=[NCBI_EUTILS_SOURCE],
                detail=f"NCBI returned {len(gene.get('candidates', []))} candidate records.",
            )

        kegg_gene_id, _ = self._find_kegg_gene(symbol)
        pathways = self._get_gene_pathways(kegg_gene_id) if kegg_gene_id else []
        sources = [NCBI_EUTILS_SOURCE]
        if kegg_gene_id:
            sources.append(KEGG_SOURCE)
        return self._response_model(
            probe_id=normalized_probe,
            platform=platform,
            organism=self.organism,
            status="resolved",
            retrieved_at=retrieved_at,
            gene_symbols=[symbol],
            gene_symbol=symbol,
            ncbi_gene_id=gene.get("gene_id") or None,
            gene_summary=gene.get("summary"),
            kegg_gene_id=kegg_gene_id,
            pathways=pathways,
            sources=sources,
        )

    def resolve_probes(
        self,
        probe_ids: Sequence[str],
        *,
        platform: str | None = None,
    ) -> list[ProbeGeneContext]:
        """Resolve probes in input order, reusing in-memory API response caches."""
        return [self.resolve_probe(probe_id, platform=platform) for probe_id in probe_ids]
