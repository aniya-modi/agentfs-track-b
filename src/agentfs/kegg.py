"""KEGG pathway snapshots and microarray probe incidence construction."""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .bio_rag import load_probe_annotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_PATH = PROJECT_ROOT / "data/annotation/gpl80_probe_to_gene_symbol.tsv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data/reference/kegg_snapshot"


def _read_or_download(cache_dir: Path, filename: str, url: str) -> tuple[str, str]:
    """Read a cached KEGG response or save the first successful download."""
    path = cache_dir / filename
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "agentfs-track-b/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        path.write_bytes(content)
    content = path.read_bytes()
    return content.decode("utf-8"), hashlib.sha256(content).hexdigest()


def build_kegg_incidence(
    probe_ids: list[str],
    annotation_path: str | Path = DEFAULT_ANNOTATION_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    probe_column: str = "Probe Set ID",
    gene_symbol_column: str = "Gene Symbol",
    kegg_organism: str = "hsa",
) -> tuple[sparse.csr_matrix, dict[str, str]]:
    """Build binary probe-by-organism-pathway incidence and return source hashes.

    KEGG REST text responses are preserved in ``cache_dir`` after first use so
    later runs use the same mapping. The caller should save the returned hashes
    with benchmark results. Delete the two cache files to intentionally refresh.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", kegg_organism):
        raise ValueError("kegg_organism must be a valid KEGG organism code, such as 'hsa'.")
    annotation: dict[str, Any] = load_probe_annotation(
        annotation_path,
        probe_column=probe_column,
        gene_symbol_column=gene_symbol_column,
    )
    cache_path = Path(cache_dir)
    links_filename = f"link_pathway_{kegg_organism}.tsv"
    genes_filename = f"list_{kegg_organism}.tsv"
    links_text, links_hash = _read_or_download(
        cache_path,
        links_filename,
        f"https://rest.kegg.jp/link/pathway/{kegg_organism}",
    )
    genes_text, genes_hash = _read_or_download(
        cache_path, genes_filename, f"https://rest.kegg.jp/list/{kegg_organism}"
    )

    kegg_id_to_symbols: dict[str, set[str]] = {}
    for line in genes_text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        gene_id = fields[0]
        # KEGG list rows may contain gene_id, type, position, and names.
        # Keep the two-column form for older snapshots and other organisms.
        names = fields[3] if len(fields) >= 4 else fields[1]
        symbols = {
            item.strip().upper()
            for item in names.split(";", maxsplit=1)[0].split(",")
            if item.strip()
        }
        if symbols:
            kegg_id_to_symbols[gene_id.strip()] = symbols

    symbol_to_pathways: dict[str, set[str]] = {}
    pathway_ids: set[str] = set()
    for line in links_text.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            continue
        gene_id, pathway_id = (field.strip() for field in fields)
        pathway_ids.add(pathway_id)
        for symbol in kegg_id_to_symbols.get(gene_id, set()):
            symbol_to_pathways.setdefault(symbol, set()).add(pathway_id)

    ordered_pathways = sorted(pathway_ids)
    pathway_index = {pathway_id: index for index, pathway_id in enumerate(ordered_pathways)}
    rows: list[int] = []
    columns: list[int] = []
    for row_index, probe_id in enumerate(probe_ids):
        memberships: set[str] = set()
        for symbol in annotation.get(probe_id, ()):
            memberships.update(symbol_to_pathways.get(str(symbol).upper(), set()))
        for pathway_id in memberships:
            rows.append(row_index)
            columns.append(pathway_index[pathway_id])

    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(len(probe_ids), len(ordered_pathways)),
    )
    hashes = {
        links_filename: links_hash,
        genes_filename: genes_hash,
    }
    return incidence, hashes
