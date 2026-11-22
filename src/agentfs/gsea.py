"""Gene set enrichment validation for selected microarray gene subsets.

The module wraps :mod:`gseapy` for two complementary analyses:

* preranked GSEA against a gene-set library; and
* over-representation analysis (ORA) of a selected gene subset.

Gene sets can be supplied as a dictionary (offline, used by tests) or loaded
from an Enrichr library once and cached as a GMT file for reproducible reruns.
These analyses describe pathway relevance; they are not clinical validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gseapy as gp
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "gseapy is required for GSEA validation. Install it with "
        "`python -m pip install -e '.[analysis]'`."
    ) from exc

DEFAULT_LIBRARY = "KEGG_2021_Human"
DEFAULT_MIN_SIZE = 15
DEFAULT_MAX_SIZE = 500
DEFAULT_PERMUTATIONS = 1000


def load_gene_sets(
    library: str,
    cache_dir: str | Path,
    *,
    organism: str = "human",
) -> dict[str, list[str]]:
    """Load an Enrichr library, caching the full GMT on first use."""
    cache_path = Path(cache_dir) / f"{library}.gmt"
    if cache_path.is_file():
        return gp.read_gmt(str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gene_sets = gp.get_library(
        library, organism=organism, min_size=0, max_size=2000, save=str(cache_path)
    )
    return {name: list(genes) for name, genes in gene_sets.items()}


def normalise_symbols(symbols: Sequence[str]) -> list[str]:
    """Upper-case, strip, and de-duplicate gene symbols while keeping order."""
    cleaned: list[str] = []
    for symbol in symbols:
        value = str(symbol).strip().upper()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def symbols_for_probes(
    probes: Sequence[str], probe_to_gene: Mapping[str, Sequence[str]]
) -> list[str]:
    """Map probes to unique gene symbols, dropping unmapped and blank rows."""
    symbols: list[str] = []
    for probe in probes:
        symbols.extend(probe_to_gene.get(probe, ()))
    return normalise_symbols(symbols)


def signed_cohens_d(X: Any, y: Any) -> np.ndarray:
    """Return Cohen's d per feature; positive means higher in the last class.

    Classes are ordered by ``np.unique`` so binary labels ``0/1`` give
    positive values for class ``1`` (AML in this project). Features with zero
    pooled variance receive a score of zero.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional samples-by-features matrix.")
    if len(y_arr) != X_arr.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError("signed_cohens_d requires exactly two classes.")
    positive = X_arr[y_arr == classes[1]]
    negative = X_arr[y_arr == classes[0]]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("Each class needs at least two samples.")

    mean_difference = positive.mean(axis=0) - negative.mean(axis=0)
    pooled_variance = (
        (len(positive) - 1) * positive.var(axis=0, ddof=1)
        + (len(negative) - 1) * negative.var(axis=0, ddof=1)
    ) / (len(positive) + len(negative) - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(pooled_variance > 0, mean_difference / np.sqrt(pooled_variance), 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def ranked_gene_frame(
    scores: Sequence[float],
    probe_ids: Sequence[str],
    probe_to_gene: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Collapse probe scores to genes, keeping the largest absolute effect."""
    if len(scores) != len(probe_ids):
        raise ValueError("scores and probe_ids must have the same length.")

    best: dict[str, float] = {}
    for probe, score in zip(probe_ids, scores, strict=True):
        value = float(score)
        for symbol in normalise_symbols(probe_to_gene.get(probe, ())):
            current = best.get(symbol)
            if current is None or abs(value) > abs(current):
                best[symbol] = value

    frame = pd.DataFrame(
        {"gene": list(best.keys()), "score": list(best.values())}
    ).sort_values("score", ascending=False, kind="mergesort")
    return frame.reset_index(drop=True)


def ranked_gene_series(
    scores: Sequence[float],
    probe_ids: Sequence[str],
    probe_to_gene: Mapping[str, Sequence[str]],
) -> pd.Series:
    """Return a unique, score-sorted Series indexed by gene symbol.

    Equal effects are ordered by gene symbol and then separated by a tiny
    deterministic offset, because GSEA requires a strictly ordered ranking.
    """
    frame = ranked_gene_frame(scores, probe_ids, probe_to_gene)
    frame = frame.sort_values(["score", "gene"], ascending=[False, True], kind="mergesort")
    values = frame["score"].to_numpy(dtype=float)
    if len(np.unique(values)) < len(values):
        values = values + np.arange(len(values), dtype=float) * 1e-12
    return pd.Series(
        values,
        index=frame["gene"].to_numpy(),
        name="score",
    )


def run_prerank(
    ranked: pd.Series | pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
    threads: int = 1,
) -> pd.DataFrame:
    """Run gseapy preranked GSEA and return one row per tested gene set."""
    if isinstance(ranked, pd.DataFrame):
        if ranked.shape[1] != 2:
            raise ValueError("ranked DataFrame must have exactly two columns.")
        ranking = pd.Series(
            ranked.iloc[:, 1].to_numpy(dtype=float),
            index=ranked.iloc[:, 0].astype(str).to_numpy(),
            name="score",
        )
    else:
        ranking = pd.Series(ranked, dtype=float, copy=True)
    ranking = ranking[~ranking.index.duplicated(keep="first")].dropna()
    if ranking.empty:
        raise ValueError("ranked gene list is empty after cleaning.")

    result = gp.prerank(
        rnk=ranking,
        gene_sets=dict(gene_sets),
        outdir=None,
        no_plot=True,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutations,
        seed=seed,
        threads=threads,
        verbose=False,
    )
    table = result.res2d.copy()
    for column in ("ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def run_ora(
    genes: Sequence[str],
    gene_sets: Mapping[str, Sequence[str]],
    background: Sequence[str],
    *,
    organism: str = "human",
) -> pd.DataFrame:
    """Run offline over-representation analysis of a selected gene subset."""
    selected = normalise_symbols(genes)
    universe = normalise_symbols(background)
    if not selected:
        raise ValueError("selected gene list is empty.")
    if not universe:
        raise ValueError("background gene list is empty.")

    result = gp.enrichr(
        gene_list=selected,
        gene_sets=dict(gene_sets),
        background=universe,
        outdir=None,
        no_plot=True,
        organism=organism,
    )
    table = result.results.copy()
    for column in ("P-value", "Adjusted P-value", "Odds Ratio", "Combined Score"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def summarise_prerank(
    table: pd.DataFrame, *, fdr: float = 0.05, top_n: int = 10
) -> dict[str, Any]:
    """Summarise prerank output at a chosen FDR threshold."""
    if "FDR q-val" not in table:
        raise ValueError("prerank table is missing the 'FDR q-val' column.")
    significant = table[table["FDR q-val"] <= fdr].copy()
    significant["abs_nes"] = significant["NES"].abs()
    significant = significant.sort_values(
        ["FDR q-val", "abs_nes"], ascending=[True, False], kind="mergesort"
    )
    records = [
        {
            "term": str(row["Term"]),
            "nes": float(row["NES"]) if pd.notna(row["NES"]) else None,
            "nominal_p": float(row["NOM p-val"]) if pd.notna(row["NOM p-val"]) else None,
            "fdr": float(row["FDR q-val"]),
            "direction": "up_in_positive_class" if row["NES"] >= 0 else "down_in_positive_class",
            "lead_genes": str(row.get("Lead_genes", "")),
        }
        for _, row in significant.head(top_n).iterrows()
    ]
    return {
        "n_tested": len(table),
        "n_significant": len(significant),
        "fdr_threshold": fdr,
        "top_terms": records,
    }


def summarise_ora(
    table: pd.DataFrame, *, fdr: float = 0.05, top_n: int = 10
) -> dict[str, Any]:
    """Summarise ORA output at a chosen adjusted-p threshold."""
    if "Adjusted P-value" not in table:
        raise ValueError("ORA table is missing the 'Adjusted P-value' column.")
    significant = table[table["Adjusted P-value"] <= fdr].sort_values(
        "Adjusted P-value", kind="mergesort"
    )
    records = [
        {
            "term": str(row["Term"]),
            "overlap": str(row.get("Overlap", "")),
            "p_value": float(row["P-value"]) if pd.notna(row["P-value"]) else None,
            "adjusted_p": float(row["Adjusted P-value"]),
            "odds_ratio": (
                float(row["Odds Ratio"]) if pd.notna(row.get("Odds Ratio")) else None
            ),
            "genes": str(row.get("Genes", "")),
        }
        for _, row in significant.head(top_n).iterrows()
    ]
    return {
        "n_tested": len(table),
        "n_significant": len(significant),
        "fdr_threshold": fdr,
        "top_terms": records,
    }
