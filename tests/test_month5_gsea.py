"""Month 5 tests for gseapy-based GSEA validation."""

import numpy as np
import pandas as pd
import pytest

from agentfs.gsea import (
    load_gene_sets,
    normalise_symbols,
    ranked_gene_frame,
    ranked_gene_series,
    run_ora,
    run_prerank,
    signed_cohens_d,
    summarise_ora,
    summarise_prerank,
    symbols_for_probes,
)


def test_signed_cohens_d_direction_and_zero_variance():
    X = np.array(
        [
            [1.0, 5.0, 5.0],
            [2.0, 4.0, 5.0],
            [3.0, 3.0, 5.0],
            [5.0, 1.0, 5.0],
            [6.0, 2.0, 5.0],
            [7.0, 3.0, 5.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    scores = signed_cohens_d(X, y)

    assert scores[0] > 0  # higher in class 1
    assert scores[1] < 0  # lower in class 1
    assert scores[2] == 0  # zero pooled variance
    with pytest.raises(ValueError, match="exactly two classes"):
        signed_cohens_d(X, np.array([0, 1, 2, 0, 1, 2]))


def test_ranked_gene_frame_collapses_probes():
    scores = [2.0, -3.0, 0.5, -9.0]
    probes = ["p1", "p2", "p3", "p4"]
    mapping = {"p1": ["A"], "p2": ["B"], "p3": ["C"], "p4": ["A", "ALIAS"]}

    frame = ranked_gene_frame(scores, probes, mapping)

    assert list(frame["gene"]) == ["C", "B", "A", "ALIAS"]
    assert dict(zip(frame["gene"], frame["score"], strict=True)) == {
        "C": 0.5,
        "B": -3.0,
        "A": -9.0,
        "ALIAS": -9.0,
    }
    assert normalise_symbols([" a ", "A", "", "b"]) == ["A", "B"]
    assert symbols_for_probes(["p1", "p2", "missing"], mapping) == ["A", "B"]


def test_ranked_gene_series_breaks_ties_deterministically():
    series = ranked_gene_series(
        [1.0, 1.0, 2.0],
        ["p1", "p2", "p3"],
        {"p1": ["B"], "p2": ["A"], "p3": ["C"]},
    )

    assert list(series.index) == ["C", "A", "B"]
    assert len(set(series.to_numpy())) == 3


def test_load_gene_sets_reads_cached_gmt(tmp_path):
    gmt = tmp_path / "KEGG_2021_Human.gmt"
    gmt.write_text(
        "SET_A\tdescription\tA\tB\tC\nSET_B\tdescription\tD\tE\n",
        encoding="utf-8",
    )

    gene_sets = load_gene_sets("KEGG_2021_Human", tmp_path)

    assert gene_sets == {"SET_A": ["A", "B", "C"], "SET_B": ["D", "E"]}


def test_offline_prerank_and_ora_return_tables():
    genes = [f"G{index}" for index in range(80)]
    ranked = pd.Series(np.linspace(3.0, -3.0, len(genes)), index=genes)
    gene_sets = {
        "TOP_SET": genes[:10],
        "MIDDLE_SET": genes[35:45],
        "BOTTOM_SET": genes[-10:],
    }

    prerank = run_prerank(
        ranked,
        gene_sets,
        min_size=3,
        max_size=20,
        permutations=20,
        seed=7,
        threads=1,
    )
    assert set(prerank["Term"]) == set(gene_sets)
    assert {"ES", "NES", "NOM p-val", "FDR q-val"} <= set(prerank.columns)

    ora = run_ora(genes[:10], gene_sets, genes)
    assert "Adjusted P-value" in ora.columns
    assert ora.sort_values("Adjusted P-value").iloc[0]["Term"] == "TOP_SET"
    assert ora["Adjusted P-value"].min() < 0.05


def test_summaries_filter_thresholds():
    prerank = pd.DataFrame(
        {
            "Term": ["A", "B", "C"],
            "NES": [2.0, -1.5, 0.5],
            "NOM p-val": [0.01, 0.02, 0.5],
            "FDR q-val": [0.01, 0.20, 0.90],
            "Lead_genes": ["G1;G2", "G3", "G4"],
        }
    )
    summary = summarise_prerank(prerank, fdr=0.05)
    assert summary["n_tested"] == 3
    assert summary["n_significant"] == 1
    assert summary["top_terms"][0]["term"] == "A"

    ora = pd.DataFrame(
        {
            "Term": ["A", "B"],
            "Overlap": ["5/10", "1/10"],
            "P-value": [0.001, 0.5],
            "Adjusted P-value": [0.01, 0.8],
            "Odds Ratio": [4.0, 1.1],
            "Genes": ["G1;G2", "G3"],
        }
    )
    ora_summary = summarise_ora(ora, fdr=0.05)
    assert ora_summary["n_significant"] == 1
    assert ora_summary["top_terms"][0]["term"] == "A"
