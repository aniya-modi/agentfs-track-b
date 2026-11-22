"""Month 5 tests for the deterministic XAI report generator."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from agentfs.baseline_selectors import MRMRSelector
from agentfs.schemas import PathwayRecord, ProbeGeneContext
from agentfs.xai_report import (
    BiomarkerEvidence,
    EnrichmentTerm,
    RetentionEvidence,
    generate_xai_report,
    load_contexts,
    load_enrichment_terms,
    load_retention,
    write_report,
)


def _context(
    probe_id: str = "X17094_at",
    symbol: str | None = "FURIN",
    *,
    status: str = "resolved",
    pathways: list[PathwayRecord] | None = None,
    summary: str | None = "FURIN is a proprotein convertase. It processes precursor proteins.",
    symbols: list[str] | None = None,
) -> ProbeGeneContext:
    resolved_symbols = symbols if symbols is not None else ([symbol] if symbol else [])
    return ProbeGeneContext(
        probe_id=probe_id,
        platform="GPL80",
        status=status,
        gene_symbols=resolved_symbols,
        gene_symbol=symbol,
        ncbi_gene_id="5045" if symbol else None,
        gene_summary=summary,
        pathways=pathways or [],
        sources=["NCBI GEO GPL80 annotation", "NCBI Gene E-utilities", "KEGG REST API"],
        retrieved_at=datetime(2026, 11, 22, tzinfo=UTC),
    )


def test_generate_report_has_traceable_narrative_and_reasons():
    pathways = [
        PathwayRecord(pathway_id="hsa03250", name="Viral life cycle - HIV-1 - Homo sapiens (human)"),
        PathwayRecord(pathway_id="hsa04382", name="Cornified envelope formation - Homo sapiens (human)"),
    ]
    report = generate_xai_report(
        "golub-hu6800",
        [
            BiomarkerEvidence(
                context=_context(pathways=pathways),
                retention=RetentionEvidence(method="mrmr", rank=2, requested_k=3),
                signed_effect_size=1.25,
                effect_metric="signed Cohen's d",
            )
        ],
        positive_class="AML",
        negative_class="ALL",
    )

    biomarker = report.biomarkers[0]
    assert biomarker.evidence_level == "high"
    assert biomarker.direction == "higher in AML"
    assert "FURIN (X17094_at) was retained by mRMR at rank 2 (requested k=3)." in biomarker.narrative
    assert "Viral life cycle" in biomarker.narrative
    assert {reason.kind for reason in biomarker.reasons} >= {
        "retention",
        "effect",
        "annotation",
        "pathway",
        "gene_summary",
    }

    markdown = report.to_markdown()
    assert "# XAI biomarker report: golub-hu6800" in markdown
    assert "### 1. FURIN (`X17094_at`)" in markdown
    assert "**Retention:** mRMR rank 2" in markdown
    assert "**Evidence completeness:** high" in markdown
    assert "**Evidence**" in markdown
    assert "## Limitations" in markdown
    assert "not a clinical interpretation" in biomarker.narrative


def test_unresolved_mapping_withholds_gene_claims():
    report = generate_xai_report(
        "golub-hu6800",
        [
            BiomarkerEvidence(
                context=_context(
                    probe_id="AMBIG_at",
                    symbol=None,
                    status="ambiguous_probe_mapping",
                    symbols=["GENE1", "GENE2"],
                    summary=None,
                )
            )
        ],
        positive_class="AML",
        negative_class="ALL",
    )

    biomarker = report.biomarkers[0]
    assert biomarker.evidence_level == "low"
    assert "withheld" in biomarker.narrative
    annotation = next(reason for reason in biomarker.reasons if reason.kind == "annotation")
    assert annotation.strength == "limited"
    limitations = [reason for reason in biomarker.reasons if reason.kind == "limitation"]
    assert any("No KEGG pathway" in reason.statement for reason in limitations)


def test_enrichment_matches_pathways_and_leading_edges():
    report = generate_xai_report(
        "golub-hu6800",
        [
            BiomarkerEvidence(
                context=_context(
                    probe_id="LAMP2_at",
                    symbol="LAMP2",
                    pathways=[
                        PathwayRecord(
                            pathway_id="hsa04142",
                            name="Lysosome - Homo sapiens (human)",
                        )
                    ],
                ),
                signed_effect_size=2.0,
                effect_metric="signed Cohen's d",
            ),
            BiomarkerEvidence(
                context=_context(probe_id="FURIN_at", symbol="FURIN", summary=None),
            ),
        ],
        enrichment_terms=[
            EnrichmentTerm(term="Lysosome", nes=2.4, fdr=0.01, source="GSEA prerank"),
            EnrichmentTerm(term="Secretory pathway", lead_genes=["FURIN"], fdr=0.03),
        ],
        positive_class="AML",
        negative_class="ALL",
    )

    lysosome = report.biomarkers[0]
    furin = report.biomarkers[1]
    assert any(reason.kind == "enrichment" for reason in lysosome.reasons)
    assert any(reason.kind == "enrichment" for reason in furin.reasons)
    assert "agrees with this gene being higher in AML" in lysosome.narrative
    assert report.enrichment_summary is not None
    assert "2 enrichment terms were supplied; 2 matched" in report.enrichment_summary


def test_loaders_accept_report_shapes(tmp_path):
    contexts_path = tmp_path / "contexts.json"
    contexts_path.write_text(
        json.dumps({"records": [_context().model_dump(mode="json")]}), encoding="utf-8"
    )
    retention_path = tmp_path / "retention.json"
    retention_path.write_text(
        json.dumps({"X17094_at": {"method": "mrmr", "rank": 1, "requested_k": 3}}),
        encoding="utf-8",
    )
    enrichment_path = tmp_path / "enrichment.json"
    enrichment_path.write_text(
        json.dumps(
            {
                "prerank": {
                    "top_terms": [
                        {
                            "term": "Lysosome",
                            "nes": 2.4,
                            "fdr": 0.01,
                            "direction": "up_in_positive_class",
                            "lead_genes": "CTSA;CTSD",
                        }
                    ]
                },
                "ora": {
                    "top_terms": [
                        {"term": "Lysosome", "adjusted_p": 0.08, "genes": "CTSB;CTSC"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    contexts = load_contexts(contexts_path)
    retention = load_retention(retention_path)
    terms = load_enrichment_terms(enrichment_path)

    assert contexts[0].gene_symbol == "FURIN"
    assert retention["X17094_at"].rank == 1
    assert [term.source for term in terms] == ["GSEA prerank", "over-representation analysis"]
    assert terms[1].fdr == pytest.approx(0.08)
    assert terms[1].lead_genes == ["CTSB", "CTSC"]


def test_csv_loaders_filter_fdr_and_parse_genes(tmp_path):
    prerank = tmp_path / "prerank.csv"
    prerank.write_text(
        "Name,Term,ES,NES,NOM p-val,FDR q-val,Lead_genes\n"
        "Lysosome,Lysosome,0.8,2.4,0.0,0.001,CTSA;CTSD\n"
        "Weak,Weak,0.4,1.2,0.1,0.2,FOO\n",
        encoding="utf-8",
    )
    terms = load_enrichment_terms(prerank, fdr=0.05)

    assert len(terms) == 1
    assert terms[0].direction == "up_in_positive_class"
    assert terms[0].lead_genes == ["CTSA", "CTSD"]
    assert terms[0].source == "GSEA prerank"

    ora = tmp_path / "ora.csv"
    ora.write_text(
        "Term,Overlap,P-value,Adjusted P-value,Odds Ratio,Genes\n"
        "Lysosome,5/80,0.0007,0.08,4.0,CTSB;CTSC\n",
        encoding="utf-8",
    )

    assert load_enrichment_terms(ora, fdr=0.05) == []
    ora_terms = load_enrichment_terms(ora)
    assert ora_terms[0].lead_genes == ["CTSB", "CTSC"]
    assert ora_terms[0].source == "over-representation analysis"


def test_enrichment_direction_conflict_is_flagged():
    report = generate_xai_report(
        "golub-hu6800",
        [
            BiomarkerEvidence(
                context=_context(
                    probe_id="M85169_at",
                    symbol="CYTH1",
                    pathways=[
                        PathwayRecord(
                            pathway_id="hsa05131",
                            name="Shigellosis - Homo sapiens (human)",
                        )
                    ],
                ),
                signed_effect_size=-0.9,
                effect_metric="signed Cohen's d",
            )
        ],
        positive_class="AML",
        negative_class="ALL",
        enrichment_terms=[
            EnrichmentTerm(
                term="Shigellosis",
                nes=1.6,
                fdr=0.04,
                direction="up_in_positive_class",
                source="GSEA prerank",
            )
        ],
    )

    biomarker = report.biomarkers[0]
    enrichment = next(reason for reason in biomarker.reasons if reason.kind == "enrichment")
    assert "term direction: higher in AML" in enrichment.statement
    assert "FDR=0.04" in enrichment.statement
    assert "opposite to this gene's effect" in enrichment.statement
    assert "pathway membership rather than concordant support" in biomarker.narrative
    assert "| Shigellosis | 1.60 | 0.04 | higher in AML |" in report.to_markdown()


def test_mrmr_exposes_selection_order():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 10)
    signal = np.tile(y, (4, 1)).T * 3 + rng.normal(scale=0.1, size=(20, 4))
    noise = rng.normal(size=(20, 20))
    X = np.hstack([signal, noise])

    selector = MRMRSelector(top_k=3, candidate_pool=10).fit(X, y)

    assert len(selector.ranking_) == 3
    assert set(selector.ranking_) == set(selector.support_)
    assert selector.ranking_[0] == 0


def test_write_report_writes_markdown_and_json(tmp_path):
    report = generate_xai_report(
        "golub-hu6800",
        [BiomarkerEvidence(context=_context())],
        positive_class="AML",
        negative_class="ALL",
    )
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_report(report, markdown_path=markdown_path, json_path=json_path)

    assert markdown_path.read_text(encoding="utf-8").startswith("# XAI biomarker report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["biomarkers"][0]["gene_symbol"] == "FURIN"
    assert payload["limitations"]
