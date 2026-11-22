# Month 5: natural-language XAI biomarker reports

## Objective

Synthesize traceable, natural-language explanations for the biomarkers a
feature selector retains. The report explains why each probe was kept and what
is known about its gene, while keeping every statement tied to a supplied
source. It is deterministic research output, not a clinical interpretation.

## Method

`src/agentfs/xai_report.py` builds a validated report from four inputs:

- **Annotation context** - a `ProbeGeneContext` from Bio-RAG (platform mapping,
  NCBI Gene identifiers, summary, KEGG pathways).
- **Retention evidence** - the selector name, rank, and requested `k` that kept
  the probe, plus its selection score when available.
- **Effect evidence** - a signed effect size (the example uses signed Cohen's
  `d`, positive meaning higher in AML).
- **Enrichment evidence** - optional GSEA prerank or ORA terms, matched to a
  biomarker by pathway name or leading-edge gene. The example loads every term
  at FDR <= 0.05 from the full prerank and ORA result CSVs, not just the top-10
  summary.

Each biomarker receives a narrative, a list of typed reasons (`retention`,
`effect`, `annotation`, `pathway`, `gene_summary`, `enrichment`,
`limitation`), and citations. Reason strength reflects the source: strong for
platform annotation, NCBI Gene, KEGG, retention, and enrichment; moderate for
the effect size; limited for missing or unresolved fields.

Enrichment statements are direction-checked. A term that is enriched for genes
higher in the opposite class is reported as pathway membership, not as
concordant support, so the report never implies agreement that the data does
not show.

Evidence completeness is a field-count score, not a statistical confidence:

| Level | Condition |
| --- | --- |
| high | at least 5 strong evidence fields |
| moderate | 3-4 strong evidence fields |
| low | fewer than 3, or an unresolved platform mapping |

Unresolved or ambiguous mappings withhold gene-level claims instead of
inventing an explanation. The report also lists pathway themes that recur
across biomarkers and emits explicit limitations.

All contracts are Pydantic V2 models with `extra="forbid"` and frozen fields,
so hand-written or malformed report fields fail validation rather than leak
into the narrative. Generation uses no language model: the text is templated
from validated fields, which keeps every sentence reproducible and auditable.

## Recorded run (2026-11-22)

Command:

```bash
python examples/generate_xai_report.py
```

The example fits `MRMRSelector(top_k=3, candidate_pool=500)` on Golub, resolves
the retained probes through the saved Bio-RAG contexts, computes signed
Cohen's `d`, and loads enrichment terms from the Month 5 GSEA CSVs. It refuses
to run if the saved contexts do not match the probes the selector actually
retains.

| Item | Value |
| --- | --- |
| Dataset | Golub, 72 samples, AML positive class |
| Selector | mRMR, top 3 probes from a 500-probe pool |
| Retained probes | `X95735_at` (ZYX), `M85169_at` (CYTH1), `X17042_at` (SRGN) |
| Bio-RAG context | GPL80 annotation, NCBI Gene, KEGG (live lookup) |
| Enrichment terms supplied | 65 significant terms (FDR <= 0.05) |

Explanations:

- **ZYX** - rank 1, Cohen's `d` +2.70 (higher in AML), 2 KEGG pathways
  (Focal adhesion, Cytoskeleton in muscle cells), high completeness.
- **CYTH1** - rank 2, Cohen's `d` -0.90 (higher in ALL), 5 KEGG pathways.
  Three significant GSEA terms match its pathways (*Pathogenic E. coli
  infection* FDR 0.021, *Shigellosis* 0.0395, *Salmonella infection* 0.042),
  but all are enriched for AML-high genes, the opposite direction to CYTH1;
  the report states this explicitly instead of calling it support.
- **SRGN** - rank 3, Cohen's `d` +2.23 (higher in AML), no KEGG pathway
  returned, moderate completeness.

ZYX's *Focal adhesion* pathway was tested but did not reach the FDR threshold
(0.146), so it is not cited as enrichment evidence.

## Outputs

- `data/results/month5_xai_report.md` - readable report.
- `data/results/month5_xai_report.json` - full validated report for reuse.
- `data/results/month5_xai_biomarker_contexts.json` - saved Bio-RAG contexts
  used by the example.

## Limitations

- Explanations summarize supplied fields only; no literature search is added.
- Selection ranks and effect sizes come from the full cohort and are not
  held-out evidence.
- Pathway coverage depends on the KEGG annotation returned by Bio-RAG; genes
  without a KEGG record receive a limitation instead of a pathway claim.
- Enrichment matching uses pathway names and leading-edge genes. Name matching
  is deliberately conservative but can still link a gene to a broad pathway;
  direction is checked, causality is not implied.
- Evidence completeness measures field coverage, not statistical significance.
- The report is research software output, not a diagnostic device.
