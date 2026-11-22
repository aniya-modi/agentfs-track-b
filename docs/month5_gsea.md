# Month 5: GSEA validation of selected gene subsets

## Objective

Statistically validate the project's selected Golub gene subsets with gene set
enrichment analysis, using `gseapy` and the KEGG 2021 human pathway library.
This module is descriptive pathway validation; it is not held-out model
evaluation and makes no clinical claim.

## Method

`src/agentfs/gsea.py` provides:

- signed Cohen's `d` per probe, with positive values meaning higher in AML;
- collapsing of multi-probe genes to one symbol per gene, keeping the effect
  with the largest absolute value;
- preranked GSEA through `gseapy.prerank` with 1,000 gene-set permutations,
  `min_size=15`, `max_size=500`, and a fixed seed;
- offline over-representation analysis (ORA) through `gseapy.enrichr`, using
  all annotated GPL80 symbols as the background; and
- FDR-filtered summaries for both tables.

`examples/run_gsea_validation.py` selects `top_k` probes with F-score, mRMR, or
L1 logistic selection, maps them through the GPL80 annotation, and runs both
analyses against the cached `KEGG_2021_Human` GMT
(`data/reference/gene_sets/KEGG_2021_Human.gmt`). The library is downloaded
from Enrichr once and reused offline.

Selection and ranking use the full 72-sample cohort, so these results are
exploratory. The earlier month-scale evaluation remains the source for
prediction metrics.

## Recorded run (2026-11-22)

Command:

```bash
python examples/run_gsea_validation.py --selector mrmr --top-k 50
```

Setup and outcome:

| Item | Value |
| --- | --- |
| Dataset | Golub, 72 samples, 7,129 probes, AML positive class |
| Selection | mRMR, top 50 probes |
| Selected symbols | 55 (50 probes after collapsing) |
| Ranked symbols | 6,069 |
| Library | KEGG_2021_Human, 320 sets cached |
| Prerank sets tested | 281 after size filtering |
| Prerank at FDR <= 0.05 | 65 |
| ORA sets tested | 113 |
| ORA at adjusted p <= 0.05 | 0 |

Top preranked terms:

| Term | NES | FDR | Direction |
| --- | ---: | ---: | --- |
| Lysosome | 2.63 | 0.00 | higher in AML |
| Spliceosome | -2.43 | 0.00 | higher in ALL |
| Rheumatoid arthritis | 2.41 | 0.00 | higher in AML |
| Legionellosis | 2.37 | 0.00 | higher in AML |
| Cell cycle | -2.36 | 0.00 | higher in ALL |
| DNA replication | -2.28 | 0.00 | higher in ALL |
| Phagosome | 2.27 | 0.00 | higher in AML |
| Proteasome | -2.25 | 0.00 | higher in ALL |

The leading AML-high terms are myeloid, lysosomal, phagosomal, and
immune-signalling programs; the ALL-high terms are cell-cycle, DNA-replication,
proteasome, and spliceosome programs. These directions are plausible for the
ALL-versus-AML contrast, but they come from one small cohort.

ORA did not reach FDR significance. Its strongest term was Lysosome
(overlap 5/80, nominal p = 0.00073, adjusted p = 0.082), followed by
Influenza A and transcriptional misregulation in cancer. This is consistent
with the selected mRMR subset being statistically strong but not a
pathway-coherent gene set; GSEA uses the full ranked list rather than only the
subset.

## Outputs

- `data/results/month5_gsea_validation.json` - run metadata and summaries.
- `data/results/month5_gsea_prerank.csv` - all tested gene sets.
- `data/results/month5_gsea_ora.csv` - all ORA results.
- `data/reference/gene_sets/KEGG_2021_Human.gmt` - cached library.

## Limitations

- Selection uses all samples; this is not nested-CV or external validation.
- Prerank p-values are permutation-based estimates on one small cohort.
- ORA and GSEA test different null hypotheses and can disagree, as they do here.
- Multi-symbol probes contribute each annotated symbol; unmapped probes are
  dropped, so symbol coverage affects both analyses.
- Gene-set content is tied to the cached Enrichr release; record the GMT file
  when reporting results.
- Enrichment does not establish diagnostic or clinical value.
