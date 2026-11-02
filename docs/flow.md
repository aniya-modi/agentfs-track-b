# Project Implementation Flow

This document describes the work implemented so far and how data moves
through the project. Proposed future experiments are marked separately.

## Architecture flow

The architecture has two related paths: a **data and prediction path** for
local expression analysis, and a **metadata/biological-context path** for
annotation and downstream agent handoffs. The paths meet through feature IDs
and pathway context; expression values and patient labels stay in the local
analysis path.

```text
                         RESEARCHER / NOTEBOOK
                                  |
                                  v
       +---------------- LOCAL DATA AND PREDICTION PATH ----------------+
       |                                                                |
       |  Reference / processed expression data                        |
       |             |                                                  |
       |             v                                                  |
       |  Data loader --> metadata profiler --> quality summary         |
       |             |                                                  |
       |             v                                                  |
       |  Preprocessing + stable probe IDs                              |
       |             |                                                  |
       |       +-----+----------------------+                           |
       |       |                            |                           |
       |       v                            v                           |
       |  Gene-only baseline         Pathway/hypergraph experiments     |
       |  rank + logistic regression  gene nodes + pathway hyperedges    |
       |       |                            |                           |
       |       +-------------+--------------+                           |
       |                     v                                          |
       |       Fold-local nested/repeated CV                            |
       |                     |                                          |
       |                     v                                          |
       |       Metrics + selected genes/pathways                        |
       +----------------------------------------------------------------+
                                  |
                    probe IDs / aggregate metadata only
                                  v
       +---------- BIOLOGICAL CONTEXT AND AGENT CONTRACT PATH -----------+
       | Matching platform annotation (e.g., GPL80)                     |
       |             |                                                  |
       |             v                                                  |
       | Probe-to-symbol resolution --> NCBI / KEGG Bio-RAG             |
       |             |                         |                        |
       |             +-------------------------+                        |
       |                                       v                        |
       |                 Validated schemas + typed AgentState           |
       |                                       |                        |
       |                               Track A / Track C handoff         |
       +----------------------------------------------------------------+
```

**Privacy/data boundary:** Bio-RAG receives mapped biological identifiers for
lookup, not expression matrices or sample labels. The profiler produces
aggregate metadata. Agent schemas carry metadata, identifiers, decisions, and
errors rather than expression arrays.

## Current end-to-end data flow

```text
Raw/reference microarray data
        |
        v
Preprocessing and stable feature IDs
        |
        +--> processed expression matrix + sample labels
        |
        +--> matching platform annotation (probe ID -> gene symbol)
        |          |
        |          v
        |      Bio-RAG NCBI/KEGG context lookup
        |
        v
Training-only feature selection and scaling inside CV
        |
        +--> baseline: statistical gene ranking + regularized logistic regression
        |
        +--> pathway/hypergraph experiments
        |
        v
Nested/repeated cross-validation comparison
        |
        v
Report prediction metrics, selected genes, pathway context, and limitations
```

## Implemented stages

### 1. Prepare and identify the datasets

- Golub leukemia is used for the current ALL-versus-AML experiment. Rows are
  samples; columns are expression features.
- The Golub Hu6800 matrix retains its original GPL80 probe-set IDs through
  processing, rather than replacing them with generated `GENE_1`-style names.
- The processed Golub and Colon matrices are stored under `data/processed/`.
- Colon `att_n` column labels are source-position identifiers; they are not
  treated as verified gene symbols.

### 2. Load data and inspect metadata

- `src/data_loader.py` loads supported expression data and can apply an
  explicitly supplied feature-ID mapping.
- `src/agentfs/metadata_profiler.py` reports dataset shape, missing/non-finite
  values, variance, target distribution, imbalance, and bounded correlation
  summaries.
- The profiler output is aggregate metadata and does not include expression
  values or sample IDs.

### 3. Resolve gene and pathway context

- The Bio-RAG client loads the annotation matching the dataset platform.
- For Golub, GPL80 probe IDs map through
  `data/annotation/gpl80_probe_to_gene_symbol.tsv` before NCBI/KEGG lookup.
- Unknown or ambiguous probe mappings remain explicit; the client does not
  guess gene symbols from opaque IDs.
- NCBI lookups require `NCBI_EMAIL`; live network lookup is a separate run-time
  step. Do not send expression values or patient labels to these services.

### 4. Establish a prediction baseline

- The baseline ranks genes using training data and fits regularized logistic
  regression to predict ALL versus AML.
- Preprocessing, feature selection, and tuning must be fitted within each
  training fold. Test folds are used only for evaluation.
- The recorded expanded search also compared a linear SVM; logistic regression
  was selected in every outer fold.

### 5. Construct the hypergraph

- Nodes represent expression features/genes.
- Each pathway is one hyperedge containing all member genes.
- A feature-by-pathway incidence matrix represents membership (1/member or
  0/non-member in the binary form used in the analysis).
- Pathway membership is used to combine gene evidence, construct pathway
  activity summaries, or regularize model weights.

### 6. Compare pathway-informed approaches

The Colab work and reusable prototypes in
`src/agentfs/hypergraph_experiments.py` cover:

- blending statistical gene ranks with pathway-propagated ranks;
- hard correlation pruning and soft redundancy penalties;
- pathway mean-activity features;
- logistic regression with a hypergraph-Laplacian penalty, with `lambda_h`
  selected by inner CV in the Colab notebook.

Recorded results are summarized in
`docs/hypergraph_accuracy_experiments.md`. In brief, these experiments made
selected genes less redundant or added pathway interpretability, but did not
consistently improve accuracy over the gene-only baseline. The Laplacian
search chose zero graph penalty in each of its five folds. Some of these newer
experiments were evaluated on one outer-fold partition only, so their scores
need repeated nested validation.

The adaptive penalty search exists in Colab Approach 3, Cell 24. Its five-fold
run selected `lambda_h = 0` on every fold. Approach 4, Cell 25 then compared
the estimator with a matched gene-only baseline across three repeats of five
outer folds. It selected `lambda_h = 0` on all 15 folds. Mean balanced
accuracy was 0.9522 for the adaptive estimator and 0.9507 for the baseline;
paired differences by repeat were +0.0111, -0.0089, and +0.0022. Because the
graph penalty was disabled by the inner search every time, this is not
evidence of a pathway-regularization accuracy gain. Cell 26 prints the compact
summary because Colab did not display Cell 25's output panel clearly.

The first Cell 24 execution failed with `BrokenProcessPool`: joblib worker
processes could not deserialize the notebook-defined `expit` symbol. Setting
the grid search to `n_jobs=1` fixed it. The initial `NameError` in Cell 25 was
from a restarted runtime with no live `X` variable; rerunning the setup,
annotation, hypergraph, and fold-building cells restored the state. The
notebook changes are saved. No repository estimator or source code was
changed.

Class-aware feature selection is implemented in Colab Approach 5, Cell 27.
Within each training fold it computes signed Cohen's `d` for every probe
(positive means higher in AML, negative means higher in ALL), aggregates the
strongest quarter of member-gene effects into each pathway score, and blends
the normalized gene and propagated pathway ranks. Inner CV chooses the blend
weight, feature count, and logistic-regression strength. On three repeats of
five outer folds, mean balanced accuracy was 0.9552 versus 0.9544 for the
matched univariate baseline. Repeat-wise paired differences were +0.0111,
0.0000, and -0.0089, so this does not establish a meaningful or consistent
accuracy gain. The most frequently selected 20 probes appeared in all 15
folds in this run. “Class-aware” here means binary ALL/AML; a later multi-class
problem needs an explicitly extended scoring rule.

Complementary gene scoring is implemented in Colab Approach 6, Cell 28. It
starts from the same fold-local class-aware gene/pathway relevance, then
greedily applies `score = relevance * (1 - gamma * maximum absolute
correlation with a selected probe)`. Inner CV chooses the gene/pathway blend,
feature count, redundancy penalty, and logistic-regression strength. Across
three repeats of five outer folds, mean balanced accuracy was 0.9589 versus
0.9507 for a matched univariate baseline; paired differences were +0.0111,
+0.0111, and +0.0022. Mean selected-set absolute pairwise correlation was
0.3395 versus 0.4701 for top-k F-score genes at the same feature count. Inner
CV chose a nonzero penalty in 8 of 15 folds. This is encouraging for this
benchmark but requires validation on additional cohorts and does not establish
clinical performance. Pairwise correlation is only a proxy for redundancy and
may miss higher-order gene interactions.

The first non-linear model screen is implemented in Colab Approach 7, Cell
29. It compares logistic regression with an RBF-kernel SVM on the same
F-score-selected genes, using repeated nested CV (3 repeats × 5 outer folds;
3 inner folds for tuning). Imputation, feature selection, scaling, and
classifier tuning are inside the training folds. Both methods averaged
0.9533 accuracy; logistic regression averaged 0.9552 balanced accuracy and
0.9494 macro F1, while RBF SVM averaged 0.9522 and 0.9488. The paired
balanced-accuracy difference (RBF minus linear) was 0.0000, -0.0100, and
+0.0011 by repeat. This screen did not show a non-linear accuracy gain among
the individually F-score-selected genes. Since this preselection may omit
interaction-only genes, the next direct hypergraph test is to compare RBF SVM
and logistic regression on the same fold-local class-aware/complementary
hypergraph-selected feature sets.

That direct test was run in Colab Approach 8, Cell 30. Both logistic
regression and RBF SVM used the same fold-local `ComplementaryGeneSelector`,
median imputation, scaling, and repeated outer splits (3 repeats × 5 folds).
Feature count, pathway blend, redundancy penalty, and each classifier's
settings were tuned in 2-fold inner CV. Logistic regression averaged 0.9584
accuracy (SD 0.0510), 0.9589 balanced accuracy (SD 0.0482), and 0.9550 macro
F1 (SD 0.0539). RBF SVM averaged 0.9540 accuracy (SD 0.0552), 0.9459
balanced accuracy (SD 0.0693), and 0.9477 macro F1 (SD 0.0646). Repeat-level
paired balanced-accuracy differences (RBF minus logistic) were +0.0211,
-0.0300, and -0.0300. On this small benchmark, the direct test did not show
an overall gain from the nonlinear classifier. This remains exploratory CV,
not external or clinical validation. A reproducibility issue remains: the
completed output includes inner-CV selections with `selector__top_k=200`, while
the currently visible Cell 30 source searches only `[50, 100]`. The reported
scores are provisional until the cell is rerun with matching source and output.
The run completed, but the exact metrics cannot yet be tied to the currently
visible source version.

### 7. Evaluate and report

- Compare models on identical data splits and report accuracy, balanced
  accuracy, and other relevant metrics.
- Report variability and fold-level outcomes, not just the best score.
- Record the feature-selection stability and pathway interpretation when
  evaluating future runs.
- Treat cross-validation as research evaluation, not clinical validation.

## Proposed next implementation flow

The nonlinear comparison has run, but its source/output reconciliation and
matching rerun remain pending; other experiments are still planned:

1. Audit the Colab notebook for leakage and verify every fit/selection step is
   fold-local.
2. Freeze a baseline, data splits, and metrics for fair comparison.
3. Investigate why the inner search consistently chooses zero, including the
   penalty scaling, objective, and candidate values.
4. Define a distinct adaptive-penalty rule only if it has a clear rationale;
   compare it with the existing inner-CV search on the same repeated splits.
5. Measure gene-selection stability across repeated splits and independent
   data when available.
6. Consider classifier-aware feature selection after earlier experiments are
   stable and understood.
7. If possible, validate the final locked method on an independent cohort.

## Project modules at a glance

| Path | Role |
| --- | --- |
| `src/data_loader.py` | Load expression datasets and preserve/apply feature IDs |
| `src/agentfs/metadata_profiler.py` | Produce privacy-conscious dataset quality metadata |
| `src/agentfs/bio_rag.py` | Resolve platform probes and retrieve NCBI/KEGG context |
| `src/agentfs/schemas.py` | Validate structured configs and biological context records |
| `src/agentfs/agent_state.py` | Define typed state for later agent orchestration |
| `src/agentfs/hypergraph_experiments.py` | Fold-local pathway and redundancy experiment components |
| `baseline_benchmark.ipynb` | Repository benchmark notebook; the detailed Colab workflow may be maintained separately |
