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
- logistic regression with a hypergraph-Laplacian penalty.

Recorded results are summarized in
`docs/hypergraph_accuracy_experiments.md`. In brief, these experiments made
selected genes less redundant or added pathway interpretability, but did not
consistently improve accuracy over the gene-only baseline. The Laplacian
search chose zero graph penalty in each of its five folds. Some of these newer
experiments were evaluated on one outer-fold partition only, so their scores
need repeated nested validation.

### 7. Evaluate and report

- Compare models on identical data splits and report accuracy, balanced
  accuracy, and other relevant metrics.
- Report variability and fold-level outcomes, not just the best score.
- Record the feature-selection stability and pathway interpretation when
  evaluating future runs.
- Treat cross-validation as research evaluation, not clinical validation.

## Proposed next implementation flow

These steps are professor-suggested next experiments; they are **not yet
implemented**:

1. Audit the Colab notebook for leakage and verify every fit/selection step is
   fold-local.
2. Freeze a baseline, data splits, and metrics for fair comparison.
3. Add one candidate at a time: adaptive graph penalty, class-aware scoring,
   complementary-gene scoring, or a non-linear classifier.
4. Tune each candidate only in the inner training folds.
5. Run repeated nested CV, then compare stability of selected genes.
6. If possible, validate the final locked method on an independent cohort.
7. Consider classifier-aware feature selection after the earlier experiments
   are stable and understood.

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
