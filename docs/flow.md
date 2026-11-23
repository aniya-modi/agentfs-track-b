# Project Implementation Flow

This document describes the work implemented so far and how data moves
through the project. Proposed future experiments are marked separately.

## Architecture flow

The architecture has two related paths: a **data and prediction path** for
local expression analysis, and a **metadata/biological-context path** for
annotation and downstream agent handoffs. The paths meet through feature IDs
and pathway context; expression values and sample labels stay in local Python
selector tools.

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
       |       +--> mRMR / L1 selection / screened knockoff experiment   |
       |                     |                                          |
       |       +-------------+--------------+                           |
       |                     v                                          |
       |       Planner proposals -> local selector executor             |
       |                     |                                          |
       |       Fold-local nested/repeated CV                            |
       |                     |                                          |
       |                     v                                          |
       |       Metrics + selected genes/pathways                        |
       +----------------------------------------------------------------+
                                  |
                    selected IDs / aggregate metrics only
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
       | Bio-Refiner -> Challenger -> Judge                              |
       |        ^                         |         |                    |
       |        +---- bounded revision ---+         v                    |
       |                              fallback / decision                |
       |                                       |                        |
       |                               Track A / Track C handoff         |
       +----------------------------------------------------------------+
```

**Privacy/data boundary:** The local selector executor receives expression
matrices and labels. Ollama receives only aggregate metadata, candidate feature
IDs, CV summaries, and bounded Bio-RAG records. Bio-RAG receives mapped
biological identifiers, not expression values or sample labels.

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
Planner proposes selector configurations
        |
        v
Local Python executes mRMR, LASSO, and optional KEGG hypergraph candidates in nested CV
        |
        +--> fit exploratory final feature lists locally
        |
        +--> resolve a bounded set of selected IDs through Bio-RAG
        |
        v
Bio-Refiner reviews pathway records -> Challenger checks measured evidence -> Judge chooses an executed candidate
        |
        v
Print selected features, CV summaries, pathway support, and limitations
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

### 8. Month 2 selector and context workflow

1. Load the processed Golub matrix with GPL80 probe IDs and the ALL/AML labels.
2. Build a probe-by-KEGG-pathway incidence matrix using the matching GPL80
   annotation. The first API responses are cached under
   `data/reference/kegg_snapshot/`; their hashes are printed for provenance.
   For four-column KEGG gene-list rows, use the fourth names field to map
   symbols; retain the two-column fallback for older response formats.
3. Compare an F-score baseline, mRMR, L1 logistic selection, and the existing
   biological/statistical soft-redundancy selector. Imputation, selector
   fitting, scaling, and logistic tuning run inside nested CV folds.
4. Summarize prediction metrics and stability across repeated stratified outer
   folds. These are exploratory estimates on one small dataset.
5. Optionally run the supervised-screen/knockoff sequence. It reports the
   requested knockoff threshold but makes no formal FDR claim; if the rule
   selects no features, it does not force any through.
6. The combined example applies pathway-aware redundancy scoring to selected
   probes and resolves biological context through Bio-RAG. It uses all sample
   labels for an exploratory report, not held-out evaluation. Set `NCBI_EMAIL`
   for live NCBI requests.

### 9. Four-agent decision graph

1. Validate input with `AgentGraphInput`; the expression matrix and labels stay
   in a local executor closure, outside `AgentState` and model prompts.
2. Planner proposes configurations from the available selector catalog.
3. Local Python selector tools execute each proposal inside nested CV and fit
   a final exploratory feature list on the full supplied input. Only selected
   feature IDs and aggregate metrics enter graph state.
4. A bounded Bio-RAG resolver maps selected IDs through the matching platform
   annotation and retrieves NCBI/KEGG context when `NCBI_EMAIL` and annotation
   are available.
5. Bio-Refiner reviews actual resolved gene/pathway records; the graph filters
   claimed pathway-supported IDs against those records.
6. Challenger reviews measured performance, stability, redundancy, and pathway
   coverage. Judge chooses among candidates that were actually executed and
   returns that candidate's selected feature IDs.
7. Invalid agent responses receive bounded retries. Selector or Bio-RAG errors
   are surfaced without sending the matrix to Ollama. If every selector
   candidate fails, the graph marks the result as fallback and returns no
   feature list or CV metrics instead of implying selection succeeded.

`build_agent_graph` accepts four optional structured-output callables. The
built-in deterministic policies make the graph runnable without a local LLM.
`src/agentfs/ollama_runners.py` supplies four schema-constrained Ollama runners;
`src/agentfs/selection_pipeline.py` keeps data local while evaluating and
fitting selector candidates. `examples/run_local_agent_graph.py` supports
Golub and other labeled numeric CSVs, optional platform-matched annotations,
and bounded Bio-RAG lookups. It prints all role/tool outputs and the final
selected feature IDs. Its candidate CV scores help choose a method, so the
winning score is exploratory rather than an unbiased final estimate; use a
separate outer evaluation for final performance claims. Ollama uses
`http://127.0.0.1:11434` by default and supports `OLLAMA_HOST` and `OLLAMA_MODEL`
overrides.

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
| `src/agentfs/agent_state.py` | Typed metadata-only state for the agent graph |
| `src/agentfs/agent_graph.py` | Four-role LangGraph, Pydantic output validation, and bounded fallbacks |
| `src/agentfs/ollama_runners.py` | Four schema-constrained local Ollama runners |
| `src/agentfs/selection_pipeline.py` | Local selector execution and nested-CV candidate evidence |
| `examples/run_local_agent_graph.py` | Labeled numeric CSV demo with local selectors, optional Bio-RAG, and Ollama agents |
| `src/agentfs/hypergraph_experiments.py` | Fold-local pathway and redundancy experiment components |
| `src/agentfs/baseline_selectors.py` | Fold-local mRMR and L1 logistic feature selectors |
| `src/agentfs/evaluation.py` | Repeated nested-CV metrics and feature stability |
| `src/agentfs/knockoff_selector.py` | Exploratory supervised screening and knockoff selection |
| `src/agentfs/kegg.py` | Cached KEGG REST responses and probe/pathway incidence builder |
| `examples/benchmark_feature_selectors.py` | Repeated nested-CV benchmark for feature selectors |
| `examples/generate_gene_pathway_report.py` | Exploratory selected-probe and Bio-RAG report |
| `baseline_benchmark.ipynb` | Repository benchmark notebook; the detailed Colab workflow may be maintained separately |

## Visual film and presentation flow

`demo/index.html` opens with a short animated film. It uses motion to make the
data and agent hand-offs visible; it does not load expression values or invoke
the model.

```text
72 Golub samples + 1,000 processed probe features
        ↓ animate as sample particles
GPL80 probe-to-symbol map + KEGG pathway membership
        ↓ draw multi-probe hyperedges and shared nodes
Fold-local relevance + pathway-context feature ranking
        ↓ shortlist cards move to the agent workflow
Planner proposes → local Python selectors execute
        ↓ candidate IDs and bounded metrics
Bio-Refiner checks mapped context → Challenger checks evidence
        ↓ Judge selects among executed candidates
Reviewable feature set + rationale + limitations
```

The animation has seven short chapters: data, hypergraph, feature ranking,
Planner, local selector, Bio-Refiner/Challenger, and Judge. Custom illustrated
human-like figures make the four agent roles visible; a moving card represents
the bounded result hand-off. Play, pause, replay, and timeline seeking are
available. Agent figures, arrows, pathway edges, and nodes are static; only
one-time entrances remain for data, bars, and shortlist elements. Presenter
names and speaking times are omitted from the site. A detailed walkthrough is
also available for explaining the dataset, selector, experiment, and agents.
The hypergraph view shows mapped GPL80 probe/gene examples
that belong to both KEGG MAPK (`hsa04010`) and NF-κB (`hsa04064`) pathways. The
page reports the current full GPL80 annotation/snapshot totals (7,129 probe
IDs, 373 pathways, 4,555 probes with membership, 31,513 memberships) separately
from the processed Golub benchmark (72 samples × 1,000 high-variance probes).

The detail walkthrough reports the Colab complementary selector's mean balanced
accuracy (0.9589 vs 0.9507) and selected-set correlation (0.3395 vs 0.4701)
across 3 repeats × 5 outer folds. It explicitly distinguishes those
experimental figures from the repository's `hypergraph_centrality` method and
the separate agent-graph run. Film rank bars are illustrative, not recorded
gene scores; this visual animation is not live model execution. Live agent
outputs are obtained by running `python examples/run_local_agent_graph.py` in
the repository environment.
