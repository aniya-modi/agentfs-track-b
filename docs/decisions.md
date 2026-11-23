# Project Decisions and Rationale

This file records decisions made for the microarray and pathway-informed
feature-selection work, why they were made, and what the experiments showed.
It separates implemented work from ideas that are still proposals. Results
below are exploratory and apply to the stated dataset and validation setup.

## Dataset and identifiers

### Use the Golub leukemia data for the current binary experiment

- **Decision:** Use the Golub microarray data to distinguish ALL from AML.
- **Why:** It is the current project benchmark and has known sample labels.
- **What it means:** ALL and AML are labels for patient samples, not groups into
  which genes are divided.
- **Status:** Implemented in the notebook experiments. The matrix has 72
  samples, 7,129 GPL80 probe features, 47 ALL samples, and 25 AML samples.

### Preserve real probe IDs and use platform-matched annotation

- **Decision:** Keep original probe-set IDs in the expression matrix and use
  the annotation matching that array platform when resolving genes.
- **Why:** Generic names such as `GENE_1` cannot be reliably looked up, and
  probe IDs from different platforms are not interchangeable.
- **What helped:** Restoring IDs makes the pipeline reproducible and supports
  biologically meaningful NCBI/KEGG lookups. It improves traceability; it is
  not itself an accuracy improvement.
- **Status:** Implemented for the Golub Hu6800/GPL80 data. Its annotation is
  stored at `data/annotation/gpl80_probe_to_gene_symbol.tsv`. Colon `att_n`
  feature labels are source positions, not verified gene IDs.

## Prediction and evaluation

### Keep a simple gene-based model as the benchmark

- **Decision:** Compare pathway and hypergraph methods with a regularized
  logistic-regression baseline using statistically ranked genes.
- **Why:** A baseline shows whether added biological structure provides
  predictive value beyond a simpler model.
- **What helped:** The baseline gives a clear, reproducible comparison point.
  In the expanded classifier search, logistic regression was selected in each
  outer fold.
- **Accuracy outcome:** More complex approaches did not consistently beat this
  baseline in the recorded runs.

### Keep feature learning inside cross-validation folds

- **Decision:** Fit supervised gene scoring, feature selection, scaling,
  correlation estimates, and model tuning using training folds only.
- **Why:** Letting test-fold labels influence these steps can inflate the
  measured score, especially with thousands of features and few samples.
- **Status:** This is the required evaluation rule for the reusable experiment
  estimators and all future comparisons. Audit the Colab notebook to confirm
  every preprocessing and selection step follows it.

### Treat current scores as exploratory, not clinical evidence

- **Decision:** Do not claim the project is ready to diagnose patients based on
  cross-validation scores from this small dataset.
- **Why:** 72 samples provide limited evidence about generalization. An
  independent cohort and clinical validation would be required before any
  clinical use.
- **Evidence:** The earlier hybrid-vs-univariate comparison had the same
  repeated nested-CV scores (0.9483 accuracy and 0.9507 balanced accuracy for
  both). Newer activity, redundancy, and Laplacian experiments were reported
  on one five-fold outer partition and need repeated validation.

## Hypergraph design and experiments

### Represent pathways as hyperedges

- **Decision:** Treat each gene/probe as a node and each biological pathway as
  a hyperedge connecting all of its member genes.
- **Why:** A pathway can relate many genes together, while an ordinary edge
  represents a pair. The incidence matrix records membership: rows are
  features, columns are pathways, and a nonzero entry means membership.
- **Status:** Implemented in the Colab analysis and represented by the `H`
  pathway-incidence input used by `src/agentfs/hypergraph_experiments.py`.

### Combine statistical and pathway evidence

- **Decision tested:** Blend a gene-level statistical rank with a
  pathway-propagated rank, then select features.
- **Formula:** `score = alpha * gene_rank + (1 - alpha) * pathway_rank`.
- **Why:** Statistical evidence reflects separation of ALL/AML in training
  samples; pathway evidence favors genes connected to relevant pathway groups.
- **Outcome:** The tested blend did not improve accuracy over the matched
  univariate baseline. It remains useful as an interpretable experiment, not a
  demonstrated accuracy gain.

### Reduce redundant selected genes

- **Decision tested:** Add hard correlation pruning or a soft redundancy
  penalty so the selected set contains less repetitive expression patterns.
- **Why:** Several highly correlated genes may carry overlapping information.
- **Outcome:** Both variants reduced average pairwise correlation, but scored
  lower than the matched baseline in the reported partition. This helped
  diversify the selected set, not prediction accuracy in that run.

### Add pathway activity and hypergraph smoothness

- **Decision tested:** Summarize pathway expression as mean standardized
  expression, and add a hypergraph-Laplacian penalty to logistic regression.
- **Why:** These methods ask whether group-level pathway signals or pathway
  membership smoothness help prediction.
- **Outcome:** Pathway activity was lower than the gene-only baseline in the
  reported split. Laplacian tuning selected `lambda_h = 0` in all five folds,
  effectively ignoring the graph penalty. Neither showed an accuracy gain.

### Choose the hypergraph penalty from training folds

- **Decision:** Tune `lambda_h` among candidate values with inner
  `GridSearchCV`, scored by balanced accuracy, rather than selecting one value
  before evaluation.
- **Why:** The amount of pathway smoothing should be selected using training
  data only. Including zero lets the model fall back to the gene-only form if
  pathway smoothing does not help.
- **Implementation:** This is already in the Colab hypergraph-Laplacian
  experiment (Approach 3, Cell 24). It searches candidate graph strengths
  inside each outer training fold, then evaluates the refitted choice on the
  held-out outer fold.
- **Outcome:** In the initial five-fold run, `lambda_h = 0` was selected in all
  five folds. The repeated nested-CV comparison (3 repeats × 5 outer folds)
  also selected `lambda_h = 0` in all 15 folds. Mean balanced accuracy was
  0.9522 for the adaptive estimator and 0.9507 for the matched gene-only
  baseline; paired mean differences by repeat were +0.0111, -0.0089, and
  +0.0022. Since the graph penalty was zero in every fold, these small score
  differences do not show a benefit from pathway regularization. They may
  reflect implementation or optimizer differences and should not be claimed
  as a hypergraph gain.
- **Runtime fix:** Colab raised `BrokenProcessPool` because process workers
  could not deserialize the notebook-defined `expit` reference when the
  Laplacian `GridSearchCV` used `n_jobs=-1`. Both searches in the repeated
  comparison now use `n_jobs=1`, which avoids that serialization path. A prior
  `NameError: X is not defined` was due to a restarted Colab runtime; rerunning
  prerequisite cells restored the variables.
- **Status:** Implemented and evaluated in Colab Approach 3, Cell 24 and
  Approach 4, Cell 25. A readable output summary was added in Cell 26. The
  notebook edits are saved; no repository estimator or source code was
  changed.

### Score genes and pathways for the ALL-versus-AML distinction

- **Decision:** Add a binary class-aware selector that ranks probes by the
  absolute ALL-versus-AML effect size, then blends that rank with a pathway
  score derived from member-gene effects.
- **Formula:** Per probe, signed Cohen's `d` is
  `(mean_AML - mean_ALL) / pooled_standard_deviation`. The signed value shows
  which class has higher expression; its absolute value ranks separation. A
  pathway score is the mean of the strongest 25% absolute member-gene effects.
  The selector blends normalized gene and pathway ranks as
  `alpha * gene_rank + (1 - alpha) * propagated_pathway_rank`.
- **Why:** This makes the selection target explicit: genes and pathways should
  help distinguish ALL from AML. If further cancer classes are added later,
  the scoring rule will need a multi-class extension.
- **Validation:** The scores are calculated inside each model's training fold.
  `alpha`, selected feature count, and logistic-regression `C` are tuned by
  inner CV. The matched F-score/logistic baseline uses identical repeated
  outer folds (3 repeats × 5 folds).
- **Outcome:** Mean balanced accuracy was 0.9552 for class-aware gene plus
  pathway selection and 0.9544 for the baseline. Paired mean differences by
  repeat were +0.0111, 0.0000, and -0.0089. This small, inconsistent change
  does not establish an accuracy improvement. The 20 most frequently selected
  probes appeared in all 15 outer folds in this run; selection consistency
  alone does not establish biological or clinical validity.
- **Status:** Implemented and evaluated in Colab Approach 5, Cell 27. The
  notebook changes are saved; no repository estimator or source code was
  changed.

### Reward complementary genes and discount repeated expression signals

- **Decision:** Add a greedy selector that starts from fold-local ALL/AML gene
  and pathway relevance, then reduces a candidate's score when it correlates
  strongly with a gene already selected.
- **Formula:** `score(g | S) = relevance(g) * (1 - gamma *
  max_{s in S} |corr(g,s)|)`. Here `relevance` blends normalized absolute
  Cohen's-d gene ranks and pathway-propagated ranks; `gamma` controls the
  redundancy penalty and is selected in inner CV. The first gene is selected
  by relevance alone. This pairwise-correlation proxy is an mRMR-style
  approximation; it does not model all higher-order interactions.
- **Why:** A large differential signal is useful, but a gene that repeats the
  expression pattern of an already selected gene may add little information.
  Greedy selection favors relevant genes that contribute a less redundant
  signal.
- **Validation:** Compared with a matched F-score/logistic baseline on the same
  3-repeat × 5-fold outer splits. Feature count, pathway blend, redundancy
  penalty, and classifier strength were tuned within inner CV. Redundancy was
  summarized as mean absolute pairwise correlation among selected probes,
  calculated on outer training folds only.
- **Outcome:** Mean balanced accuracy was 0.9589 for complementary gene plus
  pathway selection versus 0.9507 for the baseline; paired differences by
  repeat were +0.0111, +0.0111, and +0.0022. Mean selected-set correlation was
  0.3395 versus 0.4701 for the top-k F-score genes at matched feature count
  (about 27.8% lower). The inner search selected a nonzero `gamma` in 8 of 15
  folds. This is a promising result on one small dataset, not independent or
  clinical validation and not proof of a general accuracy gain.
- **Status:** Implemented and evaluated in Colab Approach 6, Cell 28. The
  notebook changes are saved; no repository estimator or source code was
  changed.

### Test for nonlinear gene-combination patterns

- **Decision:** Compare regularized logistic regression with an RBF-kernel
  support-vector machine (SVM) on the same Golub data and repeated outer
  folds. Both models use the same fold-local F-score gene selector,
  imputation, scaling, and nested tuning. The SVM's `C` and `gamma` are chosen
  using inner CV only.
- **Why:** Logistic regression learns a linear boundary. An RBF SVM can learn
  curved boundaries and interactions among the selected genes, so this is a
  first check for patterns the linear model may miss.
- **Validation:** Colab Approach 7, Cell 29; Golub ALL/AML (72 samples,
  7,129 probes), 3 repeats × 5 stratified outer folds, with 3-fold inner CV
  for tuning. On each outer training fold, both models tune the selected gene
  count from `[50, 100, 200, 500]` and `C` from `[0.01, 0.1, 1, 10]`; the RBF
  model also tunes `gamma` from `['scale', 0.001, 0.01, 0.1]`. The outer test
  fold is held out for scoring.
- **Outcome:** Mean accuracy was 0.9533 for both models. Mean balanced
  accuracy was 0.9552 for logistic regression and 0.9522 for RBF SVM; mean
  macro F1 was 0.9494 and 0.9488, respectively. Repeat-wise paired balanced-
  accuracy differences (RBF minus linear) were 0.0000, -0.0100, and +0.0011.
  This experiment did not show a predictive gain from using the nonlinear
  classifier on the selected genes.
- **Limitation:** The F-score preselection favors genes with individual
  class-separation signals. It can discard genes that are useful only through
  interactions. Therefore this result does not rule out nonlinear patterns
  among genes outside that candidate set, or nonlinear effects within a
  pathway-aware selected set. The next useful comparison is to apply the RBF
  model to the fold-local class-aware/complementary hypergraph-selected genes
  and compare against logistic regression using the same selected features and
  outer folds.
- **Status:** Implemented and evaluated in Colab Approach 7, Cell 29. This is
  a notebook experiment; no repository source code was changed.

### Direct non-linear comparison with complementary hypergraph selection

- **Decision:** Test whether an RBF SVM helps when both nonlinear and linear
  classifiers receive features from the same fold-local complementary
  hypergraph selector.
- **Why:** The earlier nonlinear screen used only univariate F-score selection
  and could have excluded genes useful through interactions. This direct test
  checks the classifier on genes selected with the pathway-aware,
  class-aware, redundancy-reducing method already evaluated in Approach 6.
- **Validation:** Colab Approach 8, Cell 30; Golub ALL/AML (72 samples,
  7,129 probes), 3 repeats × 5 stratified outer folds, with 2-fold inner CV.
  Both pipelines used fold-local median imputation, the same
  `ComplementaryGeneSelector(H=H)`, and standard scaling. The displayed output
  reports tuning feature count, pathway blend, redundancy weight, and
  classifier settings with balanced accuracy as the inner-CV metric; outer
  folds were used for final scoring. The exact feature-count grid needs
  reconciliation with the current cell source before the run is reproducible.
- **Outcome:** Linear logistic regression averaged 0.9584 accuracy (SD
  0.0510), 0.9589 balanced accuracy (SD 0.0482), and 0.9550 macro F1 (SD
  0.0539). RBF SVM averaged 0.9540 accuracy (SD 0.0552), 0.9459 balanced
  accuracy (SD 0.0693), and 0.9477 macro F1 (SD 0.0646). Paired outer-fold
  balanced-accuracy differences (RBF minus logistic), averaged within each
  repeat, were +0.0211, -0.0300, and -0.0300. The nonlinear model did not
  improve the overall average; logistic regression was higher on all three
  aggregate metrics.
- **Limitation:** This is an exploratory repeated-CV estimate on one small
  dataset, not independent or clinical validation. Repeated folds overlap in
  their training samples, so they are not 15 independent experiments. The
  result does not show that all nonlinear gene interactions are absent.
- **Reproducibility check:** The completed Cell 30 output reports inner-CV
  selections with `selector__top_k=200`, but the currently visible cell source
  searches only `[50, 100]`. The displayed output therefore does not match the
  current source. The scores below are provisional until Cell 30 is rerun with
  the source and output aligned; confirm the selected feature-count grid in
  that run before treating these metrics as reproducible.
- **Status:** A run completed in Colab Approach 8, Cell 30, but source/output
  reconciliation and a matching rerun are pending. No repository source code
  was changed.

## Month 2 selectors and benchmark implementation (2026-11-16)

### Add mRMR and explicit L1 selectors

- **Decision:** Compare F-score selection with mRMR and L1-regularized logistic
  feature ranking.
- **Why:** mRMR trades off ALL/AML relevance against correlated duplicate
  signals. L1 logistic regression provides a sparse-model-based ranking.
- **Implementation:** `MRMRSelector` learns ANOVA relevance and feature
  correlations in `fit`; `LassoSelector` standardizes training features and
  explicitly sets `penalty="l1"` with the `liblinear` solver. Both expose
  sklearn-compatible `transform` and `get_support` methods.
- **Status:** Implemented in `src/agentfs/baseline_selectors.py`. This change
  does not claim either selector improves accuracy; results must come from the
  repeated nested-CV benchmark.

### Evaluate selectors using repeated nested cross-validation

- **Decision:** Keep imputation, selection, scaling, model tuning, and
  classifier fitting inside nested CV. Report accuracy, balanced accuracy,
  macro F1, and feature stability across repeated outer folds.
- **Why:** With 7,129 probes and only 72 samples, fitting selection on all
  samples before splitting would leak label information into evaluation.
- **Implementation:** `run_repeated_nested_cv` uses 3 inner folds and repeated
  stratified outer folds by default. It records selected indices, Jaccard
  stability, and the best settings per fold. The benchmark separately
  calculates Kuncheva stability for fixed-size selections.
- **Status:** Implemented in `src/agentfs/evaluation.py` and
  `examples/benchmark_feature_selectors.py`. No benchmark run is recorded by this code
  change.

### Treat screened knockoffs as exploratory

- **Decision:** Do not report formal FDR control for a workflow that first
  selects candidates using ANOVA or mutual information computed with class
  labels.
- **Why:** The label-informed screen and unverified Gaussian Model-X
  assumptions do not establish a formal FDR guarantee. Forcing a minimum
  number of selected probes would also break fidelity to knockoff rejections.
- **Implementation:** `TwoStageKnockoffFilter` reports the threshold as a
  requested value, sets `formal_fdr_control_` to false, and returns exactly the
  knockoff rejections without a minimum-feature fallback. `knockpy` is an
  optional analysis dependency.
- **Status:** Implemented in `src/agentfs/knockoff_selector.py` and
  `pyproject.toml`. It is an exploratory selector, not a validated biomarker
  or diagnostic method.

### Preserve KEGG inputs and make the joint report explicitly exploratory

- **Decision:** Cache the KEGG API responses used to make pathway memberships,
  and label the full-data probe report as exploratory.
- **Why:** Live pathway data can change between runs, and the joint report uses
  the full Golub labels to choose probes; it is not a held-out performance
  estimate. It must not be described as clinical evidence.
- **Implementation:** `benchmark_feature_selectors.py` saves the KEGG responses and
  prints their SHA-256 hashes. `generate_gene_pathway_report.py` records its full-data
  selection status, requested knockoff threshold, and absence of formal FDR
  control in the JSON report. Live NCBI lookup requires `NCBI_EMAIL` from the
  environment.
- **Dependency boundary:** The joint example currently uses the existing
  `BioStatSoftRedundancySelector` from `main`. It does not depend on or change
  the centrality selector being brought in separately by `e35d11d`; it can be
  switched to that selector after the planned merge.
- **Status:** Implemented in the two Month 2 examples. No live API request or
  output report was generated during implementation.

## Four-agent orchestration (2026-11-17)

### Build a validated Planner, Bio-Refiner, Challenger, and Judge graph

- **Decision:** Use LangGraph to coordinate four roles around local selector
  tools: Planner proposes selector configurations, Bio-Refiner reviews the
  executed feature candidates with supplied gene/pathway context, Challenger
  examines measured validation and stability evidence, and Judge chooses among
  executed candidates or requests revision.
- **Why:** Role separation makes proposals, biological review, criticism, and
  final selection traceable. Pydantic V2 output models prevent malformed
  configurations from silently entering the handoff.
- **Fallback policy:** Retry invalid agent outputs up to the input's
  `max_agent_retries`. Judge revision requests loop through the graph up to
  `max_revision_rounds`; after that, select the best executed candidate by
  nested-CV balanced accuracy, macro-F1, stability, and pathway coverage, and
  mark the result as a fallback.
- **Data boundary:** Agent state and runner context carry dataset metadata,
  candidate feature IDs, measured summaries, and supplied Bio-RAG records.
  Expression values and sample-level labels stay in the local selector
  executor closure and never enter model context.
- **Status:** Implemented across `agent_graph.py`, `agent_state.py`,
  `schemas.py`, `ollama_runners.py`, and `selection_pipeline.py`. The demo now
  executes candidate selectors inside nested CV, fits final exploratory
  feature lists locally, resolves a bounded set of selected IDs through
  optional Bio-RAG, and prints all agent/tool outputs. This integrated path has
  not yet been run end-to-end after the change.
- **Ollama setup:** Ollama was installed with Homebrew and its `llama3.2` model
  was pulled successfully. The runner uses Ollama's local chat endpoint with
  each role's Pydantic JSON schema and validates responses again in the graph.
  Live Bio-RAG requires a platform annotation and `NCBI_EMAIL`.
- **Demo visibility:** Print each validated role output in the example so the
  audience can inspect each handoff, rather than only seeing the Judge's final
  decision. This makes the agent workflow auditable and helps distinguish
  measured feature selection from a medical diagnosis.
- **Execution failure behavior:** If a local executor was configured but no
  candidate completed, return a zero-confidence fallback with no selected IDs
  or CV metrics. Do not label a configuration-only recommendation as completed
  feature selection.

### Generalize the project around supervised numeric feature selection

- **Decision:** Accept labeled numeric tabular CSV inputs with any number of
  classes. Keep microarray platform annotations optional; enable KEGG
  hypergraph selection only when the user supplies a matching annotation.
- **Why:** This makes the selector workflow reusable beyond one cancer dataset
  without claiming that the current code supports every medical modality.
- **Data flow:** The local Python executor owns the feature matrix and target.
  LLM agents receive feature identifiers, compact nested-CV results, and
  bounded biological contexts only.
- **Evaluation boundary:** The Judge uses candidate CV summaries to select a
  method, so the winning candidate score is for agent review and is not a
  separate unbiased estimate. A separate outer evaluation is required for
  final performance claims.
- **Status:** Generic numeric CSV arguments and optional annotation support
  are implemented in the demo. Categorical encoders, images, clinical text,
  unsupervised selection, and external-cohort evaluation remain future work.

## Decided but not implemented

The following are accepted as **future experiments to investigate**, based on
the professor's suggestions. Recording an idea here does not mean it has been
coded, evaluated, or shown to improve accuracy.

| Decision | Why we want to investigate it | Implementation/evaluation still needed | Status |
| --- | --- | --- | --- |
| A different adaptive penalty rule beyond the existing CV grid | The current CV grid chose zero in each fold; a richer rule may be worth exploring only if it has a clear rationale. | Specify a distinct formula first, then compare it with the current inner-CV-tuned method and zero-penalty baseline using repeated nested CV. | Deferred; no distinct method implemented |
| Non-linear model on hypergraph-selected features | The first RBF-SVM screen on F-score-selected genes did not outperform logistic regression, but F-score preselection can miss interaction-only genes. | Align Cell 30 source and output, rerun the direct comparison, then use the matching result as exploratory evidence and seek external validation. | Run completed; reproducibility check and source-aligned rerun pending |
| Gene-selection stability | Genes repeatedly selected across resamples may be more reliable than genes selected in only one split. | Track selection frequency across repeated nested CV and, if available, a separate cohort. | Decided; not implemented |
| Classifier-aware selection | A gene set useful to one classifier may not be useful to another. | Consider joint selector/classifier tuning after the preceding experiments; guard against an overly large search. | Deferred; not implemented |

### Order for future work

1. Audit the notebook to confirm preprocessing and feature selection are
   training-fold-only, then freeze the baseline and evaluation splits.
2. Review why the inner search selects zero in every fold; inspect the
   objective, penalty scale, and candidate grid before claiming pathway
   regularization is contributing.
3. Only if justified, define a distinct adaptive-penalty rule and compare it
   with the existing inner-CV grid on the same repeated outer splits.
4. Measure selection stability across repeats and independent data when
   available.
5. Revisit classifier-aware selection after these results are understood.

Do not describe any item above as an accuracy improvement until repeated
nested evaluation supports that conclusion.

## Presentation delivery

### Use a visual animation film for the project demonstration

- **Decision:** Lead with a short, locally hosted animated film, then keep the
  detailed walkthrough available for presenters. The film animates the sample
  matrix, pathway hyperedges, feature ranking, and a hand-off among four
  illustrated human-like agents.
- **Why:** The project should be understood through visible movement and
  relationships, not a sequence of text-heavy panels. The animated hypergraph
  shows shared probe membership, while the agent scene makes each role's work
  and hand-offs visible.
- **Motion refinement:** Remove looping movement from agent illustrations,
  arrows, nodes, and pathway outlines after review found them flickery. Keep
  only one-time scene, data, bar, and shortlist entrances that explain the
  workflow. Remove presenter names and talk timings from the website.
- **Evidence shown:** The page uses the verified GPL80/KEGG incidence counts,
  the selector formula from `HypergraphCentralitySelector`, and the recorded
  3-repeat × 5-fold Colab result for the class-aware complementary selector.
  It labels that Colab method separately from the repository's current
  `hypergraph_centrality` tool.
- **Scope boundary:** The film is an explanatory animation, not a recording or
  live execution of feature selection, Ollama, or Bio-RAG. Illustrative rank
  bars are identified as illustrative. The live agent demo remains
  `python examples/run_local_agent_graph.py`.
- **Status:** Implemented in `demo/index.html` with play, pause, replay, and
  scene seeking; local launch instructions are in `README.md`.

## Decision log maintenance

When adding a decision, record its date, status (`decided; not implemented`,
`implemented`, `deferred`, or `rejected`), rationale, and evidence. Link to the
code/notebook and state the validation setup for any reported score. Keep
exploratory and external-test results distinct.

### Parse current KEGG gene-list records by their names field

- **Decision:** When a KEGG `list/{organism}` row has four tab-separated
  fields, parse the gene symbols from the fourth (names) field; retain the
  two-column fallback for older or alternate cached responses.
- **Why:** Treating the second field as the description read the gene type or
  genomic position as a symbol and severely undercounted probe-to-pathway
  memberships.
- **Evidence:** The recorded Golub mapping changed from 473 to 4,555 covered
  probes when rebuilt from the November 2026 KEGG snapshot. This supports the
  parser correction; it does not imply improved predictive accuracy.
- **Status:** Implemented in `src/agentfs/kegg.py`. The large KEGG snapshots
  and generated benchmark/report artifacts from the source commit are kept
  local and excluded from the new implementation commit.
