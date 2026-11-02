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

## Decided but not implemented

The following are accepted as **future experiments to investigate**, based on
the professor's suggestions. Recording an idea here does not mean it has been
coded, evaluated, or shown to improve accuracy.

| Decision | Why we want to investigate it | Implementation/evaluation still needed | Status |
| --- | --- | --- | --- |
| Adaptive hypergraph regularization | A data-selected graph penalty may work better than one fixed penalty. | Tune the graph strength only inside inner CV; record chosen strengths across outer folds. | Decided; not implemented |
| Class-aware hypergraph or gene selection | Use the ALL/AML distinction directly when scoring relevant genes and pathways. | Define a fold-local class-aware scoring rule. “Multi-class” only applies if additional disease classes are added. | Decided; not implemented |
| Complementary gene scoring | Prefer genes that add new predictive information, not several genes carrying nearly the same signal. | Define and test a conditional/complementarity score inside each training fold; compare selected-set redundancy and prediction metrics. | Decided; not implemented |
| Non-linear dependency model | Gene combinations may contain patterns that a linear classifier misses. | Compare a suitable non-linear candidate against the same baseline and splits, with all tuning nested. | Decided to investigate; not implemented |
| Gene-selection stability | Genes repeatedly selected across resamples may be more reliable than genes selected in only one split. | Track selection frequency across repeated nested CV and, if available, a separate cohort. | Decided; not implemented |
| Classifier-aware selection | A gene set useful to one classifier may not be useful to another. | Consider joint selector/classifier tuning after the preceding experiments; guard against an overly large search. | Deferred; not implemented |

### Order for future work

1. Audit the notebook to confirm preprocessing and feature selection are
   training-fold-only, then freeze the baseline and evaluation splits.
2. Implement adaptive regularization and class-aware/complementary scoring as
   separate experiments, one at a time.
3. Test non-linear models as separate candidates under the same nested
   evaluation.
4. Measure selection stability across repeats and independent data when
   available.
5. Revisit classifier-aware selection after these results are understood.

Do not describe any item above as an accuracy improvement until repeated
nested evaluation supports that conclusion.

## Decision log maintenance

When adding a decision, record its date, status (`decided; not implemented`,
`implemented`, `deferred`, or `rejected`), rationale, and evidence. Link to the
code/notebook and state the validation setup for any reported score. Keep
exploratory and external-test results distinct.
