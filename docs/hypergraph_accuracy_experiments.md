# Hypergraph accuracy experiments

This note records the Colab experiments run on the Golub leukemia dataset
(72 samples, 7,129 GPL80 probes; 47 ALL and 25 AML). The experiments reused the
same five outer folds and used nested inner cross-validation for parameter
selection. They are exploratory results from one outer-fold partition unless
the repeated-CV section says otherwise.

Reusable implementations for pathway activity, hard and soft redundancy
selection, and hypergraph-Laplacian logistic regression are in
`src/agentfs/hypergraph_experiments.py`. These models must be used inside a
training pipeline so imputation, scaling, supervised scoring, and correlation
estimation are fit on training folds only.

## Results

### Preprocessing and classifier search

| Experiment | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Raw expression, five-fold mean | 0.9714 | 0.9689 | 0.9689 |
| Signed-log2 expression, five-fold mean | 0.9438 | 0.9478 | 0.9396 |
| Hybrid selector, expanded classifier search | 0.9714 | 0.9689 | 0.9689 |
| Univariate F-score, expanded classifier search | 0.9714 | 0.9689 | 0.9689 |

The expanded search compared logistic regression and linear SVM. Logistic
regression was selected in every outer fold. A three-repeat, five-fold nested
comparison of the earlier hybrid and univariate selectors gave both methods
0.9483 accuracy, 0.9507 balanced accuracy, and 0.9448 macro-F1. Their paired
balanced-accuracy difference was zero in all three repeats. This repeated run
used two inner folds to keep the search tractable; it is not directly
interchangeable with the single-partition searches above.

### Biological scores and redundancy

| Experiment | Accuracy | Balanced accuracy | Mean absolute selected-gene correlation |
| --- | ---: | ---: | ---: |
| Univariate baseline (matched prototype search) | 0.9581 | 0.9589 | 0.4698 |
| Biological/statistical score with hard pruning | 0.9429 | 0.9467 | 0.3632 |
| Biological/statistical score with soft redundancy penalty | 0.9295 | 0.9367 | about 0.3013 |

Hard pruning reduced mean pairwise correlation by about 22.7%; the soft penalty
reduced it by about 35.9%. Neither improved prediction in this experiment. The
hard-pruning search selected `alpha=0.25` in each fold, where the score is
`alpha * gene_rank + (1 - alpha) * pathway_rank`. Thus pathway evidence had
greater weight than the gene-level score in that prototype.

### Pathway activity and Laplacian regularization

| Experiment | Accuracy | Balanced accuracy | Observation |
| --- | ---: | ---: | --- |
| Gene-only elastic-net baseline | 0.9714 | 0.9689 | — |
| Gene plus pathway-activity features | 0.9581 | 0.9589 | Lower than gene-only in this split |
| Hypergraph-Laplacian logistic regression | 0.9581 | 0.9589 | Inner CV chose `lambda_h=0` in all five folds |

Pathway activity was computed as mean fold-standardized expression over each
pathway's probe members. The Laplacian model combines weighted logistic loss,
L2 regularization, and a normalized hypergraph smoothness penalty.

## Interpretation and limits

The tested hypergraph approaches reduced redundancy or provided pathway
summaries, but did not improve predictive scores over gene-level selection on
these folds. The Laplacian search selected no graph penalty, and earlier hybrid
searches frequently selected the gene-only setting. This is evidence about
this dataset and these search spaces, not proof that pathway information can
never help.

Only the earlier hybrid-versus-univariate model comparison received repeated
nested CV. The newer activity, redundancy, and Laplacian prototypes were
evaluated on one five-fold outer partition. Validate a chosen candidate with
repeated nested CV and, if possible, an independent leukemia cohort before
claiming a general accuracy improvement. Scores from different inner-CV grids
should not be compared as though they were from one identical search.

The quantum-kernel idea remains unimplemented. It should be considered only as
an exploratory comparison after reducing features inside CV and against tuned
classical kernel baselines.
