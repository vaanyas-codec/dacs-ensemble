# Confirmatory Evaluation Protocol (frozen before running)

This is a self-registered protocol. Its timestamp is the git commit that
introduces this file; no confirmatory result existed when it was written. It
is not an external registry entry.

## Why this exists

The competitive-margin rule (margin = 0.05) was designed *after* observing
that the earlier selection rule lost to soft voting on Banknote
Authentication, a dataset that was then included in the final 12-dataset
evaluation. The cross-validation protocol prevents test-label leakage, but
the method's design adapted to the benchmark itself, so the main study's
"4 of 12 significant wins" is not a clean held-out confirmation. This
protocol tests the method on datasets that played no part in its design.

## Hypotheses

- **H1.** DACS+Debate (margin 0.05) has higher accuracy than the soft-voting
  baseline on datasets not used in design.
- **H2 (ablation).** On the same datasets, margin 0.05 outperforms the
  original selection rule (margin disabled, i.e. margin = 1.0).

## Frozen configuration (no changes after the first run)

| Item | Value |
|---|---|
| Candidate pool | the 13 classifiers in `code/candidate_pool.py` |
| Ensemble size k | 4 |
| Minimum accuracy floor | 0.55 |
| Arm A (proposed) | competitive margin = 0.05 |
| Arm B (original rule) | competitive margin = 1.0 (margin disabled) |
| Reliability weight alpha | 2.0 |
| Beta grid | {0.1, 0.2, 0.3, 0.5, 0.7}, chosen per arm on an inner 75/25 split of the training fold only |
| Debate | max 8 rounds, tolerance 1e-4, disagreement threshold 0.02 |
| Selection CV | 3-fold |
| Baseline | soft voting over DecisionTree(depth 5), GaussianNB, kNN(5), LogisticRegression |
| Scaling | StandardScaler fit on the training fold only |
| Outer CV | repeated stratified 5-fold, 2 repeats (10 folds); 1 repeat (5 folds) for datasets subsampled to 3000 |
| Seeds | random_state = 42 throughout |

## Dataset inclusion rule (mechanical, fixed before any run)

Source: the public `jbrownlee/Datasets` repository. A file is included if
**all** of the following hold:

1. classification task with the label in the final column;
2. every feature numeric, no missing values;
3. at least 100 rows;
4. every class has at least 20 samples;
5. not used in the design or main 12-dataset evaluation;
6. if more than 3000 rows, subsample to 3000 (stratified, seed 42).

**Included (7):** iris, wine, new-thyroid, wheat-seeds, phoneme (to 3000),
mammography (to 3000), oil-spill.

**Excluded by rule:** ecoli, glass, winequality-red, winequality-white
(a class below 20 samples); abalone, breast-cancer, horse-colic, adult,
auto_imports (categorical features, missing values or non-classification);
housing, longley and all time-series files (not classification); and the
seven datasets from this repository already used in the main study.

No dataset will be added or removed after the first run.

## Analysis plan

Per dataset: McNemar test on pooled out-of-fold predictions for
(a) Arm A vs baseline and (b) Arm A vs Arm B, with Holm-Bonferroni
correction across the 7 datasets separately for each comparison.
Aggregate: number of datasets with positive accuracy difference, mean
difference, and an exact two-sided sign test across datasets.

## Pre-specified interpretation

H1 is treated as **supported** only if the mean difference is positive, at
least 5 of 7 datasets have a positive difference, and no dataset shows a
significant loss after Holm correction. Otherwise H1 is **not supported**
and the paper must say so. H2 follows the same rule. All results are
reported regardless of outcome.

## Known limitations of this test

- Several datasets (iris, wine, wheat-seeds, new-thyroid) are near ceiling,
  so power to detect small gains is low.
- Seven datasets is a small sample; the sign test can reach p < 0.05 only
  with 7 of 7 positive.
- Four datasets are multi-class, a setting the main study did not cover.
- The authors have general prior knowledge of how easy these datasets are.
