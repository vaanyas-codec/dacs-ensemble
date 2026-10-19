# Pre-specified evaluation on new medical datasets (self-registered, frozen before any model was run)

Status: written before any classifier was fitted on these four datasets. Before this file was frozen, the datasets were
inspected only for size, class balance, number of missing cells, and the label-leak screen (`code/data_screen.py`).
This is a self-registered protocol, not an external registry entry. Its integrity rests on the SHA-256 recorded in
`PREREGISTRATION_HEALTHCARE.sha256`; publish both files in the repository before or with the results.

## Question
A reviewer suggested that the method might work better than soft voting on healthcare data. The main study gives no
support for that (mean gain +0.51 pp on its five medical datasets, +0.75 pp on the other seven), but those five datasets
were not selected for the purpose. This protocol tests the question on new medical datasets chosen in advance.
The authors had already seen the main-study results, so the hypothesis is informed by them; the new datasets are not.

## Datasets and selection rule
Rule: human clinical datasets with a binary patient outcome or diagnosis, obtainable from public repositories reachable
by the authors (scikit-learn, jbrownlee/Datasets, pycaret/pycaret, vincentarelbundock/Rdatasets), not already used in the
main or confirmatory study, with one row per patient. Chosen by name, size and column inspection only.

Included (4):
1. WDBC, malignant vs benign breast tumour (scikit-learn, 569 x 30)
2. UCI Hepatitis, died vs survived (pycaret copy, 154 x 19, 153 missing cells)
3. UCI Breast Cancer (Ljubljana), recurrence (jbrownlee copy, 286 x 41 after one-hot encoding)
4. Prostate-cancer recurrence after surgery (Rdatasets medicaldata::blood_storage, 316 x 15 after dropping the columns
   listed below, 33 missing cells)

Excluded, with reasons: Cleveland heart (cleve, heart_disease), Statlog heart, Pima/diabetes, Haberman, BUPA liver, original
Wisconsin breast cancer, new-thyroid and mammography (all already used or derived from used data); Parkinson's
telemonitoring (regression targets, repeated measures per subject); horse colic (veterinary); blood transfusion (donor
behaviour, not a patient outcome); indomethacin trial `indo_rct` (columns pep, status, bleed unclear and possibly derived
from the outcome); `survival::pbc` (censored survival outcome, three status levels).

Prostate dataset columns dropped before any model was run: row index, Censor and TimeToRecurrence (follow-up outcome
information), AnyAdjTherapy and AdjRadTherapy (therapy given after surgery). Ljubljana: '?' is kept as its own category;
one-hot encoding uses category labels only. Leak screen result: no imputation artefacts; one natural flag in Ljubljana
(a one-hot column covering 28 rows with 96.4% non-recurrence), judged a genuine category, kept.

## Protocol (identical to the main study)
Same code path as `experiment_final.fit_predict_fold`: median imputation and standardisation fitted on the training fold
only; repeated stratified 5-fold cross-validation, 2 repeats, seed 42; 13-classifier pool; margin 0.05; k = 4; alpha = 2;
beta chosen on an inner 75/25 split. Arms: V soft voting (DT depth 5, NB, kNN 5, LR), B best single pool member, W all 13
weighted, T top-4 by accuracy weighted, S margin selection without revision, A proposed method. Run with `code/run_healthcare.py`
and analysed with `code/analyze_healthcare.py`.

## Analysis plan
Per dataset: fold-mean accuracy for each arm; pooled out-of-fold McNemar test of A against V and of A against T. Holm
correction across the 4 datasets, separately for each of these two comparisons. Across datasets: mean difference
A-V, A-T and S-T. No other tests are planned; any further analysis will be labelled exploratory.

## Decision rule (fixed in advance)
- "The method works better than soft voting on these medical datasets" is SUPPORTED only if A is significantly better than
  V after Holm correction on at least 2 of the 4 datasets and significantly worse on none.
- Otherwise it is NOT SUPPORTED. With 4 datasets this test has low power, so a "not supported" result does not prove the
  absence of an effect; it means the evidence does not show one.
- "The diversity step helps" is judged by S versus T: it is SUPPORTED only if the mean of S-T is positive and S is
  significantly better than T on at least 1 dataset after Holm correction.

## Deviation policy
Any change after this file was frozen, such as a dataset dropped for a technical failure, will be listed in the paper and in
the repository with its reason. No dataset will be dropped or added because of its results.
