# Data sources and provenance

| File | Origin | Status |
|---|---|---|
| `pima_diabetes.csv` | Raw Pima Indians Diabetes data (768 rows, 268 positive). Checked against two independent public copies on GitHub (jbrownlee/Datasets `pima-indians-diabetes.data.csv` and npradaschnor/Pima-Indians-Diabetes-Dataset `diabetes.csv`); all labels and all non-missing cells match. Zero-coded missing values are converted to NaN by the loader and imputed inside each training fold. | verified |
| `deprecated/pima_diabetes_CLASS_IMPUTED_DO_NOT_USE.csv` | The earlier file: identical to the raw data except that zero-coded missing values had been replaced by class-conditional medians. Leaks the label. | quarantined |
| `heart_disease.csv` | Cleveland heart disease data (UCI); 6 rows with missing values removed (297 rows used). | original source to be recorded by the authors |
| `australian_real.csv`, `banknote.csv`, `breast_cancer_wisconsin.csv`, `german_credit.csv`, `haberman_real.csv`, `ionosphere.csv`, `liver_real.csv`, `magic_real.csv`, `sonar.csv`, `spambase_real.csv` | UCI Machine Learning Repository datasets (names as in the UCI catalogue). | exact URLs and licences to be recorded by the authors |
| `confirmatory/*.csv` (iris, wine, new-thyroid, wheat-seeds, oil-spill, phoneme, mammography) | Public benchmark datasets used in the frozen confirmatory protocol. | exact URLs and licences to be recorded by the authors |

I did not verify the origin or licence of the files marked "to be recorded". Check each dataset's page before
publishing the repository, since licence terms differ between datasets.

Leak screen (`code/data_screen.py`): all 12 main and 7 confirmatory files were screened. Only natural discrete features
were flagged (ordinal ratings, binary attributes, minimum values such as iris petal width 0.2 = setosa). No imputation
artefact was found other than the Pima file described above. A heuristic screen cannot prove the absence of leakage.

## data/healthcare/ (pre-specified medical evaluation)

| File | Origin | Notes |
|---|---|---|
| `wdbc.csv` | scikit-learn `load_breast_cancer` (Wisconsin Diagnostic Breast Cancer, UCI) | 569 x 30; target coded as in scikit-learn (1 = benign) |
| `hepatitis.csv` | UCI Hepatitis, copy from the pycaret/pycaret GitHub repository | 154 rows; 1 = died; 153 missing cells; copy differs from the UCI original by one row |
| `breast_cancer_ljubljana.csv` | UCI Breast Cancer (Ljubljana), copy from jbrownlee/Datasets | 286 rows, categorical |
| `prostate_blood_storage.csv` | `medicaldata::blood_storage` via vincentarelbundock/Rdatasets | 316 rows; follow-up and post-surgery columns dropped by the loader |

The licence of each source repository and of each original dataset was not checked here; confirm before publishing.
Upstream preprocessing (including any imputation done before these copies were made) cannot be verified.
