import sys, json, time
sys.path.insert(0, ".")
from experiment_final import (
    run_dataset, load_pima, load_heart, load_australian, load_haberman,
    load_liver, load_magic, load_spambase, load_banknote,
    load_breast_cancer_wisconsin, load_german_credit, load_ionosphere, load_sonar,
)

LOADERS = {
    "pima": ("Pima Diabetes", load_pima, None),
    "heart": ("Heart Disease", load_heart, None),
    "australian": ("Australian Credit", load_australian, None),
    "haberman": ("Haberman Survival", load_haberman, None),
    "liver": ("Liver Disorders", load_liver, None),
    "magic": ("Magic Gamma", load_magic, 3000),
    "spambase": ("Spambase", load_spambase, 2000),
    "banknote": ("Banknote Auth", load_banknote, None),
    "breast_cancer": ("Breast Cancer Wisconsin", load_breast_cancer_wisconsin, None),
    "german": ("German Credit", load_german_credit, None),
    "ionosphere": ("Ionosphere", load_ionosphere, None),
    "sonar": ("Sonar", load_sonar, None),
}

key = sys.argv[1]
name, loader, cap = LOADERS[key]

# Large/slow datasets: cut repeats to 1 (5 folds) to fit the execution
# time budget. Disclosed explicitly in the results.
if key in ("magic", "spambase") and len(sys.argv) > 2 and sys.argv[2] == "fast":
    import experiment_final
    experiment_final.N_REPEATS = 1

t0 = time.time()
X, y = loader()
run_dataset(name, X, y, max_n_for_speed=cap, verbose_folds=True)
print(f"\n[{key}] TOTAL ELAPSED: {time.time()-t0:.1f}s")
