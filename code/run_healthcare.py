"""
Run the pre-specified medical evaluation (PREREGISTRATION_HEALTHCARE.md): the same arms, folds and code path as
run_baselines.py, on the four new medical datasets. Resumable.

    python3 run_healthcare.py wdbc 250
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_baselines as RB
from healthcare_data import LOADERS

RB.OUT = os.path.join(HERE, "..", "results", "healthcare")
NAMES = {"wdbc": "wdbc", "hepatitis": "hepatitis", "ljubljana": "breast_cancer_ljubljana", "prostate": "prostate_recurrence"}
for key, loader in LOADERS.items():
    RB.DATASETS[key] = (NAMES[key], loader, None, 2)      # n <= 569 for all four, so no subsampling, 2 repeats

if __name__ == "__main__":
    RB.run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 250.0)
