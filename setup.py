from setuptools import setup, find_packages

setup(
    name="dacs-debate-ensemble",
    version="1.0.0",
    description=(
        "Competitive-margin DACS classifier selection with iterative "
        "reliability-weighted belief revision (Consensus/Debate Ensemble)."
    ),
    package_dir={"": "code"},
    py_modules=[
        "candidate_pool",
        "debate_ensemble",
        "baselines",
        "data_utils",
        "experiment_final",
    ],
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "scipy>=1.10",
    ],
    python_requires=">=3.9",
)
