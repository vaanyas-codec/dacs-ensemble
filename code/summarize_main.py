"""
Rebuild results/summary_table.csv for the 12 main datasets from results/<dataset>.json,
adding Holm-Bonferroni-corrected p-values (family = the 12 datasets), and report the
no-revision ablation across the same datasets.

    python3 summarize_main.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_final import holm_bonferroni  # noqa: E402

RES = os.path.join(HERE, "..", "results")
MAIN = ["pima_diabetes", "heart_disease", "australian_credit", "haberman_survival", "liver_disorders",
        "magic_gamma", "spambase", "banknote_auth", "breast_cancer_wisconsin", "german_credit",
        "ionosphere", "sonar"]

rows, pvals = {}, {}
for k in MAIN:
    r = json.load(open(os.path.join(RES, f"{k}.json")))
    rows[k] = r
    pvals[k] = r["mcnemar_p"]
holm = holm_bonferroni(pvals)

out = []
for k in MAIN:
    r, h = rows[k], holm[k]
    d = 100 * (r["dacs_mean"] - r["voting_mean"])
    verdict = "win" if (h["sig"] and d > 0) else ("loss" if (h["sig"] and d < 0) else "not significant")
    out.append({"dataset": k, "n": r["n"], "n_folds": r["n_folds"], "diversity_mean": round(r["diversity_mean"], 4),
                "voting_acc": round(r["voting_mean"], 4), "dacs_debate_acc": round(r["dacs_mean"], 4),
                "delta_pp": round(d, 2), "mcnemar_p_raw": round(r["mcnemar_p"], 4),
                "holm_p": round(h["holm_p"], 4), "verdict_after_holm": verdict})

with open(os.path.join(RES, "summary_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

wins = [o["dataset"] for o in out if o["verdict_after_holm"] == "win"]
losses = [o["dataset"] for o in out if o["verdict_after_holm"] == "loss"]
nominal_up = sum(o["delta_pp"] > 0 for o in out)
print(f"{'dataset':26s} {'voting':>7s} {'DACS+D':>7s} {'delta':>7s} {'p_raw':>8s} {'p_holm':>8s}  verdict")
for o in out:
    print(f"{o['dataset']:26s} {o['voting_acc']:7.4f} {o['dacs_debate_acc']:7.4f} {o['delta_pp']:+7.2f} "
          f"{o['mcnemar_p_raw']:8.4f} {o['holm_p']:8.4f}  {o['verdict_after_holm']}")
print(f"\nSignificant wins after Holm: {len(wins)} {wins}")
print(f"Significant losses after Holm: {len(losses)} {losses}")
print(f"Nominally ahead of voting on {nominal_up}/12; mean delta {np.mean([o['delta_pp'] for o in out]):+.2f} pp "
      f"(median {np.median([o['delta_pp'] for o in out]):+.2f} pp)")

ab = json.load(open(os.path.join(RES, "exploratory_no_debate_main.json")))["datasets"]
diffs = [100 * (ab[k]["accC"] - ab[k]["accA"]) for k in MAIN]
print(f"\nAblation (no revision rounds minus full method), pp: mean {np.mean(diffs):+.3f}, "
      f"max |diff| {np.max(np.abs(diffs)):.3f} over {len(diffs)} datasets")
