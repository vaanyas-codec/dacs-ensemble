"""
Analysis of the confirmatory run, exactly as specified in PREREGISTRATION.md.

Per dataset: mean fold accuracy for Arm A (margin 0.05), Arm B (margin
disabled) and the soft-voting baseline; McNemar on pooled out-of-fold
predictions for A vs baseline and A vs B; Holm correction across datasets
per comparison. Aggregate: positive-difference count, mean difference, exact
two-sided sign test. Applies the pre-specified support rule.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_final import mcnemar, holm_bonferroni  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "confirmatory")
ORDER = ["iris", "wine", "new-thyroid", "wheat-seeds", "oil-spill", "phoneme", "mammography"]


def summarize():
    rows = {}
    for key in ORDER:
        s = json.load(open(os.path.join(OUT, f"{key}.json")))
        assert len(s["folds"]) == s["n_folds_total"], f"{key} incomplete"
        y = np.concatenate([f["y"] for f in s["folds"]])
        A, B, V = (np.concatenate([f[k] for f in s["folds"]]) for k in "ABV")
        acc = lambda k: float(np.mean([np.mean(np.array(f[k]) == np.array(f["y"])) for f in s["folds"]]))
        rows[key] = {
            "n": s["n"], "classes": s["n_classes"], "folds": len(s["folds"]),
            "accA": acc("A"), "accB": acc("B"), "accV": acc("V"),
            "dAV": 100 * (acc("A") - acc("V")), "dAB": 100 * (acc("A") - acc("B")),
            "p_AV": mcnemar(y, A, V), "p_AB": mcnemar(y, A, B),
        }
    hAV = holm_bonferroni({k: r["p_AV"] for k, r in rows.items()})
    hAB = holm_bonferroni({k: r["p_AB"] for k, r in rows.items()})
    for k in rows:
        rows[k]["holm_AV"], rows[k]["holm_AB"] = hAV[k]["holm_p"], hAB[k]["holm_p"]
    return rows


def aggregate(rows, dkey, holmkey, accA_better_is_win=True):
    d = np.array([r[dkey] for r in rows.values()])
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    sign_p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    sig_loss = [k for k, r in rows.items() if r[dkey] < 0 and r[holmkey] < 0.05]
    sig_win = [k for k, r in rows.items() if r[dkey] > 0 and r[holmkey] < 0.05]
    supported = d.mean() > 0 and pos >= 5 and not sig_loss
    return {"mean_delta_pp": float(d.mean()), "positive": pos, "negative": neg,
            "ties": int((d == 0).sum()), "sign_test_p": float(sign_p),
            "significant_wins_after_holm": sig_win, "significant_losses_after_holm": sig_loss,
            "pre_specified_rule_supported": bool(supported)}


if __name__ == "__main__":
    rows = summarize()
    h1 = aggregate(rows, "dAV", "holm_AV")
    h2 = aggregate(rows, "dAB", "holm_AB")
    json.dump({"datasets": rows, "H1_A_vs_baseline": h1, "H2_A_vs_original_rule": h2},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print(f"{'dataset':12s} {'n':>5s} {'C':>2s} | {'voting':>7s} {'ruleB':>7s} {'armA':>7s} | "
          f"{'A-V pp':>7s} {'p':>7s} {'holm':>7s} | {'A-B pp':>7s} {'p':>7s} {'holm':>7s}")
    for k, r in rows.items():
        print(f"{k:12s} {r['n']:5d} {r['classes']:2d} | {r['accV']:7.4f} {r['accB']:7.4f} {r['accA']:7.4f} | "
              f"{r['dAV']:+7.2f} {r['p_AV']:7.4f} {r['holm_AV']:7.4f} | {r['dAB']:+7.2f} {r['p_AB']:7.4f} {r['holm_AB']:7.4f}")
    for name, h in (("H1  Arm A vs soft voting", h1), ("H2  Arm A vs original rule", h2)):
        print(f"\n{name}: mean {h['mean_delta_pp']:+.2f} pp | positive {h['positive']}, negative {h['negative']}, "
              f"ties {h['ties']} | sign-test p={h['sign_test_p']:.3f}")
        print(f"   significant wins (Holm): {h['significant_wins_after_holm'] or 'none'}; "
              f"significant losses (Holm): {h['significant_losses_after_holm'] or 'none'}")
        print(f"   pre-specified support rule met: {h['pre_specified_rule_supported']}")
