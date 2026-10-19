"""
Metrics from results/predictions/*.json (produced by run_with_predictions.py).

For each dataset and each arm (A = DACS+Debate with tuned beta, F = fixed
beta 0.3, V = soft voting) this reports:
  * fold-mean accuracy (the same quantity the main study reports)
  * pooled precision, recall, F1 (positive class = disease / diabetes) and ROC AUC
  * pooled confusion matrix
  * McNemar test of A vs V and F vs V on pooled predictions

It also compares fold-mean accuracy with the value saved by the main-study
runner (results/<dataset>.json) so that any drift is visible.

    python3 analyze_predictions.py            # print + write metrics.json
    python3 analyze_predictions.py --update-main pima_diabetes
        # overwrite results/pima_diabetes.json with the corrected main-study numbers
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_final import mcnemar  # noqa: E402

RES = os.path.join(HERE, "..", "results")
PRED = os.path.join(RES, "predictions")
ARMS = {"A": ("A", "pA"), "F": ("F", "pF"), "V": ("V", "pV")}


def analyse(state):
    folds = state["folds"]
    y_all = np.concatenate([f["y"] for f in folds])
    out = {"dataset": state["dataset"], "n": state["n"], "n_folds": len(folds), "arms": {}}
    for arm, (pk, qk) in ARMS.items():
        pred = np.concatenate([f[pk] for f in folds])
        prob = np.concatenate([f[qk] for f in folds])
        fold_acc = [float(np.mean(np.array(f[pk]) == np.array(f["y"]))) for f in folds]
        tn, fp, fn, tp = confusion_matrix(y_all, pred).ravel()
        out["arms"][arm] = {
            "acc_fold_mean": float(np.mean(fold_acc)), "acc_fold_std": float(np.std(fold_acc)),
            "precision": float(precision_score(y_all, pred)), "recall": float(recall_score(y_all, pred)),
            "f1": float(f1_score(y_all, pred)), "roc_auc": float(roc_auc_score(y_all, prob)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    pv = np.concatenate([f["V"] for f in folds])
    for arm in ("A", "F"):
        pa = np.concatenate([f[arm] for f in folds])
        out["arms"][arm]["mcnemar_p_vs_V"] = mcnemar(y_all, pa, pv)
        out["arms"][arm]["delta_acc_pp_vs_V"] = 100 * (
            out["arms"][arm]["acc_fold_mean"] - out["arms"]["V"]["acc_fold_mean"])
    out["spread_mean"] = float(np.mean([f["spreadA"] for f in folds]))
    out["diversity_mean"] = float(np.mean([f["diversityA"] for f in folds]))
    out["n_missing_cells_before_imputation"] = state.get("n_missing_cells_before_imputation")
    return out


def main():
    results = {}
    for fn in sorted(os.listdir(PRED)):
        if fn.endswith(".json") and fn != "metrics.json":
            st = json.load(open(os.path.join(PRED, fn)))
            assert len(st["folds"]) == st["n_folds_total"], f"{fn} incomplete"
            results[st["dataset"]] = analyse(st)
    json.dump(results, open(os.path.join(PRED, "metrics.json"), "w"), indent=2)

    for name, r in results.items():
        print(f"\n=== {name}  (n={r['n']}, folds={r['n_folds']}) ===")
        for arm, label in (("A", "DACS+Debate (tuned beta)"), ("F", "DACS+Debate (beta=0.3)"), ("V", "Soft voting")):
            a = r["arms"][arm]
            extra = f"  dAcc={a['delta_acc_pp_vs_V']:+.2f}pp  McNemar p={a['mcnemar_p_vs_V']:.4f}" if arm != "V" else ""
            c = a["confusion"]
            print(f"{label:26s} acc={a['acc_fold_mean']:.4f}  P={a['precision']:.4f} R={a['recall']:.4f} "
                  f"F1={a['f1']:.4f} AUC={a['roc_auc']:.4f}  [TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}]{extra}")
        ref_path = os.path.join(RES, f"{name}.json")
        if os.path.exists(ref_path):
            ref = json.load(open(ref_path))
            print(f"  saved main-study values: dacs={ref['dacs_mean']:.4f} voting={ref['voting_mean']:.4f} "
                  f"p={ref['mcnemar_p']:.4f}   | recomputed: dacs={r['arms']['A']['acc_fold_mean']:.4f} "
                  f"voting={r['arms']['V']['acc_fold_mean']:.4f} p={r['arms']['A']['mcnemar_p_vs_V']:.4f}")

    if "--update-main" in sys.argv:
        name = sys.argv[sys.argv.index("--update-main") + 1]
        r = results[name]
        ref_path = os.path.join(RES, f"{name}.json")
        old = json.load(open(ref_path))
        os.makedirs(os.path.join(RES, "deprecated"), exist_ok=True)
        json.dump(old, open(os.path.join(RES, "deprecated", f"{name}_LEAKED_before_fix.json"), "w"), indent=2)
        A, V = r["arms"]["A"], r["arms"]["V"]
        new = {"n": r["n"], "dacs_mean": A["acc_fold_mean"], "dacs_std": A["acc_fold_std"],
               "voting_mean": V["acc_fold_mean"], "voting_std": V["acc_fold_std"],
               "spread_mean": r["spread_mean"], "diversity_mean": r["diversity_mean"],
               "mcnemar_p": A["mcnemar_p_vs_V"], "n_folds": r["n_folds"],
               "note": "Recomputed with raw data and in-fold median imputation (earlier file leaked labels)."}
        json.dump(new, open(ref_path, "w"), indent=2)
        print(f"\nUpdated {ref_path}; previous (leaked) version archived in results/deprecated/")


if __name__ == "__main__":
    main()
