"""
Heuristic screen for label-leaking artefacts in a tabular dataset.

It flags any feature value that (a) repeats in at least `min_frac` of the rows
(and at least `min_rows` rows), (b) is almost pure in one class, and (c) has a
class rate far from the overall class prior. This is exactly the signature of
class-conditional imputation (for example, filling missing Insulin with 102.5
for negatives and 169.5 for positives). Natural discrete features can also be
flagged (ordinal ratings, binary attributes), so flags need a human look. A
clean screen does not prove the absence of leakage.

    python3 data_screen.py          # screens every loader in experiment_final
"""
import numpy as np


def screen(X, y, min_rows=20, min_frac=0.03, purity=0.93, prior_gap=0.25):
    y = np.asarray(y)
    n = len(y)
    flags = []
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = ~np.isnan(col)
        vals, cnt = np.unique(col[ok], return_counts=True)
        if len(vals) > 0.6 * n:        # near-unique continuous feature
            continue
        for v, c in zip(vals, cnt):
            if c < max(min_rows, min_frac * n):
                continue
            for cls in np.unique(y):
                p = float(np.mean(y[ok & (col == v)] == cls))
                if p >= purity and abs(p - float(np.mean(y == cls))) > prior_gap:
                    flags.append({"feature": j, "value": float(v), "rows": int(c), "class": int(cls), "purity": round(p, 3)})
    return flags


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import experiment_final as E
    for name in [n for n in dir(E) if n.startswith("load_")]:
        try:
            X, y = getattr(E, name)()
        except Exception as e:  # loaders for files that are not present
            print(f"{name}: skipped ({type(e).__name__})")
            continue
        print(f"{name:32s} flags: {screen(X, y)[:5]}")
