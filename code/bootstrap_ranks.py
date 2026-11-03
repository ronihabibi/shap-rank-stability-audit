#!/usr/bin/env python3
"""bootstrap_ranks.py - TEMPLATE for producing the bootstrap rank matrix R.

Input : an anonymized table (CSV) with a binary outcome column and predictor columns,
        plus a JSON file mapping each domain name to its list of (encoded) columns.
Method: for b = 0..B-1 draw n rows with replacement, fit a Random Forest, compute
        TreeExplainer SHAP values on the resampled (training) rows, take mean |SHAP|
        per column, SUM over the columns of each domain, rank domains (1 = largest;
        ties go to the domain earlier in column order).
Output: ranks.csv (B x k) for srsi_interval.py.

NOTE: the Random Forest hyperparameters below are placeholders. Set them to the values
used in the study before reporting any numbers; the seeds follow the manuscript
(resample indices: seed 42; Random Forest: 42 + b).
Usage: python bootstrap_ranks.py data.csv domains.json --outcome status --out ranks.csv --B 200
"""
import argparse, json
import numpy as np, pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = dict(n_estimators=300)  # <- replace with the study's settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("domains")
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ranks.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    dom = json.load(open(a.domains))
    y = df[a.outcome].values
    X = df.drop(columns=[a.outcome])
    names = list(dom)
    cols = [X.columns.get_indexer(dom[d]) for d in names]
    n, rng, rows = len(df), np.random.default_rng(a.seed), []
    for b in range(a.B):
        idx = rng.integers(0, n, n)
        rf = RandomForestClassifier(random_state=a.seed + b, **RF_PARAMS).fit(X.iloc[idx], y[idx])
        sv = shap.TreeExplainer(rf).shap_values(X.iloc[idx])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imp = np.abs(sv).mean(axis=0)
        score = np.array([imp[c].sum() for c in cols])
        order = np.argsort(-score, kind="stable")      # stable: ties -> earlier domain
        r = np.empty(len(names), int); r[order] = np.arange(1, len(names) + 1)
        rows.append(r)
    pd.DataFrame(rows, columns=names).to_csv(a.out, index=False)
    print("wrote", a.out, np.array(rows).shape)


if __name__ == "__main__":
    main()
