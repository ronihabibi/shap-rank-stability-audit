"""Reference implementation of the calibrated bootstrap rank audit (Paper A, Section 2.6).

audit_domains() retrains the whole pipeline on B bootstrap resamples, ranks SHAP domains,
and returns the rank distribution, modal rank frequency, SRSI and the Stable flag.
flag_probability() converts a flag into a probability with a likelihood ratio and a prior.
"""
import numpy as np, pandas as pd, shap
from sklearn.ensemble import RandomForestClassifier

def _ranks(M, y, idx, seed, rows):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1).fit(M[rows], y[rows])
    sv = shap.TreeExplainer(rf).shap_values(M[rows])
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    imp = np.abs(sv).mean(0)
    score = np.array([imp[c].sum() for c in idx])          # sum aggregation per domain
    order = np.argsort(-score, kind="stable")
    r = np.empty(len(idx), int); r[order] = np.arange(1, len(idx) + 1)
    return r

def audit_domains(M, y, idx, names, B=40, seed=0, modal_cut=0.8, srsi_cut=0.85):
    """M: feature matrix (n x p); y: binary target; idx: list of column-index arrays, one per domain."""
    n, k = len(y), len(idx)
    rng = np.random.default_rng(seed)
    R = np.array([_ranks(M, y, idx, seed + b, rng.integers(0, n, n)) for b in range(B)])  # B x k
    sd = R.std(axis=0, ddof=1)
    sd_max = np.sqrt((k ** 2 - 1) / 12)
    srsi = np.maximum(0, 1 - sd / sd_max)
    modal = np.array([np.bincount(R[:, j]).max() / B for j in range(k)])
    lo, hi = np.percentile(R, 2.5, axis=0), np.percentile(R, 97.5, axis=0)
    out = pd.DataFrame({"domain": names, "median_rank": np.median(R, axis=0), "sd": sd, "srsi": srsi,
                        "modal_freq": modal, "rank_interval": hi - lo})
    out["flag_stable"] = out.modal_freq >= modal_cut      # protocol rule (SRSI >= srsi_cut is the band rule)
    out["band"] = np.where(out.srsi >= srsi_cut, "Stable", np.where(out.srsi >= 0.60, "Caution", "Unreliable"))
    return out.sort_values("median_rank").reset_index(drop=True), R

def flag_probability(lr_plus, prior):
    """Posterior probability that a flagged domain is reproducible: odds = prior odds x LR+."""
    odds = lr_plus * prior / (1 - prior)
    return odds / (1 + odds)
