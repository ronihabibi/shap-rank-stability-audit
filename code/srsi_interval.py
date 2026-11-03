#!/usr/bin/env python3
"""srsi_interval.py - recompute SRSI, bands, Kendall's W and percentile intervals
from a bootstrap rank matrix R (B resamples x k domains).

Input : CSV, one row per bootstrap resample, one column per domain,
        cell = rank of the domain in that resample (1 = largest domain score;
        each row must be a permutation of 1..k).
Output: CSV table with one row per domain (printed and written to --out).

Definitions (as in the manuscript)
  SD_max  = sqrt((k^2 - 1) / 12)            SD of a rank drawn uniformly from 1..k
  SRSI'_i = 1 - SD_i / SD_max               unclipped value
  SRSI_i  = max(0, SRSI'_i)
  Bands   : Stable >= 0.85; Caution 0.60 to <0.85; Unreliable < 0.60 (unrounded values)
  W       = 1 - (1/k) * sum_i (1 - SRSI'_i)^2   (Kendall's W; equals the classical W
            when SD uses divisor B, i.e. --ddof 0)
Interval: 2.5th and 97.5th percentiles of SRSI over --n-boot row resamplings of R
          with replacement (default 2000, seed 0).
SD divisor: --ddof 1 (sample SD, used for real data) or 0 (population SD, used in
            the simulations).
Usage: python srsi_interval.py ranks.csv --out srsi_table.csv [--ddof 1] [--n-boot 2000] [--seed 0]
       python srsi_interval.py --selftest
"""
import argparse, sys
import numpy as np
import pandas as pd


def srsi_prime(R, ddof=1):
    k = R.shape[1]
    sd_max = np.sqrt((k**2 - 1) / 12.0)
    return 1.0 - R.std(axis=0, ddof=ddof) / sd_max


def band(x):
    return np.where(x >= 0.85, "Stable", np.where(x >= 0.60, "Caution", "Unreliable"))


def kendall_w_from_srsi(sp):
    return 1.0 - np.mean((1.0 - sp) ** 2)


def kendall_w_classical(R):
    """Classical Kendall's W for m = B rankings of k items without ties."""
    m, k = R.shape
    S = ((R.sum(axis=0) - m * (k + 1) / 2.0) ** 2).sum()
    return 12.0 * S / (m**2 * (k**3 - k))


def check_permutations(R):
    k = R.shape[1]
    target = np.arange(1, k + 1)
    bad = [i for i, row in enumerate(np.sort(R, axis=1)) if not np.array_equal(row, target)]
    if bad:
        raise ValueError(f"{len(bad)} rows are not permutations of 1..{k} (first: {bad[:5]})")


def analyse(R, ddof=1, n_boot=2000, seed=0):
    R = np.asarray(R, dtype=float)
    check_permutations(R)
    B, k = R.shape
    sp = srsi_prime(R, ddof)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, k))
    for b in range(n_boot):
        boots[b] = np.maximum(0.0, srsi_prime(R[rng.integers(0, B, B)], ddof))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    rlo, rhi = np.percentile(R, [2.5, 97.5], axis=0)
    srsi = np.maximum(0.0, sp)
    return {
        "mean_rank": R.mean(axis=0), "rank_sd": R.std(axis=0, ddof=ddof),
        "srsi_unclipped": sp, "srsi": srsi, "band": band(sp),
        "srsi_ci_low": lo, "srsi_ci_high": hi,
        "rank_p2.5": rlo, "rank_p97.5": rhi,
        "pct_rank1": (R == 1).mean(axis=0) * 100,
    }, kendall_w_from_srsi(sp)


def selftest():
    rng = np.random.default_rng(1)
    for k in (4, 7, 14):
        base = rng.normal(size=k) * np.linspace(2, 0.2, k)
        R = np.array([np.argsort(np.argsort(-(base + rng.normal(size=k)))) + 1 for _ in range(200)])
        sp = srsi_prime(R, ddof=0)
        assert abs(kendall_w_from_srsi(sp) - kendall_w_classical(R)) < 1e-12, "W identity failed"
        res, w = analyse(R, ddof=1, n_boot=200)
        assert (res["srsi"] >= 0).all() and (res["srsi"] <= 1).all()
    print("selftest ok: Eq.(7) matches classical Kendall's W for k = 4, 7, 14 (ddof 0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ranks", nargs="?")
    ap.add_argument("--out", default="srsi_table.csv")
    ap.add_argument("--ddof", type=int, default=1, choices=(0, 1))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.ranks:
        ap.error("ranks.csv is required (or use --selftest)")
    df = pd.read_csv(a.ranks)
    res, w = analyse(df.values, a.ddof, a.n_boot, a.seed)
    out = pd.DataFrame(res, index=df.columns).round(4)
    out.index.name = "domain"
    out.to_csv(a.out)
    print(out.to_string())
    print(f"\nB = {len(df)}, k = {df.shape[1]}, Kendall's W (Eq. 7) = {w:.3f}")


if __name__ == "__main__":
    sys.exit(main())
