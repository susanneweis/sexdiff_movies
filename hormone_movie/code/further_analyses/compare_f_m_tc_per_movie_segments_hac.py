#!/usr/bin/env python3
"""
Compare female vs male time courses per region and per movie segment
using a HAC/Newey–West one-sample t-test on the absolute difference |y_f - y_m|.

Significant (q_hac_abs True) => the average separation between female and male
time courses across the segment is > 0 beyond what temporal autocorrelation allows.

Outputs:
  - results_sex_movie_phasecc__movie-<label>.csv   (per movie)
  - results_sex_movie_phasecc__ALL_MOVIES.csv      (combined)

Columns per row (region):
  movie, region, seg_start_idx_1b, seg_end_idx_1b, n_samples,
  p_hac_abs, q_hac_abs, sig_hac_abs,
  hac_abs_mean, hac_abs_t, hac_abs_se, hac_abs_N, hac_abs_L
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

# =============================
# CONFIG (keep your paths/fields the same)
# =============================
female_csv = "female_long.csv"   # path to long-format female time courses
male_csv   = "male_long.csv"     # path to long-format male time courses
TR         = 2.0                 # sampling interval (seconds) - not used by HAC, kept for consistency
region_col = "region"
value_col  = "PC score 1"

# base name for outputs (one-per-movie + a combined file)
out_base   = "results_sex_movie_phasecc"

# Movie segments (1-based inclusive -> will be converted to 0-based slices)
MOVIES = {
    "dd":     (1,   458),
    "s":      (459, 898),
    "dps":    (899, 1372),
    "fg":     (1373,1958),
    "dmw":    (1959,2475),
    "lib":    (2476,2924),
    "tgtbtu": (2925,3431),
}

# =============================
# FDR helper
# =============================
def fdr_bh(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR. Returns (reject_bool, q_values) in original order."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotone
    reject = q <= alpha
    q_full = np.empty_like(q)
    rej_full = np.empty_like(reject)
    q_full[order] = q
    rej_full[order] = reject
    return rej_full, q_full

# =============================
# HAC / Newey–West test on the mean
# =============================
def hac_ttest_mean(d, L=None):
    """
    Newey–West (Bartlett) HAC t-test that E[d_t] = 0.
    Returns dict: t, p, se, N, N_eff (AR1 approx), df, L (bandwidth), mean.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    N = d.size
    if N < 5:
        return dict(t=np.nan, p=np.nan, se=np.nan, N=N, N_eff=np.nan, df=np.nan, L=0, mean=np.nan)

    m = d.mean()
    u = d - m

    # Automatic bandwidth (Andrews-like heuristic)
    if L is None:
        L = int(np.floor(4 * (N/100.0)**(2.0/9.0)))
        L = max(1, min(L, N - 1))

    # HAC long-run variance of the mean with Bartlett kernel
    gamma0 = np.dot(u, u) / N
    S = gamma0
    for k in range(1, L + 1):
        gamma_k = np.dot(u[k:], u[:-k]) / N
        w = 1.0 - k / (L + 1.0)  # Bartlett weight
        S += 2.0 * w * gamma_k

    var_mean = S / N
    se = np.sqrt(var_mean) if var_mean > 0 else np.inf
    t_stat = m / se if se > 0 else np.inf

    # Effective DOF via AR(1) approximation (conservative)
    if N > 2:
        rho1 = np.corrcoef(d[:-1], d[1:])[0, 1]
        N_eff = N * (1 - rho1) / (1 + rho1) if np.isfinite(rho1) and (1 + rho1) != 0 else N
    else:
        N_eff = N
    df = max(2, int(round(N_eff - 1)))

    p = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    return dict(t=float(t_stat), p=float(p), se=float(se),
                N=int(N), N_eff=float(N_eff), df=float(df), L=int(L), mean=float(m))

def hac_abs_diff(yf, ym):
    """HAC test on the absolute difference |yf - ym| (shape/timing divergence regardless of sign)."""
    d_abs = np.abs(np.asarray(yf, float) - np.asarray(ym, float))
    return hac_ttest_mean(d_abs)

# =============================
# Loaders & slicing
# =============================
def build_region_series_from_long(csv_path, region_col="region", value_col="PC score 1"):
    """
    Reconstruct per-region 1D time series from a long/stacked CSV with columns:
      - region: region name (repeats for each timepoint)
      - value_col: numeric value per row (timepoint)
    No explicit time column is required; original row order defines time.
    """
    df = pd.read_csv(csv_path)
    assert region_col in df.columns, f"'{region_col}' not found in {csv_path}"
    assert value_col  in df.columns, f"'{value_col}' not found in {csv_path}"

    df = df[[region_col, value_col]].copy()
    df["_row_order_"] = np.arange(len(df))

    series_map = {}
    for reg, g in df.groupby(region_col, sort=False):
        g = g.sort_values(by="_row_order_", kind="mergesort")
        arr = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
        series_map[str(reg)] = arr
    return series_map

def slice_segment(arr, start_1b, end_1b):
    """Slice 1-based inclusive indices into 0-based half-open slice safely."""
    if arr is None or len(arr) == 0:
        return np.array([], dtype=float)
    n = len(arr)
    start0 = max(0, start_1b - 1)
    end0_excl = min(n, end_1b)
    if start0 >= end0_excl:
        return np.array([], dtype=float)
    return np.asarray(arr[start0:end0_excl], dtype=float)

# =============================
# Main per-movie analysis (HAC only)
# =============================
def analyze_per_movie_segments(
    female_csv, male_csv, movies_dict, TR=2.0,
    region_col="region", value_col="PC score 1", out_base="results_sex_movie_phasecc"
):
    fem = build_region_series_from_long(female_csv, region_col, value_col)
    mal = build_region_series_from_long(male_csv,   region_col, value_col)

    regions = sorted(set(fem.keys()).intersection(mal.keys()))
    print(f"Found {len(regions)} overlapping regions.")

    combined_rows = []

    for mv_label, (start_1b, end_1b) in movies_dict.items():
        rows = []
        for r in regions:
            y_f_full = fem[r]
            y_m_full = mal[r]

            y_f = slice_segment(y_f_full, start_1b, end_1b)
            y_m = slice_segment(y_m_full, start_1b, end_1b)

            # Drop to equal length & remove any NaNs jointly
            L = min(len(y_f), len(y_m))
            y_f = y_f[:L]; y_m = y_m[:L]
            mask = np.isfinite(y_f) & np.isfinite(y_m)
            y_f = y_f[mask]; y_m = y_m[mask]

            if len(y_f) < 5:
                rows.append(dict(
                    movie=mv_label,
                    region=r,
                    seg_start_idx_1b=start_1b,
                    seg_end_idx_1b=end_1b,
                    n_samples=int(len(y_f)),
                    p_hac_abs=np.nan,
                    hac_abs_mean=np.nan,
                    hac_abs_t=np.nan,
                    hac_abs_se=np.nan,
                    hac_abs_N=int(len(y_f)),
                    hac_abs_L=np.nan
                ))
                continue

            hac_abs = hac_abs_diff(y_f, y_m)

            rows.append(dict(
                movie=mv_label,
                region=r,
                seg_start_idx_1b=start_1b,
                seg_end_idx_1b=end_1b,
                n_samples=int(len(y_f)),
                p_hac_abs=hac_abs["p"],
                hac_abs_mean=hac_abs["mean"],   # average |difference| across time (original units)
                hac_abs_t=hac_abs["t"],
                hac_abs_se=hac_abs["se"],
                hac_abs_N=hac_abs["N"],
                hac_abs_L=hac_abs["L"]
            ))

        df_mv = pd.DataFrame(rows).sort_values(["region"]).reset_index(drop=True)

        # ---- FDR per movie (across regions) ----
        if "p_hac_abs" in df_mv.columns:
            mask = df_mv["p_hac_abs"].notna()
            if mask.any():
                rej, q = fdr_bh(df_mv.loc[mask, "p_hac_abs"].values, alpha=0.05)
                df_mv.loc[mask, "q_hac_abs"] = q
                df_mv.loc[mask, "sig_hac_abs"] = rej
            else:
                df_mv["q_hac_abs"] = np.nan
                df_mv["sig_hac_abs"] = False

        out_path = Path(f"{out_base}__movie-{mv_label}.csv")
        df_mv.to_csv(out_path, index=False)
        print(f"Saved: {out_path.resolve()}")

        combined_rows.extend(df_mv.to_dict("records"))

    # Combined master CSV (movie-specific q/flags included)
    df_all = pd.DataFrame(combined_rows).sort_values(["movie", "region"]).reset_index(drop=True)
    out_all = Path(f"{out_base}__ALL_MOVIES.csv")
    df_all.to_csv(out_all, index=False)
    print(f"Saved combined results: {out_all.resolve()}")

# =============================
# RUN
# =============================
if __name__ == "__main__":
    analyze_per_movie_segments(
        female_csv=female_csv,
        male_csv=male_csv,
        movies_dict=MOVIES,
        TR=TR,
        region_col=region_col,
        value_col=value_col,
        out_base=out_base
    )
