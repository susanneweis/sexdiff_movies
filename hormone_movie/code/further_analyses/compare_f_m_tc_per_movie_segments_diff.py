#!/usr/bin/env python3
"""
Compare female vs male time courses per region and per movie segment.
Significant = DIFFERENT SHAPES (low similarity relative to a 'same-shape' null).
Also tests timing (lag) differences. FDR is applied per movie.

Outputs:
  - results_sex_movie_phasecc__movie-<label>.csv   (per movie)
  - results_sex_movie_phasecc__ALL_MOVIES.csv      (combined)

Columns per row (region):
  movie, region, seg_start_idx_1b, seg_end_idx_1b, n_samples,
  r_max_obs,
  p_shape_diff, q_shape_diff, sig_shape_diff,
  lag_obs_samples, lag_obs_seconds,
  p_lag, q_lag, sig_lag
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from pathlib import Path

# =============================
# Helpers
# =============================
def fdr_bh(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR. Returns (reject_bool, q_values) in pvals order."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    reject = q <= alpha
    q_full = np.empty_like(q)
    rej_full = np.empty_like(reject)
    q_full[order] = q
    rej_full[order] = reject
    return rej_full, q_full

def _z(x):
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std()
    return (x - m) / s if s > 0 else x*0.0

def phase_randomize(x, rng):
    """Spectrum-preserving, phase-randomized surrogate of x."""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x)
    Xr = X.copy()
    if n % 2 == 0:
        idx = np.arange(1, Xr.size - 1)
        Xr[-1] = Xr[-1].real + 0j  # Nyquist real
    else:
        idx = np.arange(1, Xr.size)
    phi = rng.uniform(0, 2*np.pi, size=idx.size)
    Xr[idx] *= np.exp(1j * phi)
    Xr[0] = Xr[0].real + 0j       # DC real
    return np.fft.irfft(Xr, n=n)

def phase_randomize_synced(x, y, rng):
    """
    'Same-shape' null: apply the SAME random phases to both x and y magnitudes.
    This yields surrogates that share temporal structure on average.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(x.size, y.size)
    X = np.fft.rfft(x[:n]); Y = np.fft.rfft(y[:n])

    if n % 2 == 0:
        idx = np.arange(1, X.size - 1)
        phi = rng.uniform(0, 2*np.pi, size=idx.size)
        phase = np.exp(1j * phi)
        Xn = X.copy(); Yn = Y.copy()
        Xn[idx] = np.abs(X[idx]) * phase
        Yn[idx] = np.abs(Y[idx]) * phase
        Xn[-1] = np.abs(X[-1]) + 0j
        Yn[-1] = np.abs(Y[-1]) + 0j
    else:
        idx = np.arange(1, X.size)
        phi = rng.uniform(0, 2*np.pi, size=idx.size)
        phase = np.exp(1j * phi)
        Xn = X.copy(); Yn = Y.copy()
        Xn[idx] = np.abs(X[idx]) * phase
        Yn[idx] = np.abs(Y[idx]) * phase

    Xn[0] = np.abs(X[0]) + 0j
    Yn[0] = np.abs(Y[0]) + 0j

    xs = np.fft.irfft(Xn, n=n)
    ys = np.fft.irfft(Yn, n=n)
    return xs, ys

def crosscorr_max_and_lag(a, b):
    """Return max normalized cross-correlation and the lag (samples) where it occurs."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    cc = correlate(a - a.mean(), b - b.mean(), mode="full")
    cc /= (len(a) * a.std() * b.std())
    lags = correlation_lags(len(a), len(b), mode="full")
    k = int(np.argmax(cc))
    return float(cc[k]), int(lags[k])

def shape_difference_test(y_f, y_m, dt, n_perm, seed):
    """
    Test shape difference and timing difference between y_f and y_m.
    Returns:
      r_max_obs: observed max cross-corr on z-scored signals
      lag_obs_samples / seconds: lag at r_max_obs (positive => female leads)
      p_shape_diff:  left-tail p under 'same-shape' null (synced-phase surrogates)
                     small p => observed similarity unusually LOW => shapes DIFFER
      p_lag: two-sided p for |lag| under independent phase-randomization null
    """
    rng = np.random.default_rng(seed)
    y_f = np.asarray(y_f, float); y_m = np.asarray(y_m, float)

    # align length and drop joint NaNs
    L = min(y_f.size, y_m.size)
    y_f, y_m = y_f[:L], y_m[:L]
    mask = np.isfinite(y_f) & np.isfinite(y_m)
    y_f, y_m = y_f[mask], y_m[mask]
    if y_f.size < 10:
        return dict(r_max_obs=np.nan, lag_obs_samples=np.nan, lag_obs_seconds=np.nan,
                    p_shape_diff=np.nan, p_lag=np.nan)

    # z-score for shape comparison
    yf = _z(y_f); ym = _z(y_m)

    # observed similarity and lag
    r_max_obs, lag_obs = crosscorr_max_and_lag(yf, ym)
    lag_obs_sec = lag_obs * dt

    # --- shape difference p: 'same-shape' null via synced-phase surrogates ---
    rmax_syn = np.empty(n_perm)
    for i in range(n_perm):
        xs, ys = phase_randomize_synced(yf, ym, rng)  # shared random phases
        xs = _z(xs); ys = _z(ys)
        rmax_syn[i], _ = crosscorr_max_and_lag(xs, ys)
    # left-tail: unusually LOW similarity implies DIFFERENCE
    p_shape_diff = (np.sum(rmax_syn <= r_max_obs) + 1) / (n_perm + 1)

    # --- timing difference p: independent phase-rand -> null of no consistent lag ---
    lag_null = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        xs = _z(phase_randomize(yf, rng))
        ys = _z(phase_randomize(ym, rng))
        _, l = crosscorr_max_and_lag(xs, ys)
        lag_null[i] = l
    p_lag = (np.sum(np.abs(lag_null) >= abs(lag_obs)) + 1) / (n_perm + 1)

    return dict(
        r_max_obs=r_max_obs,
        lag_obs_samples=lag_obs,
        lag_obs_seconds=lag_obs_sec,
        p_shape_diff=p_shape_diff,
        p_lag=p_lag
    )

# =============================
# Loaders & slicing
# =============================
def build_region_series_from_long(csv_path, region_col="region", value_col="PC score 1"):
    """
    Reconstruct per-region 1D time series from a long/stacked CSV
    with columns: region, value_col (no explicit time column).
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
# Main per-movie analysis
# =============================
def main():
    # female_csv
    # male_csv
    # movies_dict
    # TR=2.0
    # n_perm=5000
    # seed=7,
    # region_col="region", 
    # value_col="PC score 1", 
    # out_base="results_sex_movie_phasecc"

    TR = 0.98
    # for real make n_perm much larger, e.g. 5000
    n_perm=100
    seed = 7

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

    #movies_dict=MOVIES

    outpath = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses_diff"
    out_csv = f"{outpath}/results_sex_movie_phasecc_per_movie_diff.csv"

    # CSVs
    path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/kristina/PCA" 
    csv_f = "PC1_scores_female_allROI.csv"
    csv_m = "PC1_scores_male_allROI.csv" 
    
    female_csv = f"{path}/{csv_f}"
    male_csv = f"{path}/{csv_m}"

    region_col = "Region"
    value_col = "PC_score_1"

    # base name for outputs (same root, now one-per-movie + a combined file)
    out_base   = "results_sex_movie_phasecc_diff"
    out_base_path  = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses_diff"

    fem = build_region_series_from_long(female_csv, region_col, value_col)
    mal = build_region_series_from_long(male_csv,   region_col, value_col)

    regions = sorted(set(fem.keys()).intersection(mal.keys()))
    print(f"Found {len(regions)} overlapping regions.")

    combined_rows = []

    for mv_label, (start_1b, end_1b) in MOVIES.items():
        rows = []
        for r in regions:
            y_f_full = fem[r]
            y_m_full = mal[r]

            y_f = slice_segment(y_f_full, start_1b, end_1b)
            y_m = slice_segment(y_m_full, start_1b, end_1b)

            res = shape_difference_test(y_f, y_m, TR, n_perm, seed)
            rows.append(dict(
                movie=mv_label,
                region=r,
                seg_start_idx_1b=start_1b,
                seg_end_idx_1b=end_1b,
                n_samples=int(min(len(y_f), len(y_m))),
                r_max_obs=res["r_max_obs"],
                p_shape_diff=res["p_shape_diff"],
                lag_obs_samples=res["lag_obs_samples"],
                lag_obs_seconds=res["lag_obs_seconds"],
                p_lag=res["p_lag"]
            ))

        df_mv = pd.DataFrame(rows).sort_values(["region"]).reset_index(drop=True)

        # ---- FDR per movie (across regions) ----
        for pcol, qcol, scol in [
            ("p_shape_diff", "q_shape_diff", "sig_shape_diff"),
            ("p_lag",        "q_lag",        "sig_lag")
        ]:
            mask = df_mv[pcol].notna()
            if mask.any():
                rej, q = fdr_bh(df_mv.loc[mask, pcol].values, alpha=0.05)
                df_mv.loc[mask, qcol] = q
                df_mv.loc[mask, scol] = rej
            else:
                df_mv[qcol] = np.nan
                df_mv[scol] = False

        os.makedirs(outpath, exist_ok=True)
        out_path = Path(f"{out_base_path}/{out_base}__movie-{mv_label}.csv")
        df_mv.to_csv(out_path, index=False)
        print(f"Saved: {out_path.resolve()}")

        combined_rows.extend(df_mv.to_dict("records"))

    # Combined master CSV (movie-specific q/flags included)
    df_all = pd.DataFrame(combined_rows).sort_values(["movie", "region"]).reset_index(drop=True)
    os.makedirs(outpath, exist_ok=True)
    out_all = Path(f"{out_base_path}/{out_base}__ALL_MOVIES.csv")
    df_all.to_csv(out_all, index=False)
    print(f"Saved combined results: {out_all.resolve()}")



if __name__ == "__main__":
    main()

# =============================
# RUN
# =============================
# if __name__ == "__main__":
#     analyze_per_movie_segments(
#         female_csv=female_csv,
#         male_csv=male_csv,
#         movies_dict=MOVIES,
#         TR=TR,
#         n_perm=n_perm,
#         seed=seed,
#         region_col=region_col,
#         value_col=value_col,
#         out_base=out_base
#     )
