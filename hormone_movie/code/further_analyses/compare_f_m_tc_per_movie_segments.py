import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from pathlib import Path

# =============================
# FDR (Benjamini–Hochberg) helper
# =============================
def fdr_bh(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR. Returns (reject_bool, q_values) in the same order as pvals."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    reject = q <= alpha
    # back to original order
    q_full = np.empty_like(q)
    rej_full = np.empty_like(reject)
    q_full[order] = q
    rej_full[order] = reject
    return rej_full, q_full

# =============================
# Phase-randomized CC test
# =============================
def phase_randomize(x, rng):
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x)
    Xr = X.copy()
    if n % 2 == 0:
        idx = np.arange(1, Xr.size - 1)
        Xr[-1] = Xr[-1].real + 0j
    else:
        idx = np.arange(1, Xr.size)
    phi = rng.uniform(0, 2*np.pi, size=idx.size)
    Xr[idx] *= np.exp(1j * phi)
    Xr[0] = Xr[0].real + 0j
    return np.fft.irfft(Xr, n=n)

def crosscorr_phase_test(y_f, y_m, dt, n_perm, seed, zscore):
    rng = np.random.default_rng(seed)
    y_f = np.asarray(y_f, float)
    y_m = np.asarray(y_m, float)

    L = min(y_f.size, y_m.size)
    y_f, y_m = y_f[:L], y_m[:L]
    mask = np.isfinite(y_f) & np.isfinite(y_m)
    y_f, y_m = y_f[mask], y_m[mask]
    if y_f.size < 10:
        return dict(r_max_obs=np.nan, p_corr_strength=np.nan,
                    lag_obs_samples=np.nan, lag_obs_seconds=np.nan, p_lag=np.nan)

    if zscore:
        y_f = (y_f - y_f.mean()) / y_f.std()
        y_m = (y_m - y_m.mean()) / y_m.std()

    a, b = y_f - y_f.mean(), y_m - y_m.mean()
    cc = correlate(a, b, mode="full") / (len(a) * a.std() * b.std())
    lags = correlation_lags(len(a), len(b), mode="full")
    k = np.argmax(cc)
    r_max_obs, lag_obs = float(cc[k]), int(lags[k])
    lag_obs_sec = lag_obs * dt

    rmax_null = np.empty(n_perm)
    lag_null  = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        yf = phase_randomize(y_f, rng)
        ym = phase_randomize(y_m, rng)
        af, bm = yf - yf.mean(), ym - ym.mean()
        cc_n = correlate(af, bm, mode="full") / (len(af) * af.std() * bm.std())
        l_n  = correlation_lags(len(af), len(bm), mode="full")
        j = np.argmax(cc_n)
        rmax_null[i] = cc_n[j]
        lag_null[i]  = int(l_n[j])

    p_corr_strength = (np.sum(rmax_null <= r_max_obs) + 1) / (n_perm + 1)
    p_lag = (np.sum(np.abs(lag_null) >= abs(lag_obs)) + 1) / (n_perm + 1)

    return dict(
        r_max_obs=r_max_obs,
        p_corr_strength=p_corr_strength,
        lag_obs_samples=lag_obs,
        lag_obs_seconds=lag_obs_sec,
        p_lag=p_lag
    )

# =============================
# Loader for long/stacked CSV (no time column)
# =============================
def build_region_series_from_long(csv_path, region_col, value_col):
    # Reconstruct per-region 1D time series from a long/stacked CSV.
    # Returns: dict {region_name: np.array([...])}

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

# =============================
# Segment slicer (1-based inclusive -> 0-based slice)
# =============================
def slice_segment(arr, start_1b, end_1b):
    if arr is None or len(arr) == 0:
        return np.array([], dtype=float)
    n = len(arr)
    start0 = max(0, start_1b - 1)
    end0_excl = min(n, end_1b)  # Python slice end is exclusive
    if start0 >= end0_excl:
        return np.array([], dtype=float)
    return np.asarray(arr[start0:end0_excl], dtype=float)

# =============================
# Analyze per movie, add FDR per movie, save
# =============================
def main():
    # female_csv, 
    # male_csv, 
    # movies_dict, 
    # TR=2.0, 
    # n_perm=5000, 
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

    outpath = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses"
    out_csv = f"{outpath}/results_sex_movie_phasecc_per_movie.csv"

    # CSVs
    path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/kristina/PCA" 
    csv_f = "PC1_scores_female_allROI.csv"
    csv_m = "PC1_scores_male_allROI.csv" 
    
    female_csv = f"{path}/{csv_f}"
    male_csv = f"{path}/{csv_m}"

    region_col = "Region"
    value_col = "PC_score_1"

    # base name for outputs (same root, now one-per-movie + a combined file)
    out_base   = "results_sex_movie_phasecc"
    out_base_path  = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses"

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

            res = crosscorr_phase_test(y_f, y_m, TR, n_perm, seed, True)
            rows.append(dict(
                movie=mv_label,
                region=r,
                seg_start_idx_1b=start_1b,
                seg_end_idx_1b=end_1b,
                n_samples=int(min(len(y_f), len(y_m))),
                r_max_obs=res["r_max_obs"],
                p_corr_strength=res["p_corr_strength"],
                lag_obs_samples=res["lag_obs_samples"],
                lag_obs_seconds=res["lag_obs_seconds"],
                p_lag=res["p_lag"]
            ))

        df_mv = pd.DataFrame(rows).sort_values(["region"]).reset_index(drop=True)

        # ---- FDR per movie (across regions) ----
        for pcol, qcol, scol in [
            ("p_corr_strength", "q_corr_strength", "sig_corr_strength"),
            ("p_lag",           "q_lag",           "sig_lag")
        ]:
            mask = df_mv[pcol].notna()
            if mask.any():
                rej, q = fdr_bh(df_mv.loc[mask, pcol].values, alpha=0.05)
                df_mv.loc[mask, qcol] = q
                df_mv.loc[mask, scol] = rej
            else:
                df_mv[qcol] = np.nan
                df_mv[scol] = False


        out_path = Path(f"{out_base_path}/{out_base}__movie-{mv_label}.csv")
        df_mv.to_csv(out_path, index=False)
        print(f"Saved: {out_path.resolve()}")

        combined_rows.extend(df_mv.to_dict("records"))

    # Also write a combined master CSV (movie-specific q/flags already in rows)
    df_all = pd.DataFrame(combined_rows).sort_values(["movie", "region"]).reset_index(drop=True)
    out_all = Path(f"{out_base_path}/{out_base}__ALL_MOVIES.csv")
    df_all.to_csv(out_all, index=False)
    print(f"Saved combined results: {out_all.resolve()}")

# =============================
# RUN
# =============================
# Execute script
if __name__ == "__main__":
    main()


#if __name__ == "__main__":
#    analyze_per_movie_segments(
#        female_csv=female_csv,
#        male_csv=male_csv,
#        movies_dict=MOVIES,
#        TR=TR,
#        n_perm=n_perm,
#        seed=seed,
#        region_col=region_col,
#        value_col=value_col,
#        out_base=out_base
#    )
