import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
import os

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

    # Align lengths (and drop NaNs jointly)
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
# Long/stacked CSV loader
# =============================
def build_region_series_from_long(csv_path,region_col,value_col):
    # Reconstruct per-region 1D time series from a long/stacked CSV.
    # Returns: dict {region_name: np.array([...])}

    df = pd.read_csv(csv_path)

    assert region_col in df.columns, f"'{region_col}' not found in {csv_path}"
    assert value_col  in df.columns, f"'{value_col}' not found in {csv_path}"

    # Keep only needed columns; add row order index for stable ordering
    df = df[[region_col, value_col]].copy()
    df["_row_order_"] = np.arange(len(df))

    series_map = {}
    # groupby preserves input order by default; we enforce it explicitly
    for reg, g in df.groupby(region_col, sort=False):
        g = g.sort_values(by="_row_order_", kind="mergesort")
        arr = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
        series_map[str(reg)] = arr
    return series_map

# =============================
# Batch over regions + FDR
# =============================
def main():
 #   female_csv,
 #   male_csv,
 #   TR=2.0,
 #   n_perm=5000,
 #   seed=7,
 #   region_col="region",
 #   value_col="PC score 1",
 #   time_col=None,
 #   out_csv="sex_diff_movie_region_stats.csv"
#):
    TR = 0.98
    # for real make n_perm much larger, e.g. 5000
    n_perm=100
    seed = 7

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

    for curr_mov in movies:

        outpath = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses/sep_PCAs"
        os.makedirs(outpath, exist_ok=True)
        out_csv = f"/{outpath}/results_sex_movie_phasecc_{curr_mov}.csv"

        # CSVs
        path = f"/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/results_PCA/{curr_mov}" 
        csv_f = "PC1_scores_female_allROI.csv"
        csv_m = "PC1_scores_male_allROI.csv" 
        
        female_csv = f"{path}/{csv_f}"
        male_csv = f"{path}/{csv_m}"

        region_col = "Region"
        value_col = "PC_score_1"

        fem = build_region_series_from_long(female_csv, region_col, value_col)
        mal = build_region_series_from_long(male_csv,   region_col, value_col)

        regions = sorted(set(fem.keys()).intersection(mal.keys()))
        rows = []
        length_warnings = []

        for r in regions:
            y_f = fem[r]
            y_m = mal[r]
            if y_f.size != y_m.size:
                length_warnings.append((r, y_f.size, y_m.size))
            res = crosscorr_phase_test(y_f, y_m, TR, n_perm, seed, True)
            rows.append(dict(
                region=r,
                n_samples=int(min(y_f.size, y_m.size)),
                r_max_obs=res["r_max_obs"],
                p_corr_strength=res["p_corr_strength"],
                lag_obs_samples=res["lag_obs_samples"],
                lag_obs_seconds=res["lag_obs_seconds"],
                p_lag=res["p_lag"]
            ))

        out = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)

        # FDR across regions (two families)
        for pcol in ["p_corr_strength", "p_lag"]:
            mask = out[pcol].notna()
            if mask.any():
                rej, q = fdrcorrection(out.loc[mask, pcol].values, alpha=0.05)
                out.loc[mask, pcol.replace("p_", "q_")] = q
                out.loc[mask, pcol.replace("p_", "sig_")] = rej.astype(bool)
            else:
                out[pcol.replace("p_", "q_")] = np.nan
                out[pcol.replace("p_", "sig_")] = False

        out_path = Path(out_csv)
        out.to_csv(out_path, index=False)
        print(f"Saved: {out_path.resolve()}")

        if length_warnings:
            print("⚠️ Length mismatches (female vs male) — truncated to min length for testing:")
            for r, lf, lm in length_warnings[:10]:
                print(f"   {r}: female={lf}, male={lm}")
            if len(length_warnings) > 10:
                print(f"   ...and {len(length_warnings)-10} more regions")

# Execute script
if __name__ == "__main__":
    main()


# -------------------------------
# Example call:
# -------------------------------
# analyze_long_stacked_csvs(
#     female_csv="female_long.csv",
#     male_csv="male_long.csv",
#     TR=2.0,                 # your sampling interval in seconds
#     n_perm=5000,
#     region_col="region",
#     value_col="PC score 1",
#     time_col=None,          # or "time" if you have a time column
#     out_csv="results_sex_movie_phasecc.csv"
# )
