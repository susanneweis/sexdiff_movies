#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from pathlib import Path
import os

def fdr_bh(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR. Returns (reject_bool, q_values)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n+1))
    # monotone non-increasing from the end
    q = np.minimum.accumulate(q[::-1])[::-1]
    reject = q <= alpha
    # back to original order
    q_full = np.empty_like(q)
    rej_full = np.empty_like(reject)
    q_full[order] = q
    rej_full[order] = reject
    return rej_full, q_full

def main():
    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    data_path = f"{base_path}/data_pipeline"
    results_path = f"{base_path}/results_pipeline"
    outpath = f"{results_path}/compare_loadings/sep_PCAs"
    os.makedirs(outpath, exist_ok=True)

    region_col  = "Region"
    subject_col = "Subject_ID"
    loading_col = "PC_loading_1"

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]
    for curr_mov in movies:
            
        out_csv = f"/{outpath}/results_comp_l_{curr_mov}.csv"

        # CSVs
        path = f"{results_path}/results_PCA/{curr_mov}" 
        csv_f = "PC1_loadings_female_allROI.csv"
        csv_m = "PC1_loadings_male_allROI.csv"    

        fem = pd.read_csv(f"{path}/{csv_f}")
        mal = pd.read_csv(f"{path}/{csv_m}")

        # Clean columns (trim spaces)
        fem.columns = [c.strip() for c in fem.columns]
        mal.columns = [c.strip() for c in mal.columns]

        regions = fem[region_col].drop_duplicates()
        print(f"Found {len(regions)} regions.")

        rows = []
        for r in regions:
            f_vals = fem.loc[fem[region_col] == r, loading_col].dropna().values
            m_vals = mal.loc[mal[region_col] == r, loading_col].dropna().values

            if len(f_vals) >= 2 and len(m_vals) >= 2:
                # Welch's t-test (unequal variances)
                t_stat, p_val = ttest_ind(f_vals, m_vals, equal_var=False)
                row = dict(
                    Region=r,
                    n_female=len(f_vals),
                    n_male=len(m_vals),
                    mean_female=float(np.mean(f_vals)) if len(f_vals) else np.nan,
                    mean_male=float(np.mean(m_vals)) if len(m_vals) else np.nan,
                    t_stat=float(t_stat),
                    p_val=float(p_val),
                    ttest_sig = p_val <= 0.05
                )
            else:
                row = dict(
                    Region=r,
                    n_female=len(f_vals),
                    n_male=len(m_vals),
                    mean_female=float(np.mean(f_vals)) if len(f_vals) else np.nan,
                    mean_male=float(np.mean(m_vals)) if len(m_vals) else np.nan,
                    t_stat=np.nan, p_val=np.nan
                )
            rows.append(row)

        # res = pd.DataFrame(rows).sort_values("Region").reset_index(drop=True)
        res = pd.DataFrame(rows).reset_index(drop=True)

        # FDR across regions (only where p is finite)
        mask = res["p_val"].notna()
        if mask.any():
            rej, q = fdr_bh(res.loc[mask, "p_val"].values, alpha=0.05)
            res.loc[mask, "q_val"] = q
            res.loc[mask, "sig_q"] = rej
        else:
            res["q_val"] = np.nan
            res["sig_q"] = False

        # Save results

        res.to_csv(out_csv, index=False)
        print(f"✅ Results saved to {outpath}")

# Execute script
if __name__ == "__main__":
    main()