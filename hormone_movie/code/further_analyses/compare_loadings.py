#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# CONFIG
# ==============================
path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/kristina/PCA"
female_file = "PC1_loadings_female_allROI.csv"
male_file   = "PC1_loadings_male_allROI.csv"
outpath = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/compare_time_courses"
output_csv  = "PC1_Loading_SexDiff.csv"
plots_dir   = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/pca_plots"

region_col  = "Region"
subject_col = "Subject_ID"
loading_col = "PC_loading_1"

TOP_N_PLOTS = 12   # how many regions to plot as boxplots

# ==============================
# Helpers
# ==============================
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

def hedges_g(x, y):
    """Hedges' g (small-sample corrected Cohen's d) for two independent samples."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled SD
    sp2 = ((nx-1)*sx + (ny-1)*sy) / (nx + ny - 2)
    if sp2 <= 0:
        return np.sign(mx - my) * 0.0
    d = (mx - my) / np.sqrt(sp2)
    # Hedges correction
    J = 1 - (3 / (4*(nx + ny) - 9))
    return d * J

def safe_title(s):
    return str(s).replace("/", "-").replace("\\", "-")

# ==============================
# Load data
# ==============================
fem = pd.read_csv(f"{path}/{female_file}")
mal = pd.read_csv(f"{path}/{male_file}")

# Clean columns (trim spaces)
fem.columns = [c.strip() for c in fem.columns]
mal.columns = [c.strip() for c in mal.columns]

# Coerce types
for df in (fem, mal):
    df[region_col] = df[region_col].astype(str)
    df[loading_col] = pd.to_numeric(df[loading_col], errors="coerce")

regions = sorted(set(fem[region_col]) & set(mal[region_col]))
print(f"Found {len(regions)} overlapping regions.")

# ==============================
# Stats per region
# ==============================
rows = []
for r in regions:
    f_vals = fem.loc[fem[region_col] == r, loading_col].dropna().values
    m_vals = mal.loc[mal[region_col] == r, loading_col].dropna().values

    if len(f_vals) >= 2 and len(m_vals) >= 2:
        # Welch's t-test (unequal variances)
        t_stat, p_val = ttest_ind(f_vals, m_vals, equal_var=False)
        g = hedges_g(f_vals, m_vals)  # positive => females higher than males
        row = dict(
            Region=r,
            n_female=len(f_vals),
            n_male=len(m_vals),
            mean_female=float(np.mean(f_vals)) if len(f_vals) else np.nan,
            mean_male=float(np.mean(m_vals)) if len(m_vals) else np.nan,
            t_stat=float(t_stat),
            p_val=float(p_val),
            hedges_g=float(g)
        )
    else:
        row = dict(
            Region=r,
            n_female=len(f_vals),
            n_male=len(m_vals),
            mean_female=float(np.mean(f_vals)) if len(f_vals) else np.nan,
            mean_male=float(np.mean(m_vals)) if len(m_vals) else np.nan,
            t_stat=np.nan, p_val=np.nan, hedges_g=np.nan
        )
    rows.append(row)

res = pd.DataFrame(rows).sort_values("Region").reset_index(drop=True)

# FDR across regions (only where p is finite)
mask = res["p_val"].notna()
if mask.any():
    rej, q = fdr_bh(res.loc[mask, "p_val"].values, alpha=0.05)
    res.loc[mask, "q_val"] = q
    res.loc[mask, "sig_fdr"] = rej
else:
    res["q_val"] = np.nan
    res["sig_fdr"] = False

# Save results
out_path = f"{outpath}/{output_csv}"
res.to_csv(out_path, index=False)
print(f"✅ Results saved to {out_path}")

# ==============================
# Plots
# ==============================
plots_dir = Path(plots_dir)
plots_dir.mkdir(parents=True, exist_ok=True)

# 1) Volcano-like plot: effect size vs -log10(p)
volcano_path = plots_dir / "volcano_effectsize_vs_p.png"
plt.figure(figsize=(8,6))
p_for_plot = res["p_val"].replace(0, np.nextafter(0,1))
neglogp = -np.log10(p_for_plot)
colors = np.where(res.get("sig_fdr", False), "tab:red", "tab:gray")
plt.scatter(res["hedges_g"], neglogp, s=28, alpha=0.9, c=colors)
plt.axhline(-np.log10(0.05), linestyle="--")
plt.axvline(0, linestyle=":")
plt.xlabel("Hedges' g ( + : female > male )")
plt.ylabel("-log10(p)")
plt.title("PC1 loadings: effect size vs significance (per region)")
plt.tight_layout()
plt.savefig(volcano_path, dpi=150)
plt.close()
print(f"Saved volcano plot: {volcano_path.resolve()}")

# 2) p-value histogram
hist_path = plots_dir / "pvalue_hist.png"
plt.figure(figsize=(7,5))
valid_p = res["p_val"].dropna().values
plt.hist(valid_p, bins=20, edgecolor="black")
plt.xlabel("p-value")
plt.ylabel("Count of regions")
plt.title("Distribution of p-values across regions")
plt.tight_layout()
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Saved p-value histogram: {hist_path.resolve()}")

# 3) Boxplots for top N regions by q (or p if q missing)
def pick_top_regions(df, top_n=TOP_N_PLOTS):
    if df["q_val"].notna().any():
        order = df.sort_values(["sig_fdr","q_val","p_val"], ascending=[False,True,True])
    else:
        order = df.sort_values("p_val", ascending=True)
    return order["Region"].head(top_n).tolist()

top_regions = pick_top_regions(res, TOP_N_PLOTS)

def boxplot_region(region_name):
    f_vals = fem.loc[fem[region_col]==region_name, loading_col].dropna().values
    m_vals = mal.loc[mal[region_col]==region_name, loading_col].dropna().values
    title = (f"{region_name}  |  "
             f"meanF={np.mean(f_vals):.3f} (n={len(f_vals)}), "
             f"meanM={np.mean(m_vals):.3f} (n={len(m_vals)})\n"
             f"t={res.loc[res.Region==region_name,'t_stat'].values[0]:.3f}, "
             f"p={res.loc[res.Region==region_name,'p_val'].values[0]:.3g}, "
             f"q={res.loc[res.Region==region_name,'q_val'].values[0]:.3g} "
             f"g={res.loc[res.Region==region_name,'hedges_g'].values[0]:.3f}")

    plt.figure(figsize=(6.5,5))
    plt.boxplot([f_vals, m_vals], labels=["Female","Male"], showfliers=True)
    plt.ylabel("PC Loading 1")
    plt.title(title)
    plt.tight_layout()
    fname = plots_dir / f"box_{safe_title(region_name)}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

saved = []
for r in top_regions:
    try:
        saved.append(boxplot_region(r))
    except Exception as e:
        print(f"Plot failed for {r}: {e}")

print(f"Saved {len(saved)} region boxplots to {plots_dir.resolve()}")

# Summary to console
n_sig = int(res.get("sig_fdr", pd.Series([], dtype=bool)).sum())
print(f"\nSummary: {n_sig} regions significant after FDR (alpha=0.05).")
print("Top regions plotted (by q or p):")
for p in saved:
    print(" -", p.name)
