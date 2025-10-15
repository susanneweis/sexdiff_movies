#!/usr/bin/env python3
import argparse
import sys
from difflib import get_close_matches

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot PCScores1 time course for a chosen brain region."
    )
    
    parser.add_argument(
        "--region",
        default = "17Networks_LH_VisPeri_ExStrInf_5", 
        # required=True,
        help="Exact region name to plot (must match a value in the 'Regions' column)",
    )

    args = parser.parse_args()

    # Load CSV
    path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/results_PCA/tgtbtu/" 
    csv_f = "PC1_scores_female_allROI.csv"
    csv_m = "PC1_scores_male_allROI.csv" 

    df_f = pd.read_csv( f"{path}/{csv_f}")
    df_m = pd.read_csv( f"{path}/{csv_m}")

    # Validate required columns
    #required_cols = {"Region", "PC_score_1"}
    #missing = required_cols - set(df.columns)
    #if missing:
    #    print(f"Error: CSV must contain columns {required_cols}, missing: {missing}", file=sys.stderr)
    #    sys.exit(1)

    # Filter rows for the chosen region
    region = args.region
    sub_f = df_f[df_f["Region"] == region].copy()
    sub_m = df_m[df_m["Region"] == region].copy()

    if sub_f.empty:
        # Help the user with possible close matches
        unique_regions = sorted(df["Region"].dropna().unique())
        suggestions = get_close_matches(region, unique_regions, n=5, cutoff=0.6)
        print(f"Error: region '{region}' not found in 'Region' column.", file=sys.stderr)
        if suggestions:
            print("Did you mean one of:", ", ".join(suggestions), file=sys.stderr)
        else:
            print("Available regions (first 20):", ", ".join(unique_regions[:20]), file=sys.stderr)
        sys.exit(1)

    # Use the original order as time index
    # Keep the original row order as in the CSV
    sub_f = sub_f.reset_index(drop=True)
    sub_m = sub_m.reset_index(drop=True)

    if len(sub_f) != len(sub_m):
        raise ValueError(f"sub_f and sub_m have different lengths: {len(sub_f)} vs {len(sub_m)}")

    t = sub_f.index.values
    x_label = "Time (sample index)"

    y_f = sub_f["PC_score_1"].astype(float).values
    y_m = sub_m["PC_score_1"].astype(float).values

    #Plot
    plt.figure()
    plt.plot(t, y_f, color='red', label='female')
    plt.plot(t, y_m, color='blue', label='male')
    plt.title(f"{region} — PC_score_1 time course")
    plt.xlabel(x_label)
    plt.ylabel("PC_score_1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    s_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/pca_plots/tests"
    save_to = f"{s_path}/plot_f_m_{args.region}.png"
  
    plt.savefig(save_to, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {save_to}")

    # Show the plot (comment this out if running headless)
    plt.show()


if __name__ == "__main__":
    main()
