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
        "--path",
        default = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/kristina/PCA", 
        help="Path to the CSV file (default: /data/project/brainvar_sexdiff_movies/hormone_movie/results/kristina/PCA/)",
    )
    parser.add_argument(
        "--csv",
        default="PC1_scores_female_allROI.csv",
        help="CSV file (default: PC1_scores_female_allROI.csv.csv)",
    )
    parser.add_argument(
        "--region",
        default = "17Networks_LH_DorsAttnA_ParOcc_1", 
        # required=True,
        help="Exact region name to plot (must match a value in the 'Regions' column)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot (e.g., plot.png). If omitted, only shows the plot.",
    )
    args = parser.parse_args()

    # Load CSV
    try:
        df = pd.read_csv( f"{args.path}/{args.csv}")
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    required_cols = {"Region", "PC_score_1"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV must contain columns {required_cols}, missing: {missing}", file=sys.stderr)
        sys.exit(1)

    # Filter rows for the chosen region
    region = args.region
    sub = df[df["Region"] == region].copy()

    if sub.empty:
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
    sub = sub.reset_index(drop=True)
    t = sub.index.values
    x_label = "Time (sample index)"

    y = sub["PC_score_1"].astype(float).values

    # Plot
    plt.figure()
    plt.plot(t, y)
    plt.title(f"{region} — PC_score_1 time course")
    plt.xlabel(x_label)
    plt.ylabel("PC_score_1")
    plt.grid(True)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {args.save}")

    # Show the plot (comment this out if running headless)
    plt.show()


if __name__ == "__main__":
    main()
