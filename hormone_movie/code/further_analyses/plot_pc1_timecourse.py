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
        "--csv",
        default="PC1_scores_female_allROI.csv",
        help="Path to the CSV file (default: PC1_scores_female_allROI.csv)",
    )
    parser.add_argument(
        "--region",
        required=True,
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
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    required_cols = {"Regions", "PCScores1"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV must contain columns {required_cols}, missing: {missing}", file=sys.stderr)
        sys.exit(1)

    # Filter rows for the chosen region
    region = args.region
    sub = df[df["Regions"] == region].copy()

    if sub.empty:
        # Help the user with possible close matches
        unique_regions = sorted(df["Regions"].dropna().unique())
        suggestions = get_close_matches(region, unique_regions, n=5, cutoff=0.6)
        print(f"Error: region '{region}' not found in 'Regions' column.", file=sys.stderr)
        if suggestions:
            print("Did you mean one of:", ", ".join(suggestions), file=sys.stderr)
        else:
            print("Available regions (first 20):", ", ".join(unique_regions[:20]), file=sys.stderr)
        sys.exit(1)

    # If there is a time column, use it; otherwise use sample index
    time_col_candidates = [c for c in df.columns if c.lower() in {"time", "t", "frame", "index"}]
    if time_col_candidates:
        tcol = time_col_candidates[0]
        t = sub[tcol].values
        x_label = tcol
    else:
        # Use the original order as time index
        # Keep the original row order as in the CSV
        sub = sub.reset_index(drop=True)
        t = sub.index.values
        x_label = "Time (sample index)"

    y = sub["PCScores1"].astype(float).values

    # Plot
    plt.figure()
    plt.plot(t, y)
    plt.title(f"{region} — PCScores1 time course")
    plt.xlabel(x_label)
    plt.ylabel("PCScores1")
    plt.grid(True)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {args.save}")

    # Show the plot (comment this out if running headless)
    plt.show()


if __name__ == "__main__":
    main()
