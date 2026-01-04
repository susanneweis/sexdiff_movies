import os
import pandas as pd


def main(base_path):
    # -------------------------
    # Configuration
    # -------------------------

    #nn_values = [3, 5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90, 100]
    nn_values = [3, 5, 10, 15, 20, 25, 30, 35, 50]
    
    perc_values = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]

    value_column = "overall classification correct"

    # -------------------------
    # Collect results
    # -------------------------
    results = []

    for nn in nn_values:
        row = {"nn": nn}

        for perc in perc_values:
            file_path = os.path.join(
                base_path,
                "results_run_sTOPF_v2", 
                f"results_nn{nn}",
                "ind_classification",
                f"classification_subjects_across_movies_nn{nn}_top_{perc}perc.csv"
            )

            if not os.path.exists(file_path):
                row[perc] = None
                print(f"⚠️ Missing: {file_path}")
                continue

            df = pd.read_csv(file_path)

            # Ensure numeric
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

            row[perc] = df[value_column].mean()

        results.append(row)

    # -------------------------
    # Create final DataFrame
    # -------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("nn").sort_index()

    # -------------------------
    # Save output
    # -------------------------
    out_file = f"{base_path}/results_run_sTOPF_v2/mean_classification_across_nn_and_perc.csv"

    results_df.to_csv(out_file)

    # -------------------------
    # Collect results
    # -------------------------
    results_corr = []

    for perc in perc_values:
        file_path = os.path.join(
            base_path,
            "results_run_sTOPF_v2", 
            f"results_nn15",
            "ind_classification",
            f"classification_subjects_across_movies_corr_top_{perc}perc.csv"
        )

        if not os.path.exists(file_path):
            row[perc] = None
            print(f"⚠️ Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Ensure numeric
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

        results_corr.append(df[value_column].mean())

    # -------------------------
    # Create final DataFrame
    # -------------------------
    results_corr_df = pd.DataFrame(results_corr)

    # -------------------------
    # Save output
    # -------------------------
    out_file_corr = f"{base_path}/results_run_sTOPF_v2/mean_classification_corr_across_perc.csv"

    results_corr_df.to_csv(out_file_corr)


# Execute script
if __name__ == "__main__":
    main()
