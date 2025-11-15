import pandas as pd

def main(): 

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    nn = 3
    
    results_path = f"{base_path}/results_run_sTOPF"
    exp_path = f"{results_path}/individual_expression_all_nn{nn}.csv" 

    ind_exp = pd.read_csv(exp_path)

    ind_exp['diff_mi'] = ind_exp['fem_mi'] - ind_exp['mal_mi']

    # Create a pivot table: one row per subject, columns = movie × region
    cluster_input = ind_exp.pivot_table(
        index='subject',
        columns=['movie', 'region'],
        values='diff_mi'
    )

    # Flatten column names (movie__region)
    cluster_input.columns = [f"{mv}__{rg}" for mv, rg in cluster_input.columns]

    # Add sex column (unique per subject)
    sex_map = ind_exp[['subject', 'sex']].drop_duplicates().set_index('subject')
    cluster_input['sex'] = sex_map['sex']

    # Save to file
    out_f = f"{results_path}/individual_expression_all_nn{nn}_diff_MI_wide.csv" 
    cluster_input.to_csv(out_f)

    cluster_input.head()


# Execute script
if __name__ == "__main__":
    main()