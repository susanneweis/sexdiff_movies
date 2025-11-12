import pandas as pd

def main(base_path, nn_mi):
    results_path = f"{base_path}/results_rjn_sTOPF"

    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)
    subs = ind_ex_data["subject"].unique().tolist()

    quant = 20
    quantile = quant*0.01

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "rest_run-1", "rest_run-2"]

    results = []

    for curr_mov in movies:

        cmp_tc_path = f"{results_path}/compare_time_courses/results_compare_time_courses_{curr_mov}.csv" 
        cmp_tc_data = pd.read_csv(cmp_tc_path)

        thresh = cmp_tc_data["mutual_inf"].quantile(quantile)
        diff_regs = cmp_tc_data.loc[cmp_tc_data["mutual_inf"] < thresh, "region"].tolist()

        for curr_sub in subs:

            curr_sub_data = ind_ex_data[ind_ex_data["subject"] == curr_sub]
            sub_sex = curr_sub_data["sex"].unique()[0]

            curr_sub_movie_data = curr_sub_data[curr_sub_data["movie"] == curr_mov]

            sub_mov_reg_data = curr_sub_movie_data[curr_sub_movie_data["region"].isin(diff_regs)]

            class_fem = sub_mov_reg_data["fem_mi"] > sub_mov_reg_data["mal_mi"]
            # Count how often this is True
            count = class_fem.sum()

            # proportion (percentage)
            prop = class_fem.mean()

            class_result = "female" if prop > 0.5 else "male"
            class_corr = 1 if sub_sex == class_result else 0

            results.append({"subject": curr_sub, "sex": sub_sex, "movie": curr_mov, "percent fem": prop, "classification": class_result, "classification correct": class_corr})


    out_df = pd.DataFrame(results, columns=["subject", "sex", "movie", "percent fem", "classification", "classification correct"])
    out_csv = f"{results_path}/classification_subjects_movies_nn{nn_mi}_top_{quant}perc.csv"
    out_df.to_csv(out_csv, index=False)

    in_df = f"{results_path}/classification_subjects_movies_nn{nn_mi}_top_{quant}perc.csv"
    sub_mov_class = pd.read_csv(in_df)

    subs = sub_mov_class["subject"].unique().tolist()

    overall_res = []

    act_mv = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

    for curr_sub in subs: 
        curr_sub_all_class = sub_mov_class[sub_mov_class["subject"] == curr_sub]
        sub_sex = curr_sub_all_class["sex"].unique()[0]

        curr_sub_all_class = curr_sub_all_class[curr_sub_all_class["movie"].isin(act_mv)]

        perc_female = (curr_sub_all_class["classification"] == "female").mean() * 100
        overall_class = "female" if perc_female > 50 else "male"

        overall_class_corr = 1 if sub_sex == overall_class else 0

        overall_res.append({"subject": curr_sub, "sex": sub_sex, "percent fem": perc_female, "overall classification": overall_class, "overall classification correct": overall_class_corr})

    out_df = pd.DataFrame(overall_res, columns=["subject", "sex", "percent fem", "overall classification", "overall classification correct"])
    out_csv = f"{results_path}/classification_subjects_across_movies_nn{nn_mi}_top_{quant}perc.csv"
    out_df.to_csv(out_csv, index=False)












# Execute script
if __name__ == "__main__":
    main()