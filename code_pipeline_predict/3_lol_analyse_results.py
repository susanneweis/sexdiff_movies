import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm

def main():

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    results_path = f"{base_path}/results_pipeline_predict"
    
    # Change this later 
    ind_expr_path = f"{results_path}/individual_expression_all.csv"
    ind_expr = pd.read_csv(ind_expr_path)

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "rest_run-1", "rest_run-2"]
    subjects = ind_expr["subject"].astype(str).drop_duplicates().tolist()
    regions = ind_expr["s"].astype(str).drop_duplicates().tolist()
    
    # res_summary = []

    ind_expr["class_corr"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["femaleness"] >= 0)) |
        ((ind_expr["sex"] == "male") & (ind_expr["femaleness"] < 0))
    )

    ind_expr.to_csv(f"{results_path}/correct_classification.csv", index=False)

    movie_class_summary = []
    for curr_mov in movies:
        mv_class = ind_expr.loc[ind_expr["movie"] == curr_mov, ["sex","class_corr"]].reset_index(drop=True)
        mv_class_fem = ind_expr.loc[mv_class["sex"] == "female", ["class_corr"]].reset_index(drop=True)
        mv_class_mal = ind_expr.loc[mv_class["sex"] == "male", ["class_corr"]].reset_index(drop=True)




    # region_class_summary = []

    # ERRROR - all Subjects are male - also, femaleness_scores.csv already produced in 2_loo_visualization

    # for subj in subjects:
    #     sub_res = ind_expr.loc[ind_expr["subject"] == subj, ["sex","movie","femaleness"]].reset_index(drop=True)
    #     sub_sex = ind_expr["sex"].astype(str).drop_duplicates().tolist()
    #     sub_sex = sub_sex[0]

    #     for mv_str in movies: 
    #         sub_res_mov = sub_res.loc[sub_res["movie"] == mv_str, ["sex","femaleness"]].reset_index(drop=True)
    #         mean_score = sub_res_mov["femaleness"].mean()


    #         res_summary.append({"subject": subj, "sex": sub_sex, "movie": mv_str,  "femaleness": mean_score})

    # res_sum_df = pd.DataFrame(res_summary)
    # res_sum_df.to_csv(f"{results_path}/individual_expressions_summary.csv", index=False)

# Execute script
if __name__ == "__main__":
    main()
