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
    regions = ind_expr["region"].astype(str).drop_duplicates().tolist()
    
    # res_summary = []

    ind_expr["class_corr"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["femaleness"] >= 0)) |
        ((ind_expr["sex"] == "male") & (ind_expr["femaleness"] < 0))
    )

    ind_expr["class_corr_sim"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["fem_similarity"] >= 0)) |
        ((ind_expr["sex"] == "male") & (ind_expr["fem_similarity"] < 0))
    )
    ind_expr.to_csv(f"{results_path}/correct_classification_femaleness.csv", index=False)


    movie_class_summary = []
    for curr_mov in movies:
        mv_class = ind_expr.loc[ind_expr["movie"] == curr_mov, ["sex","class_corr","class_corr_sim"]].reset_index(drop=True)

        mv_class_fem = mv_class.loc[mv_class["sex"] == "female", ["class_corr"]].reset_index(drop=True)
        mv_class_mal = mv_class.loc[mv_class["sex"] == "male", ["class_corr"]].reset_index(drop=True)
        count_true_fem = mv_class_fem["class_corr"].sum()
        count_true_mal = mv_class_mal["class_corr"].sum()
        nr_fem = len(mv_class_fem)
        nr_mal = len(mv_class_mal)

        mv_class_fem_sim = mv_class.loc[mv_class["sex"] == "female", ["class_corr_sim"]].reset_index(drop=True)
        mv_class_mal_sim = mv_class.loc[mv_class["sex"] == "male", ["class_corr_sim"]].reset_index(drop=True)
        count_true_fem_sim = mv_class_fem_sim["class_corr_sim"].sum()
        count_true_mal_sim = mv_class_mal_sim["class_corr_sim"].sum()
        nr_fem_sim = len(mv_class_fem_sim)
        nr_mal_sim = len(mv_class_mal_sim)

        movie_class_summary.append({"movie": curr_mov, "female corr femaleness": count_true_fem/nr_fem, "male corr femaleness": count_true_mal/nr_mal, "female corr fem_sim": count_true_fem_sim/nr_fem_sim, "male corr fem_sim": count_true_mal_sim/nr_mal_sim})

    movie_class_summary_df = pd.DataFrame(movie_class_summary)
    movie_class_summary_df.to_csv(f"{results_path}/correct_classification_per_movie.csv", index=False)

    region_class_summary = []
    for curr_reg in regions:
        reg_class = ind_expr.loc[ind_expr["region"] == curr_reg, ["sex","class_corr","class_corr_sim"]].reset_index(drop=True)

        reg_class_fem = reg_class.loc[reg_class["sex"] == "female", ["class_corr"]].reset_index(drop=True)
        reg_class_mal = reg_class.loc[reg_class["sex"] == "male", ["class_corr"]].reset_index(drop=True)
        count_true_fem_r = reg_class_fem["class_corr"].sum()
        count_true_mal_r = reg_class_mal["class_corr"].sum()
        nr_fem = len(reg_class_fem)
        nr_mal = len(reg_class_mal)

        reg_class_fem_sim = reg_class.loc[reg_class["sex"] == "female", ["class_corr_sim"]].reset_index(drop=True)
        reg_class_mal_sim = reg_class.loc[reg_class["sex"] == "male", ["class_corr_sim"]].reset_index(drop=True)
        count_true_fem_r_sim = reg_class_fem_sim["class_corr_sim"].sum()
        count_true_mal_r_sim = reg_class_mal_sim["class_corr_sim"].sum()
        nr_fem_sim = len(reg_class_fem_sim)
        nr_mal_sim = len(reg_class_mal_sim)

        region_class_summary.append({"region": curr_reg, "female corr femaleness": count_true_fem_r/nr_fem, "male corr femaleness": count_true_mal_r/nr_mal, "female corr fem_sim": count_true_fem_r_sim/nr_fem_sim, "male corr fem_sim": count_true_mal_r_sim/nr_mal_sim})

    region_class_summary_df = pd.DataFrame(region_class_summary)
    region_class_summary_df.to_csv(f"{results_path}/correct_classification_per_region.csv", index=False)



# Execute script
if __name__ == "__main__":
    main()
