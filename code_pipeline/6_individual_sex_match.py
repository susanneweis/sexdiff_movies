import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

def main():
    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies"    
    data_path = f"{base_path}/data_pipeline"
    results_path = f"{base_path}/results_pipeline" 

    movie_path =  f"{data_path}/fMRIdata" 
    pca_base_path = f"{results_path}/results_PCA"

    complete_participants_path = f"{results_path}/complete_participants.csv"

    outpath = f"{results_path}/results_individual_exp"
    os.makedirs(outpath, exist_ok=True)

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]
  
    movies_properties = {
        "dd": {"min_timepoint": 6, "max_timepoint": 463},
        "s": {"min_timepoint": 6, "max_timepoint": 445},
        "dps": {"min_timepoint": 6, "max_timepoint": 479},
        "fg": {"min_timepoint": 6, "max_timepoint": 591},
        "dmw": {"min_timepoint": 6, "max_timepoint": 522},
        "lib": {"min_timepoint": 6, "max_timepoint": 454},
        "tgtbtu": {"min_timepoint": 6, "max_timepoint": 512}
    }

    results = []

    for curr_mov in movies:

        bold_data = f"BOLD_Schaefer400_subcor36_mean_task-{curr_mov}_MOVIES_INM7.csv" 
        bold_df = pd.read_csv(f"{movie_path}/{bold_data}")

        pca_movie_path = f"{pca_base_path}/{curr_mov}"

        typical_fem = pd.read_csv(f"{pca_movie_path}/PC1_scores_female_allROI.csv")
        typical_mal = pd.read_csv(f"{pca_movie_path}/PC1_scores_male_allROI.csv")
        
        # Correlation per subject × region across time

        subject_col = "subject"
        regions = typical_fem["Region"].drop_duplicates().tolist()

        complete_subjects = set(pd.read_csv(complete_participants_path)["subject"].astype(str))
        bold_df = bold_df[bold_df["subject"].isin(complete_subjects)].copy()

        for subj, sub_df in bold_df.groupby(subject_col, sort=False):

            sub = sub_df.set_index("timepoint")

            for region in regions:
                x = sub[region]
                #Discard first 5 timepoints
                min_t = movies_properties[curr_mov]["min_timepoint"]
                max_t = movies_properties[curr_mov]["max_timepoint"]
                x = x[min_t-1 : max_t]

                yf = typical_fem.loc[typical_fem["Region"] == region, "PC_score_1"]
                rf, p = pearsonr(x, yf)
                ym = typical_mal.loc[typical_fem["Region"] == region, "PC_score_1"]
                rm, p = pearsonr(x, ym)

                results.append({"subject": subj, "region": region, "correlation_female": rf, "correlation_male": rm})

        out_df = pd.DataFrame(results, columns=["subject", "region", "correlation_female", "correlation_male"])
        os.makedirs(outpath, exist_ok=True)
        out_csv = f"{outpath}/indiviudal_exp_{curr_mov}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()