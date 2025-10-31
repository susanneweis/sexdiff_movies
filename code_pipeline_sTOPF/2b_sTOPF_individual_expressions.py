import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket
import re
from scipy.stats import pearsonr
from compute_PCA import perform_pca
from compute_PCA import standardize_data
import statsmodels.api as sm

def main(): 
    # Local setup for testing 
    # for Juseless Version see Kristina's code: PCA_foreachsex_allROI_latestversion.py

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    data_path = f"{base_path}/data_pipeline_sTOPF"

    ### change later 
    results_path = f"{base_path}/results_pipeline_predict"
    ind_path = f"{results_path}/individual_expressions"
    os.makedirs(ind_path, exist_ok=True)

    phenotype_path = f"{data_path}/Participant_sex_info.csv"
    complete_participants_path = f"{data_path}/complete_participants.csv"
    # not relevant yet, as currently not considering hormones
    # exclude_path = f"{base_path}/results_pipeline/excluded_subjects.csv"

    sex_mapping = {1: 'male', 2: 'female'}
    subs_sex = pd.read_csv(f"{data_path}/Participant_sex_info.csv", sep = ";")
    subs_sex['gender'] = subs_sex['gender'].replace(sex_mapping)

    # Define movie timepoint parameters
    movies_properties = {
        "dd": {"min_timepoint": 6, "max_timepoint": 463},
        "s": {"min_timepoint": 6, "max_timepoint": 445},
        "dps": {"min_timepoint": 6, "max_timepoint": 479},
        "fg": {"min_timepoint": 6, "max_timepoint": 591},
        "dmw": {"min_timepoint": 6, "max_timepoint": 522},
        "lib": {"min_timepoint": 6, "max_timepoint": 454},
        "tgtbtu": {"min_timepoint": 6, "max_timepoint": 512},
        "rest_run-1": {"min_timepoint": 6, "max_timepoint": 499},
        "rest_run-2": {"min_timepoint": 6, "max_timepoint": 499}
    }

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "rest_run-1", "rest_run-2"]

    # Load phenotype data (assumed to be a CSV with a subject ID and gender columns)
    phenotypes = pd.read_csv(phenotype_path, sep=';')
    phenotypes.columns = ['subject_ID', 'gender']

    # Load list of complete participants (verified list with participants_verification.py)
    complete_participants = set(pd.read_csv(complete_participants_path)['subject'].astype(str))

    # Load list of excluded subjects (hormonal outlier detection with hormone_outlier_detection_SD.py)
    # not yet relevant here 
    # exclude_df = pd.read_csv(exclude_path, sep=',')
    # excluded_subjects = set(exclude_df['PCode'].astype(str))

    # Get valid subjects and exclude outliers
    phenotype_subjects = set(phenotypes['subject_ID'].astype(str))
    valid_subjects = complete_participants.intersection(phenotype_subjects)
    # not yet relevant here
    # valid_subjects = valid_subjects.difference(excluded_subjects)

    print(f"Number of included valid subjects after exclusion: {len(valid_subjects)}")

    loo_results_all = []

    for subj in valid_subjects:

        loo_results_subj = []

        for curr_mov in movies:
            dataset = f"BOLD_Schaefer400_subcor36_mean_task-{curr_mov}_MOVIES_INM7.csv"
            movie_path =  f"{data_path}/fMRIdata/{dataset}" # Path to fMRI data

            properties = movies_properties[curr_mov] # Get timepoint properties for the movie
            
            # Load fMRI data
            movie_data = pd.read_csv(movie_path)
            if "Unnamed: 0" in movie_data.columns:
                movie_data = movie_data.drop(columns=["Unnamed: 0"]) # Drop unnecessary columns
                
            # Define column names and brain regions
            brain_regions = movie_data.columns[2:]  # Extract all brain region columns (assuming the first two columns are not brain regions) 

            # Filter timepoints based on movie properties
            movie_data = movie_data[
                (movie_data["timepoint"] >= properties["min_timepoint"]) & 
                (movie_data["timepoint"] <= properties["max_timepoint"])
            ] 
            print(f"movie properties {curr_mov}", movie_data["timepoint"].min(), movie_data["timepoint"].max(),"\n") 
            
            subj_movie_data = movie_data.loc[movie_data["subject"] == subj].copy()

            # Define the output directory
            # if hostname == "cpu44":
            #   output_dir =r_rootdir # Remote root directory
            #else:
            output_dir = f"{results_path}/results_PCA/{curr_mov}/{subj}" # Local results directory
            os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

            for region in brain_regions:        
                
                region_data = subj_movie_data[["subject", "timepoint", region]]
                

                rf, p = pearsonr(subj_movie_data[region], pc_scores_female["PC1_score"])
                rm, p = pearsonr(subj_movie_data[region], pc_scores_male["PC1_score"])

                diff = np.arctanh(rf) - np.arctanh(rm)
                diff = np.tanh(diff)

                # standardize
                y = (subj_movie_data[region] - np.mean(subj_movie_data[region])) / np.std(subj_movie_data[region])
                xf = (pc_scores_female["PC1_score"] - np.mean(pc_scores_female["PC1_score"])) / np.std(pc_scores_female["PC1_score"])
                xm = (pc_scores_male["PC1_score"] - np.mean(pc_scores_male["PC1_score"])) / np.std(pc_scores_male["PC1_score"])

                # design matrix
                X = np.column_stack([xf, xm])
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                beta_f, beta_m = model.params[1], model.params[2]
                fem_similarity = (beta_f - beta_m) / (abs(beta_f) + abs(beta_m))
                
                sub_sex = subs_sex.loc[subs_sex["subject_ID"] == subj, "gender"].iloc[0]

                loo_results_all.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "femaleness": diff, "fem_similarity": fem_similarity})
                loo_results_subj.append({"subject": subj, "sex": sub_sex, "movie": curr_mov, "region": region, "correlation_female": rf, "correlation_male": rm, "femaleness": diff, "fem_similarity": fem_similarity})
            

        
        out_df = pd.DataFrame(loo_results_subj, columns=["subject","sex","movie","region","correlation_female","correlation_male","femaleness","fem_similarity"])
        out_csv = f"{ind_path}/individual_expression_{subj}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    out_df = pd.DataFrame(loo_results_all, columns=["subject","sex","movie","region","correlation_female","correlation_male","femaleness","fem_similarity"])
    out_csv = f"{results_path}/individual_expression_all.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# Execute script
if __name__ == "__main__":
    main()