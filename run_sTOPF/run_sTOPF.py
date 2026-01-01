#adjusted version
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket
import _1a_sTOPF_PCA_per_sex
import _1b_sTOPF_loo_PCA
import _2a_sTOPF_result_full_group_PCA
import _2b_sTOPF_individual_expressions
import _3_sTOPF_analyse_results
import _4a_sTOPF_visualize_group_glass_brains
import _4b_sTOPF_visualize_individual_glass_brains
import _5b_ind_classification

# Setup for paths
hostname = socket.gethostname()
if "cpu" in hostname: # Run on Juseless

    # Arguments 

    base_path = sys.argv[1]
    project_ext = sys.argv[2]

    # Parameter for Mutual Information Estimation
    nn_mi = int(sys.argv[3])

    # wkdir = sys.argv[1] # Project directory
    # r_rootdir = sys.argv[2] # Result root directory
    # phenotype = sys.argv[3]  # Phenotype file 
    # complete_participants = sys.argv[4] # Complete participants file
    # excluded_subjects = sys.argv[5] # Exclusion file due to hormonal outliers
    # dataset = sys.argv[6] 
    
    # dataset_list = dataset.split(",") # Split dataset into a list
    # print(f"Dataset list: {dataset_list}")
    # number_of_movies = len(dataset_list) # Number of movies
    # print(f"number of movies {number_of_movies}")
    
    # # Define paths and Check if they exist
    # base_path = f"{wkdir}/data"
    # movie_path =  f"{base_path}/{dataset_list[0]}.csv" # Path to fMRI data - first movie
    # phenotype_path = f"{wkdir}/data/{phenotype}.csv"
    # complete_participants_path = f"{wkdir}/data/{complete_participants}.csv"
    # exclude_path = f"{wkdir}/data/{excluded_subjects}.csv"

else:
    # Local setup for testing 
    
    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    project_ext = "v2"

    # dataset_list = ["BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7", "BOLD_Schaefer400_subcor36_mean_task-tgtbtu_MOVIES_INM7"] # only 2 movies
    # dataset = "BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7.csv" 
    # base_path =  "/Users/kbauer/Desktop/master thesis/codes/fMRIdata" 
    # movie_path =  f"{base_path}/{dataset}" # Path to fMRI data
    # phenotype_path = f"{base_path}/movies_phenotype_results.csv"
    # complete_participants_path = f"{base_path}/complete_participants.csv"
    # exclude_path = f"{base_path}/outlier_results/excluded_subjects.csv"
    # Parameter for Mutual Information Estimation
    
    nn_mi = 3

# Define movie timepoint parameters
mov_prop = {
    "DD": {"min_timepoint": 6, "max_timepoint": 463},
    "S": {"min_timepoint": 6, "max_timepoint": 445},
    "DPS": {"min_timepoint": 6, "max_timepoint": 479},
    "FG": {"min_timepoint": 6, "max_timepoint": 591},
    "DMV": {"min_timepoint": 6, "max_timepoint": 522},
    "LIB": {"min_timepoint": 6, "max_timepoint": 454},
    "TGTBTU": {"min_timepoint": 6, "max_timepoint": 512},
    "SS": {"min_timepoint": 6, "max_timepoint": 642},
    "REST1": {"min_timepoint": 6, "max_timepoint": 499},
    "REST2": {"min_timepoint": 6, "max_timepoint": 499}
}

for path in [base_path]:
    if not os.path.exists(path): 
        print(f"File not found: {path}")
        raise FileNotFoundError
# print(f"\nPath and Files found: \n - {movie_path}\n - {phenotype_path} \n - {complete_participants_path}\n {exclude_path}\n")    
print(f"\n Path and Files found: \n - {base_path}\n")    

#_1a_sTOPF_PCA_per_sex.main(base_path, project_ext, mov_prop)
#_1b_sTOPF_loo_PCA.main(base_path, project_ext, mov_prop)
#_2a_sTOPF_result_full_group_PCA.main(base_path, project_ext, nn_mi, mov_prop)
_2b_sTOPF_individual_expressions.main(base_path, project_ext, nn_mi, mov_prop)
_3_sTOPF_analyse_results.main(base_path, project_ext, nn_mi, mov_prop)
_4a_sTOPF_visualize_group_glass_brains.main(base_path, project_ext, nn_mi, mov_prop)
_4b_sTOPF_visualize_individual_glass_brains.main(base_path, project_ext, nn_mi, mov_prop)
_5b_ind_classification.main(base_path, project_ext, nn_mi, mov_prop)
