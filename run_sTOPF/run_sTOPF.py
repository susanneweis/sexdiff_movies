#adjusted version
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket
import re
import _1a_sTOPF_PCA_per_sex

# Setup for paths
hostname = socket.gethostname()
if "cpu" in hostname: # Run on Juseless

    # Arguments 

    base_path = sys.argv[1]

    # Parameter for Mutual Information Estimation
    nn_for_mi = sys.argv[2]

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

    # Parameter for Mutual Information Estimation
    nn_for_mi = 3


    # dataset_list = ["BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7", "BOLD_Schaefer400_subcor36_mean_task-tgtbtu_MOVIES_INM7"] # only 2 movies
    # dataset = "BOLD_Schaefer400_subcor36_mean_task-dps_MOVIES_INM7.csv" 
    # base_path =  "/Users/kbauer/Desktop/master thesis/codes/fMRIdata" 
    # movie_path =  f"{base_path}/{dataset}" # Path to fMRI data
    # phenotype_path = f"{base_path}/movies_phenotype_results.csv"
    # complete_participants_path = f"{base_path}/complete_participants.csv"
    # exclude_path = f"{base_path}/outlier_results/excluded_subjects.csv"

for path in [base_path]:
    if not os.path.exists(path): 
        print(f"File not found: {path}")
        raise FileNotFoundError
# print(f"\nPath and Files found: \n - {movie_path}\n - {phenotype_path} \n - {complete_participants_path}\n {exclude_path}\n")    
print(f"\n Path and Files found: \n - {base_path}\n")    

1a_sTOPF_PCA_per_sex_1a.main(base_path)
