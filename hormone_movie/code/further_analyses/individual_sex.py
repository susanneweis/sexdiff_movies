import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket
import re





def main():
    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie" 
    movie_path =  f"{base_path}/data/fMRIdata" 
    pca_base_path = f"{base_path}/results/results_PCA"

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

    for curr_mov in movies:
        bold_data = f"BOLD_Schaefer400_subcor36_mean_task-{curr_mov}_MOVIES_INM7.csv" 
        pca_movie_path = f"{pca_base_path}"/{curr_mov}"

        ind_tc = pd.read_csv(f"{movie_path}/{bold_data}")
        pca_fem = pd.read_csv(f"{pca_movie_path}/PC1_scores_female_allROI.csv")
        pca_mal = pd.read_csv(f"{pca_movie_path}/PC1_scores_male_allROI.csv")


    



if __name__ == "__main__":
    main()