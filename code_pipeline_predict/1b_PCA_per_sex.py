#adjusted version
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket
import re

# Function to extract the movie name from the filename
def extract_movie_part(movie):
    match = re.search(r'task-(.*?)_MOVIES', movie)
    if match:
        return match.group(1)
    else:
        return None
    if movie_abbrev is None:
        raise ValueError(f"Could not extract movie abbreviation from: {movie}")
    
# Standardize data using StandardScaler (zero mean, unit variance for each feature)
def standardize_data(matrix):
    scaler = StandardScaler() 
    return matrix.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)
# PCA Function

def perform_pca(matrix):
    if matrix.empty:
        return None, None, None # Return None if matrix is empty
                
    pca = PCA(n_components=2) # Apply PCA with 2 components (PC1 and PC2)
    pc_scores = pca.fit_transform(matrix) # Transform the data to get the PCA scores
    explained_variance = pca.explained_variance_ratio_ # Variance explained by each component

    # Calculate PCA loadings
    pc_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc_loadings_df = pd.DataFrame({
        "Subject_ID": matrix.columns,
        "PC1_loading": pc_loadings[:, 0],
        "PC2_loading": pc_loadings[:, 1]
    })
                
    pc_scores_df = pd.DataFrame(pc_scores, columns=[f'PC{i+1}_score' for i in range(2)])
                             
    return pc_loadings_df, pc_scores_df, explained_variance[0], explained_variance[1], 

def main(): 




    # from here PCA on combined movies     
    dataset_list = [
        f"BOLD_Schaefer400_subcor36_mean_task-{movie}_MOVIES_INM7"
        for movie in movies
    ]

    all_data = [] # List to store all movie data
    # Loop through each movie in the dataset list
    for movie in dataset_list:    
        movie_path =  f"{data_path}/fMRIdata/{movie}.csv" # Path to current movie data
        movie_abbrev = extract_movie_part(movie) # Extract movie abbrevation
        properties = movies_properties[movie_abbrev] # Get timepoint properties for the movie
    
        # Load fMRI data
        movie_data = pd.read_csv(movie_path)
        if "Unnamed: 0" in movie_data.columns:
            movie_data = movie_data.drop(columns=["Unnamed: 0"]) # Drop unnecessary columns
        
        # Define column names and brain regions
        subject_column, time_column = "subject", "timepoint"
        brain_regions = movie_data.columns[2:]  # Extract all brain region columns (assuming the first two columns are not brain regions) 

        # Filter timepoints based on movie properties
        movie_data = movie_data[
            (movie_data["timepoint"] >= properties["min_timepoint"]) & 
            (movie_data["timepoint"] <= properties["max_timepoint"])
        ] 
        print(f"movie properties {movie_abbrev}", movie_data["timepoint"].min(), movie_data["timepoint"].max(),"\n") 
    
        #Filter subjects based on the valid subject list
        movie_data = movie_data[movie_data["subject"].isin(valid_subjects)]
        movie_data["movie"] = movie_abbrev  # Add movie identifier to the data
        all_data.append(movie_data)     # Append to the list of all movie data

    output_dir = f"{results_path}/results_PCA/concatenated_PCA" # Local results directory
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

    # Prepare lists for PCA results by gender 
    pc1_loadings_female_allROIs = []
    pc2_loadings_female_allROIs = []
    pc1_scores_female_allROIs = []
    pc2_scores_female_allROIs = []
    explained_variance_1_female_allROIs = []
    explained_variance_2_female_allROIs = []

    pc1_loadings_male_allROIs = []
    pc2_loadings_male_allROIs = []
    pc1_scores_male_allROIs = []
    pc2_scores_male_allROIs = []
    explained_variance_1_male_allROIs = []
    explained_variance_2_male_allROIs = []

    # Create a dictionary to store the standardized matrices for each brain region
    standardized_matrices = {}
    # Create a dictionary to store the combined matrices for each brain region
    concatenated_matrices = {}

    # Perform PCA on each brain region
    for region in brain_regions:        
        region_matrices = []  # List to store matrices from all movies for this region
    
        for movie_data, movie in zip(all_data, dataset_list):  # Iterate through all movies   
            movie_abbrev = extract_movie_part(movie) # Extract movie abbreviation
        
            # Extract and format data for the current region and movie
            region_data = movie_data[[subject_column, time_column, region]]
            formatted_matrix = region_data.pivot(index=time_column, columns=subject_column, values=region)
        
            # Add movie_abbrevation to matrix 
            formatted_matrix.insert(0, "movie_abbrev", movie_abbrev)  # Insert movie abbreviation as first column
            standardized_matrix = standardize_data(formatted_matrix.drop(columns=["movie_abbrev"]))  # Standardize data (excluding movie)
            standardized_matrix.insert(0, "movie_abbrev", movie_abbrev)  # Reinsert movie abbreviation
        
            # Store the standardized matrix for later merging
            region_matrices.append(standardized_matrix)
        
        # Concatenate matrices for the region across all movies (stacked by timepoints)
        combined_matrix = pd.concat(region_matrices, axis=0)
        concatenated_matrices[region] = combined_matrix
    
        # Separate subjects by gender for PCA
        female_subjects = phenotypes[phenotypes['gender'] == 2]['subject_ID']
        male_subjects = phenotypes[phenotypes['gender'] == 1]['subject_ID']
    
        # Ensure the subjects exist in the standardized matrix
        female_subjects = female_subjects[female_subjects.isin(combined_matrix.columns)]
        male_subjects = male_subjects[male_subjects.isin(combined_matrix.columns)]

        # Perform PCA separatley for males and females
        concatenated_matrix_female = combined_matrix.loc[:, female_subjects]
        concatenated_matrix_male = combined_matrix.loc[:, male_subjects]

        # Perform PCA for females
        pc_loadings_female, pc_scores_female, explained_variance_female_1, explained_variance_female_2 = perform_pca(concatenated_matrix_female)
        if pc_loadings_female is not None:
            for idx, row in pc_loadings_female.iterrows():
                pc1_loadings_female_allROIs.append([region, row['Subject_ID'], row['PC1_loading']])
                pc2_loadings_female_allROIs.append([region, row['Subject_ID'], row['PC2_loading']])
            for idx, value in enumerate(pc_scores_female['PC1_score']):
                pc1_scores_female_allROIs.append([region, value])
            for idx, value in enumerate(pc_scores_female['PC2_score']):
                pc2_scores_female_allROIs.append([region, value])
            explained_variance_1_female_allROIs.append([region, explained_variance_female_1])
            explained_variance_2_female_allROIs.append([region, explained_variance_female_2])
    
        # Perform PCA for males
        pc_loadings_male, pc_scores_male, explained_variance_male_1, explained_variance_male_2 = perform_pca(concatenated_matrix_male)
        if pc_loadings_male is not None:
            for idx, row in pc_loadings_male.iterrows():
                pc1_loadings_male_allROIs.append([region, row['Subject_ID'], row['PC1_loading']])
                pc2_loadings_male_allROIs.append([region, row['Subject_ID'], row['PC2_loading']])
            for idx, value in enumerate(pc_scores_male['PC1_score']):
                pc1_scores_male_allROIs.append([region, value])
            for idx, value in enumerate(pc_scores_male['PC2_score']):
                pc2_scores_male_allROIs.append([region, value])
            explained_variance_1_male_allROIs.append([region, explained_variance_male_1])
            explained_variance_2_male_allROIs.append([region, explained_variance_male_2])

    # Save the PCA results to CSV files for each gender
    pd.DataFrame(pc1_loadings_female_allROIs, columns=["Region", "Subject_ID", "PC_loading_1"]).to_csv(f"{output_dir}/PC1_loadings_female_allROI.csv", index=False)
    pd.DataFrame(pc2_loadings_female_allROIs, columns=["Region", "Subject_ID", "PC_loading_2"]).to_csv(f"{output_dir}/PC2_loadings_female_allROI.csv", index=False)
    pd.DataFrame(pc1_scores_female_allROIs, columns=["Region", "PC_score_1"]).to_csv(f"{output_dir}/PC1_scores_female_allROI.csv", index=False)
    pd.DataFrame(pc2_scores_female_allROIs, columns=["Region", "PC_score_2"]).to_csv(f"{output_dir}/PC2_scores_female_allROI.csv", index=False)
    pd.DataFrame(explained_variance_1_female_allROIs, columns=["Region", "explained_variance_1"]).to_csv(f"{output_dir}/explained_variance_1_female_allROI.csv", index=False)
    pd.DataFrame(explained_variance_2_female_allROIs, columns=["Region", "explained_variance_2"]).to_csv(f"{output_dir}/explained_variance_2_female_allROI.csv", index=False)
    pd.DataFrame(pc1_loadings_male_allROIs, columns=["Region", "Subject_ID", "PC_loading_1"]).to_csv(f"{output_dir}/PC1_loadings_male_allROI.csv", index=False)
    pd.DataFrame(pc2_loadings_male_allROIs, columns=["Region", "Subject_ID", "PC_loading_2"]).to_csv(f"{output_dir}/PC2_loadings_male_allROI.csv", index=False)
    pd.DataFrame(pc1_scores_male_allROIs, columns=["Region", "PC_score_1"]).to_csv(f"{output_dir}/PC1_scores_male_allROI.csv", index=False)
    pd.DataFrame(pc2_scores_male_allROIs, columns=["Region", "PC_score_2"]).to_csv(f"{output_dir}/PC2_scores_male_allROI.csv", index=False)
    pd.DataFrame(explained_variance_1_male_allROIs, columns=["Region", "explained_variance_1"]).to_csv(f"{output_dir}/explained_variance_1_male_allROI.csv", index=False)
    pd.DataFrame(explained_variance_2_male_allROIs, columns=["Region", "explained_variance_2"]).to_csv(f"{output_dir}/explained_variance_2_male_allROI.csv", index=False)
    print(f"PCA analysis completed. The results have been saved to {output_dir}")
   

# Execute script
if __name__ == "__main__":
    main()