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

def main(): 

    # Local setup for testing 
    # for Juseless Version see Kristina's code: PCA_foreachsex_allROI_latestversion.py

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    data_path = f"{base_path}/data_pipeline"
    results_path = f"{base_path}/results_pipeline"

    phenotype_path = f"{data_path}/Participant_sex_info.csv"
    complete_participants_path = f"{results_path}/complete_participants.csv"
    # not relevant yet, as currently not considering hormones
    # exclude_path = f"{base_path}/results_pipeline/excluded_subjects.csv"

    # Define movie timepoint parameters
    movies_properties = {
        "dd": {"min_timepoint": 6, "max_timepoint": 463},
        "s": {"min_timepoint": 6, "max_timepoint": 445},
        "dps": {"min_timepoint": 6, "max_timepoint": 479},
        "fg": {"min_timepoint": 6, "max_timepoint": 591},
        "dmw": {"min_timepoint": 6, "max_timepoint": 522},
        "lib": {"min_timepoint": 6, "max_timepoint": 454},
        "tgtbtu": {"min_timepoint": 6, "max_timepoint": 512}
    }

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]
    for curr_mov in movies:
        dataset = f"BOLD_Schaefer400_subcor36_mean_task-{curr_mov}_MOVIES_INM7.csv"
        movie_path =  f"{data_path}/fMRIdata/{dataset}" # Path to fMRI data

        for path in [movie_path, phenotype_path, complete_participants_path]:
        # for path in [movie_path, phenotype_path, complete_participants_path, exclude_path]:           
            if not os.path.exists(path): 
                print(f"File not found: {path}")
                raise FileNotFoundError
        print(f"\nPath and Files found: \n - {movie_path}\n - {phenotype_path} \n - {complete_participants_path}\n")
        #print(f"\nPath and Files found: \n - {movie_path}\n - {phenotype_path} \n - {complete_participants_path}\n {exclude_path}\n")
            
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

        # for each movie seperately 
        all_data = [] # List to store all movie data

        # Loop through each movie in the dataset list
        
        properties = movies_properties[curr_mov] # Get timepoint properties for the movie
            
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
        print(f"movie properties {curr_mov}", movie_data["timepoint"].min(), movie_data["timepoint"].max(),"\n") 
            
        # Filter subjects based on the valid subject list
        movie_data = movie_data[movie_data["subject"].isin(valid_subjects)]
        movie_data["movie"] = curr_mov  # Add movie identifier to the data
        all_data.append(movie_data)     # Append to the list of all movie data

        # Define the output directory
        # if hostname == "cpu44":
        #   output_dir =r_rootdir # Remote root directory
        #else:
        output_dir = f"{results_path}/results_PCA/{curr_mov}" # Local results directory
        os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist


        # use this for concatenated data
        # all_data = [] # List to store all movie data

        # # Loop through each movie in the dataset list
        # for movie in dataset_list:    
        #     movie_path =  f"{base_path}//data/fMRIdata/{movie}.csv" # Path to current movie data
        #     movie_abbrev = extract_movie_part(movie) # Extract movie abbrevation
        #     properties = movies_properties[movie_abbrev] # Get timepoint properties for the movie
            
        #     # Load fMRI data
        #     movie_data = pd.read_csv(movie_path)
        #     if "Unnamed: 0" in movie_data.columns:
        #         movie_data = movie_data.drop(columns=["Unnamed: 0"]) # Drop unnecessary columns
                
        #     # Define column names and brain regions
        #     subject_column, time_column = "subject", "timepoint"
        #     brain_regions = movie_data.columns[2:]  # Extract all brain region columns (assuming the first two columns are not brain regions) 

        #     # Filter timepoints based on movie properties
        #     movie_data = movie_data[
        #         (movie_data["timepoint"] >= properties["min_timepoint"]) & 
        #         (movie_data["timepoint"] <= properties["max_timepoint"])
        #     ] 
        #     print(f"movie properties {movie_abbrev}", movie_data["timepoint"].min(), movie_data["timepoint"].max(),"\n") 
            
        #     # Filter subjects based on the valid subject list
        #     movie_data = movie_data[movie_data["subject"].isin(valid_subjects)]
        #     movie_data["movie"] = movie_abbrev  # Add movie identifier to the data
        #     all_data.append(movie_data)     # Append to the list of all movie data

        # # Define the output directory
        # if hostname == "cpu44":
        #     output_dir =r_rootdir # Remote root directory
        # else:
        #     output_dir = f"{base_path}/results/results_PCA" # Local results directory
        #     os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

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

        # Standardize data using StandardScaler (zero mean, unit variance for each feature)
        def standardize_data(matrix):
            scaler = StandardScaler() 
            return matrix.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)

        # Create a dictionary to store the standardized matrices for each brain region
        standardized_matrices = {}
        # Create a dictionary to store the combined matrices for each brain region
        concatenated_matrices = {}

        # Perform PCA on each brain region
        for region in brain_regions:        
            region_matrices = []  # List to store matrices from all movies for this region
              
            region_data = movie_data[[subject_column, time_column, region]]
            formatted_matrix = region_data.pivot(index=time_column, columns=subject_column, values=region)
                
            standardized_matrix = standardize_data(formatted_matrix)  # Standardize data (excluding movie)
                
            # # Store the standardized matrix for later merging
            # region_matrices.append(standardized_matrix)
                
            # # Concatenate matrices for the region across all movies (stacked by timepoints)
            # combined_matrix = pd.concat(region_matrices, axis=0)
            # concatenated_matrices[region] = combined_matrix
            
            # Separate subjects by gender for PCA
            female_subjects = phenotypes[phenotypes['gender'] == 2]['subject_ID']
            male_subjects = phenotypes[phenotypes['gender'] == 1]['subject_ID']
            
            # Ensure the subjects exist in the standardized matrix
            female_subjects = female_subjects[female_subjects.isin(standardized_matrix.columns)]
            male_subjects = male_subjects[male_subjects.isin(standardized_matrix.columns)]

            # Perform PCA separatley for males and females
            # keep the original naming for further reference

            concatenated_matrix_female = standardized_matrix.loc[:, female_subjects]
            concatenated_matrix_male = standardized_matrix.loc[:, male_subjects]

     
            # this will be needed later for connected matrix

            # for movie_data, movie in zip(all_data, dataset_list):  # Iterate through all movies   
            #     movie_abbrev = extract_movie_part(movie) # Extract movie abbreviation
                
            #     # Extract and format data for the current region and movie
            #     region_data = movie_data[[subject_column, time_column, region]]
            #     formatted_matrix = region_data.pivot(index=time_column, columns=subject_column, values=region)
                
            #     # Add movie_abbrevation to matrix 
            #     formatted_matrix.insert(0, "movie_abbrev", movie_abbrev)  # Insert movie abbreviation as first column
            #     standardized_matrix = standardize_data(formatted_matrix.drop(columns=["movie_abbrev"]))  # Standardize data (excluding movie)
            #     standardized_matrix.insert(0, "movie_abbrev", movie_abbrev)  # Reinsert movie abbreviation
                
            #     # Store the standardized matrix for later merging
            #     region_matrices.append(standardized_matrix)
                
            # # Concatenate matrices for the region across all movies (stacked by timepoints)
            # combined_matrix = pd.concat(region_matrices, axis=0)
            # concatenated_matrices[region] = combined_matrix
            
            # # Separate subjects by gender for PCA
            # female_subjects = phenotypes[phenotypes['gender'] == 2]['subject_ID']
            # male_subjects = phenotypes[phenotypes['gender'] == 1]['subject_ID']
            
            # # Ensure the subjects exist in the standardized matrix
            # female_subjects = female_subjects[female_subjects.isin(combined_matrix.columns)]
            # male_subjects = male_subjects[male_subjects.isin(combined_matrix.columns)]

            # # Perform PCA separatley for males and females
            # concatenated_matrix_female = combined_matrix.loc[:, female_subjects]
            # concatenated_matrix_male = combined_matrix.loc[:, male_subjects]

            # end uses later 

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
                
                # Store explained variance
                explained_variance_df = pd.DataFrame({
                    "ROI": [region],
                    "PC1_Explained_Variance": [explained_variance[0]],
                    "PC2_Explained_Variance": [explained_variance[1]]
                })
                
                return pc_loadings_df, pc_scores_df, explained_variance_df

            # Perform PCA for females
            pc_loadings_female, pc_scores_female, explained_variance_female = perform_pca(concatenated_matrix_female)
            if pc_loadings_female is not None:
                for idx, row in pc_loadings_female.iterrows():
                    pc1_loadings_female_allROIs.append([region, row['Subject_ID'], row['PC1_loading']])
                    pc2_loadings_female_allROIs.append([region, row['Subject_ID'], row['PC2_loading']])
                for idx, value in enumerate(pc_scores_female['PC1_score']):
                    pc1_scores_female_allROIs.append([region, value])
                for idx, value in enumerate(pc_scores_female['PC2_score']):
                    pc2_scores_female_allROIs.append([region, value])
                explained_variance_1_female_allROIs.append([region, explained_variance_female.iloc[0, 1]])
                explained_variance_2_female_allROIs.append([region, explained_variance_female.iloc[0, 2]])
            
            # Perform PCA for males
            pc_loadings_male, pc_scores_male, explained_variance_male = perform_pca(concatenated_matrix_male)
            if pc_loadings_male is not None:
                for idx, row in pc_loadings_male.iterrows():
                    pc1_loadings_male_allROIs.append([region, row['Subject_ID'], row['PC1_loading']])
                    pc2_loadings_male_allROIs.append([region, row['Subject_ID'], row['PC2_loading']])
                for idx, value in enumerate(pc_scores_male['PC1_score']):
                    pc1_scores_male_allROIs.append([region, value])
                for idx, value in enumerate(pc_scores_male['PC2_score']):
                    pc2_scores_male_allROIs.append([region, value])
                explained_variance_1_male_allROIs.append([region, explained_variance_male.iloc[0, 1]])
                explained_variance_2_male_allROIs.append([region, explained_variance_male.iloc[0, 2]])

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