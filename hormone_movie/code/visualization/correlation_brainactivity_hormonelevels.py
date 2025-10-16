import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm

# Paths
base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie"
atlas_path = f"{base_path}/data/Susanne_Schaefer_436.nii"

PC1_loadings_female_path = f"{base_path}/results/results-PCA/{mv_str}/PC1_loadings_female_allROI.csv"
PC1_loadings_male_path = f"{base_path}/results/results-PCA/{mv_str}/PC1_loadings_male_allROI.csv"

hormone_path = f"{base_path}/data/Hormone_data.csv"
complete_path = f"{base_path}/data/complete_participants.csv" # Participants which completed all movies
exclude_path = f"{base_path}/results/excluded_subjects.csv" # Participants excluded due to outliers

# Check if files exist
for path in [atlas_path, PC1_loadings_female_path, PC1_loadings_male_path, hormone_path, complete_path, exclude_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load datasets
PC1_loadings_female = pd.read_csv(PC1_loadings_female_path, sep=';')
PC1_loadings_male = pd.read_csv(PC1_loadings_male_path)
PC1_loadings_female.rename(columns={"Subject_ID": "PCode"}, inplace=True)
PC1_loadings_male.rename(columns={"Subject_ID": "PCode"}, inplace=True)

hormone_df = pd.read_csv(hormone_path, sep="\t", encoding="ISO-8859-1")
complete_df = pd.read_csv(complete_path)
exclude_df = pd.read_csv(exclude_path, sep=',')

# Filter hormone data to include only participants in complete list
valid_subjects = complete_df['subject'].unique()

# Remove subjects marked as outliers
excluded_subjects = exclude_df['PCode'].unique()
valid_subjects = [subject for subject in valid_subjects if subject not in excluded_subjects]

# Filter hormone data to include only valid participants
hormone_df = hormone_df[hormone_df['PCode'].isin(valid_subjects)]

print(f"Number of participants included: {len(valid_subjects)}")

# Select only the relevant columns for the correlation analysis
hormone_subset = hormone_df[['PCode', 'Cortisol_µg/dl_Mean', 'Estradiol_pg/ml_Mean',
                             'Progesterone_pg/ml_Mean', 'Testosterone_pg/ml_Mean']]

# Merge PC1 loadings with hormone data using the 'PCode' 
merged_female = pd.merge(PC1_loadings_female, hormone_subset, on="PCode")
merged_male = pd.merge(PC1_loadings_male, hormone_subset, on="PCode")

# Define hormones to correlate
hormones = ['Cortisol_µg/dl_Mean', 'Estradiol_pg/ml_Mean', 
            'Progesterone_pg/ml_Mean', 'Testosterone_pg/ml_Mean']

# Store results
region_corr_results = []

# Assign unique numerical IDs to each region name and adds them as a new column
def assign_roi_ids(df):
    unique_regions = df['Region'].drop_duplicates().tolist()
    region_to_id = {region: i+1 for i, region in enumerate(unique_regions)}  
    df['ROI_ID'] = df['Region'].map(region_to_id)
    return df, region_to_id

merged_female, region_to_id_f = assign_roi_ids(merged_female)
merged_male, region_to_id_m = assign_roi_ids(merged_male)

# Function to calculate correlation per region
def calculate_region_correlations(df, sex_label):
    for hormone in hormones:
        grouped = df.groupby('ROI_ID')
        for roi_id, group in grouped:
            if len(group[hormone].dropna()) > 2:
                rho, pval = spearmanr(group['PC_loading_1'], group[hormone])
                region_corr_results.append({
                    'ROI Label': roi_id,
                    'Hormone': hormone,
                    'Sex': sex_label,
                    'Spearman\'s rho': rho,
                    'p-value': pval
                })
# Calculate for both sexes
calculate_region_correlations(merged_female, 'female')
calculate_region_correlations(merged_male, 'male')

# Convert to DataFrame and save
region_corr_df = pd.DataFrame(region_corr_results)
region_corr_df.to_csv(f"{base_path}/regionwise_correlation_results.csv", index=False)
print("Region-wise correlation results saved.")

##### Brain maps
n_roi = 436

# Create output directory for brain maps
brainmap_output_path = os.path.join(base_path, "brain_maps")
os.makedirs(brainmap_output_path, exist_ok=True)

def create_img_for_glassbrain_plot(stat_to_plot, atlas_path, n_roi):
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    # Create empty output image
    new_img = np.zeros(atlas_data.shape)
    
    # Sanity checks
    a_roi = int(np.max(atlas_data))
    if n_roi != a_roi:
        print(f"Mismatch between input ROIs ({n_roi}) and atlas ROIs ({a_roi})")

    # Check if stat_to_plot has the expected length
    if len(stat_to_plot) != n_roi:
        raise ValueError(f"Length of stat_to_plot ({len(stat_to_plot)}) does not match expected n_roi ({n_roi})")

    # Reshape data if needed
    stat_to_plot = np.reshape(stat_to_plot, (n_roi, 1))

    # Assign values to ROIs
    for roi in range(n_roi):
        voxel_indices = np.where(atlas_data == roi + 1)  # 1-based indexing
        if voxel_indices[0].size == 0:
            print(f"ROI {roi+1} not found in atlas.")
        new_img[voxel_indices] = stat_to_plot[roi]

    # Return Nifti image
    img_nii = nib.Nifti1Image(new_img, atlas_img.affine)
    return img_nii

# Create brain maps
for sex in ['female', 'male']:
    for hormone in hormones:
        subset = region_corr_df[(region_corr_df['Sex'] == sex) &
                                (region_corr_df['Hormone'] == hormone)]

        # Initialize array for all ROIs
        roi_values = np.full(n_roi, np.nan)

        # Fill in Spearman's rho for each region (convert Region to 0-based index)
        for _, row in subset.iterrows():
            region_index = int(row['ROI Label']) - 1  
            if 0 <= region_index < n_roi:
                roi_values[region_index] = row["Spearman's rho"]

        # Create image
        img = create_img_for_glassbrain_plot(roi_values, atlas_path, n_roi)

        # Define output filename
        hormone_name_clean = hormone.split('_')[0]
        title = f"Spearman correlation: {sex.capitalize()} – {hormone_name_clean}"
        output_file = os.path.join(brainmap_output_path, f"{sex}_{hormone_name_clean}.png")

        cmap = cm.RdBu_r  # Diverging colormap with blue (negative) and red (positive)
        
        # Plot and save glass brain
        plot_glass_brain(img, threshold=0, vmax=1, vmin=-1,display_mode='lyrz', colorbar=True, cmap = cmap, title=title, plot_abs=False)
        plt.savefig(output_file, bbox_inches='tight',dpi=300)
        plt.close()

        print(f"Saved brain map: {output_file}")