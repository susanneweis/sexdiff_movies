import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm

# Assign unique numerical IDs to each region name and adds them as a new column
def assign_roi_ids(df):
    unique_regions = df['region'].drop_duplicates().tolist()
    region_to_id = {region: i+1 for i, region in enumerate(unique_regions)}  
    df['ROI_ID'] = df['region'].map(region_to_id)
    return df, region_to_id

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

def fill_glassbrain(n_r,res_df,column):
    # Initialize array for all ROIs
    roi_values = np.full(n_r, np.nan)

    # Fill in corr (convert Region to 0-based index)
    for _, row in res_df.iterrows():
        region_index = int(row['ROI_ID']) - 1  
        if 0 <= region_index < n_r:
            roi_values[region_index] = row[column]
    return roi_values

def create_glassbrains(vals, at_path, nrois, title_str,o_file,min,max):
    
     # Create image
    img = create_img_for_glassbrain_plot(vals, at_path, nrois)

    # Define output filename

    cmap = cm.RdBu_r  # Diverging colormap with blue (negative) and red (positive)
                
    # Plot and save glass brain
    plot_glass_brain(img, threshold=0, vmax=max, vmin=min,display_mode='lyrz', colorbar=True, cmap = cmap, title=title_str, plot_abs=False)
    plt.savefig(o_file, bbox_inches='tight',dpi=300)
    plt.close()
    
    print(f"Saved brain map: {o_file}")


def main(base_path,nn_mi):

    results_path = f"{base_path}/results_run_sTOPF"
    data_path = f"{base_path}/data_run_sTOPF"

    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    
    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "rest_run-1", "rest_run-2"]

    sex_mapping = {1: 'male', 2: 'female'}

    subs_sex = pd.read_csv(f"{data_path}/Participant_sex_info.csv", sep = ";")
    subs_sex['gender'] = subs_sex['gender'].replace(sex_mapping)

    for mv_str in movies:
        
        outpath = f"{results_path}/glass_brains_nn{nn_mi}/individual_expressions/{mv_str}"
        os.makedirs(outpath, exist_ok=True)

        #subjects = ind_brain["subject"].astype(str).drop_duplicates().tolist()

        for subj in subs_sex["subject_ID"]:

            ind_brain_path = f"{results_path}/individual_expressions/individual_expression_{subj}.csv"
            
            if os.path.exists(ind_brain_path):

                # Load data

                ind_brain = pd.read_csv(ind_brain_path)

                sub_brain = ind_brain.loc[ind_brain["movie"] == mv_str, ["region", "correlation_female", "correlation_male","fem_mi"]].reset_index(drop=True)

                diff = np.arctanh(sub_brain["correlation_female"]) - np.arctanh(sub_brain["correlation_male"])
                sub_brain["fem-mal"] = np.tanh(diff)

                # change to proper comparisons of correlations
                mean = np.arctanh(sub_brain["fem-mal"]).mean()
                fem_mal_score = np.tanh(mean)

                sub_sex = subs_sex.loc[subs_sex["subject_ID"] == subj, "gender"].iloc[0]
                sub_brain, region_to_id_f = assign_roi_ids(sub_brain)

                ##### Brain maps Femaleness

                n_roi = sub_brain["region"].nunique()

                roi_values = fill_glassbrain(n_roi,sub_brain,"fem-mal")

                # Define output filename
                title = f"Femaleness {mv_str} {subj} {sub_sex}. Score: {fem_mal_score:.2f}"
                output_file = f"{outpath}/{mv_str}_ind_expression{mv_str}_{subj}.png"
                # output_file = os.path.join(outpath, f"{mv_str}_ind_expression{mv_str}_{subj}.png")

                create_glassbrains(roi_values, atlas_path, n_roi, title,output_file,-1,1)

                ##### Brain maps MI

                n_roi = sub_brain["region"].nunique()

                roi_values = fill_glassbrain(n_roi,sub_brain,"fem_mi")

                mean_fem_mi = sub_brain["fem_mi"].mean()
                # Define output filename
                title = f"Female MI {mv_str} {subj} {sub_sex}. Score: {mean_fem_mi:.2f}"
                output_file = f"{outpath}/{mv_str}_ind_expression_mi_{mv_str}_{subj}.png"
                # output_file = os.path.join(outpath, f"{mv_str}_ind_expression{mv_str}_{subj}.png")

                min_val = sub_brain["fem_mi"].min()
                max_val = sub_brain["fem_mi"].max()

                create_glassbrains(roi_values, atlas_path, n_roi, title,output_file,min_val,max_val)




# Execute script
if __name__ == "__main__":
    main()
