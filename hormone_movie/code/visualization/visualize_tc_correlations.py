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

def create_glassbrains(vals, at_path, nrois, title_str,o_file):
    
     # Create image
    img = create_img_for_glassbrain_plot(vals, at_path, nrois)

    # Define output filename

    cmap = cm.RdBu_r  # Diverging colormap with blue (negative) and red (positive)
                
    # Plot and save glass brain
    plot_glass_brain(img, threshold=0, vmax=1, vmin=-1,display_mode='lyrz', colorbar=True, cmap = cmap, title=title_str, plot_abs=False)
    plt.savefig(o_file, bbox_inches='tight',dpi=300)
    plt.close()
    
    print(f"Saved brain map: {o_file}")

# Paths
base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie"
atlas_path = f"{base_path}/data/Susanne_Schaefer_436.nii"

movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

for mv_str in movies:

    tc_comp_res_corr = f"{base_path}/results/compare_time_courses_tt_corr/sep_PCAs/results_sex_movie_ttest_{mv_str}.csv"
    comp_loadings = f"{base_path}/results/compare_loadings/sep_PCAs/results_comp_l_{mv_str}.csv"

    # Check if files exist
    for path in [atlas_path, tc_comp_res_corr]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load datasets
    res_tc_corr = pd.read_csv(tc_comp_res_corr)

    res_tc_corr, region_to_id_f = assign_roi_ids(res_tc_corr)

    ##### Brain maps
    n_roi = 436

    # Create output directory for brain maps
    brainmap_output_path = os.path.join(base_path,"results","compare_tc_corr","brain_maps")
    os.makedirs(brainmap_output_path, exist_ok=True)

    roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr")

    # Define output filename
    title = f"Time Course Correlation {mv_str}"
    output_file = os.path.join(brainmap_output_path, f"{mv_str}_time_course_correlation.png")

    create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    # second one 

    res_tc_corr["non_sig"] = (~res_tc_corr["corr_sig"]).astype(int)
    roi_values = fill_glassbrain(n_roi,res_tc_corr,"non_sig")

    title = f"Non sig Time Course Correlation {mv_str}"
    output_file = os.path.join(brainmap_output_path, f"{mv_str}_non_sig_time_course_correlation.png")

    create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    # third one 

    comp_load = pd.read_csv(comp_loadings)
    comp_load.rename(columns={"Region": "region"}, inplace=True)
    comp_load, region_to_id_f = assign_roi_ids(comp_load)

    comp_load["sig_p"] = (comp_load["p_val"] < 0.05).astype(int)
    comp_load["sig_p_for_similar"] = np.where(res_tc_corr["corr_sig"], comp_load["sig_p"], 0)

    roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr")

    title = f"Correlations {mv_str}"
    output_file = os.path.join(brainmap_output_path, f"{mv_str}_correlations.png")

    create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    # fourth one 

    comp_load["load_diff"] = comp_load["mean_female"] - comp_load["mean_male"]
    comp_load["load_diff_for_similar"] = np.where(res_tc_corr["corr_sig"], comp_load["load_diff"], 0)

    roi_values = fill_glassbrain(n_roi,comp_load,"load_diff_for_similar")
    
    title = f"Load Difference {mv_str}"
    output_file = os.path.join(brainmap_output_path, f"{mv_str}_load_difference.png")

    create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)
