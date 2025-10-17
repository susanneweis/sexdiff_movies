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

# Paths
base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie"
atlas_path = f"{base_path}/data/Susanne_Schaefer_436.nii"

movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

for mv_str in movies:

    tc_comp_res_corr = f"{base_path}/results/compare_time_courses_tt_corr/sep_PCAs/results_sex_movie_ttest_{mv_str}.csv"

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

    # Initialize array for all ROIs
    roi_values = np.full(n_roi, np.nan)

    # Fill in corr (convert Region to 0-based index)
    for _, row in res_tc_corr.iterrows():
        region_index = int(row['ROI_ID']) - 1  
        if 0 <= region_index < n_roi:
            roi_values[region_index] = row["corr"]

    # Create image
    img = create_img_for_glassbrain_plot(roi_values, atlas_path, n_roi)

    # Define output filename
    title = f"Time Course Correlation {mv_str}"
    output_file = os.path.join(brainmap_output_path, f"{mv_str}_time_course_correlation.png")

    cmap = cm.RdBu_r  # Diverging colormap with blue (negative) and red (positive)
                
    # Plot and save glass brain
    plot_glass_brain(img, threshold=0, vmax=1, vmin=-1,display_mode='lyrz', colorbar=True, cmap = cmap, title=title, plot_abs=False)
    plt.savefig(output_file, bbox_inches='tight',dpi=300)
    plt.close()

    print(f"Saved brain map: {output_file}")