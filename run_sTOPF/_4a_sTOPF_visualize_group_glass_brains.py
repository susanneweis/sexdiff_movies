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

def create_glassbrains(vals, at_path, nrois, title_str,o_file, min, max):
    
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
    
    outpath = f"{results_path}/glass_brains_nn{nn_mi}/full_group_PCAs"
    os.makedirs(outpath, exist_ok=True)

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "rest_run-1", "rest_run-2"]

    for mv_str in movies:

        tc_corr = f"{results_path}/compare_time_courses_nn{nn_mi}/results_compare_time_courses_{mv_str}.csv"

        # Load datasets
        res_tc_corr = pd.read_csv(tc_corr)

        res_tc_corr, region_to_id_f = assign_roi_ids(res_tc_corr)

        ##### Brain maps

        # Correlations 
        n_roi = res_tc_corr["region"].nunique()

        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr")

        # Define output filename
        title = f"Female vs. Male Time Course Correlations {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_tc_correlations.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, -1, 1)

        # High Correlations
        
        res_tc_corr["corr_high"] = res_tc_corr["corr"].where(res_tc_corr["corr"] > 0.9, 0)
        
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr_high")
        title = f"Female vs. Male Time Course Correlations > 0.9 {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_high_corr.png")
        
        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, 0, 1)

        # Low Correlations
        
        res_tc_corr["corr_low"] = (res_tc_corr["corr"] < 0.1).astype(int)
    
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr_low")
        title = f"Female vs. Male Time Course Correlations < 0.1 {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_low_corr.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, 0, 1)

        # Mutual Information
            
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"mutual_inf")
        title = f"Female vs. Male Time Course Mutual Information {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_mi.png")

        min_val = res_tc_corr["mutual_inf"].min()
        max_val = res_tc_corr["mutual_inf"].max()

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, min_val, max_val)

        # Low Mutual Information
        
        low = res_tc_corr["mutual_inf"].quantile(0.10)
        res_tc_corr["mi_low"] = (res_tc_corr["mutual_inf"] < low).astype(int)
    
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"mi_low")
        title = f"Female vs. Male Time Course Low Mutual Information {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_mi_low.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, 0, 1)

        # High Mutual Information
        
        high = res_tc_corr["mutual_inf"].quantile(0.90)
        res_tc_corr["mi_high"] = (res_tc_corr["mutual_inf"] > high).astype(int)
    
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"mi_high")
        title = f"Female vs. Male Time Course High Mutual Information {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_mi_high.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file, 0, 1)


# Execute script
if __name__ == "__main__":
    main()
