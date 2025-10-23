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


def main():

    base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
    atlas_path = f"{base_path}/data_pipeline/Susanne_Schaefer_436.nii"
    results_path = f"{base_path}/results_pipeline"
    
  

    movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]

    for mv_str in movies:
        
        outpath = f"{results_path}/glass_brains/individual_expressions/{mv_str}"
        os.makedirs(outpath, exist_ok=True)

        ind_brain_path = f"{results_path}/results_individual_exp/indiviudal_exp_{mv_str}.csv"

        # Load data
        ind_brain = pd.read_csv(ind_brain_path)

        subjects = ind_brain["subject"].astype(str).drop_duplicates().tolist()

        # the following is worng - need to look through subjects and extract df for each subject
        sub_brain = {
            s: ind_brain.loc[ind_brain["subject"] == s, ["region", "correlation_female", "correlation_male"]].reset_index(drop=True)
            for s in subjects
        }

        #res_tc_corr, region_to_id_f = assign_roi_ids(res_tc_corr)

        ind_brain, region_to_id_f = assign_roi_ids(ind_brain)

        ##### Brain maps

        # Correlations 
        n_roi = res_tc_corr["region"].nunique()

        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr")

        # Define output filename
        title = f"Female vs. Male Time Course Correlations {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_tc_correlations.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

        # High Correlations
        
        res_tc_corr["corr_high"] = res_tc_corr["corr"].where(res_tc_corr["corr"] > 0.9, 0)
        
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr_high")
        title = f"Female vs. Male Time Course Correlations > 0.9 {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_high_corr.png")
        
        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

        # Low Correlations
        
        res_tc_corr["corr_low"] = (res_tc_corr["corr"] < 0.1).astype(int)
    
        roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr_low")
        title = f"Female vs. Male Time Course Correlations < 0.1 {mv_str}"
        output_file = os.path.join(outpath, f"{mv_str}_low_corr.png")

        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    #     # second one 
    

    #     res_tc_corr["non_sig"] = (~res_tc_corr["corr_sig"]).astype(int)
    #     roi_values = fill_glassbrain(n_roi,res_tc_corr,"non_sig")

    #     title = f"Non sig Time Course Correlation {mv_str}"
    #     output_file = os.path.join(brainmap_output_path, f"{mv_str}_non_sig_time_course_correlation.png")

    #     create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    #     # third one 
    # # there might be an error here

    #     comp_load = pd.read_csv(comp_loadings)
    #     comp_load.rename(columns={"Region": "region"}, inplace=True)
    #     comp_load, region_to_id_f = assign_roi_ids(comp_load)

    #     comp_load["sig_p"] = (comp_load["p_val"] < 0.05).astype(int)
    #     comp_load["sig_p_for_similar"] = np.where(res_tc_corr["corr_sig"], comp_load["sig_p"], 0)

    #     roi_values = fill_glassbrain(n_roi,comp_load,"sig_p_for_similar")

    #     title = f"Sig load diff for similar tc {mv_str}"
    #     output_file = os.path.join(brainmap_output_path, f"{mv_str}_similar_tc_diff_load.png")

    #     create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)

    #     # fourth one 

    #     comp_load["load_diff"] = comp_load["mean_female"] - comp_load["mean_male"]
    #     comp_load["load_diff_for_similar"] = np.where(res_tc_corr["corr_sig"], comp_load["load_diff"], 0)

    #     roi_values = fill_glassbrain(n_roi,comp_load,"load_diff_for_similar")
        
    #     title = f"Load Difference {mv_str}"
    #     output_file = os.path.join(brainmap_output_path, f"{mv_str}_load_difference.png")

    #     create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)


    #     #sixth one
        
    #     res_tc_corr["corr_neg"] = (res_tc_corr["corr"] < 0).astype(int)
        
    #     roi_values = fill_glassbrain(n_roi,res_tc_corr,"corr_neg")

    #     title = f" Correlation < 0 {mv_str}"
    #     output_file = os.path.join(brainmap_output_path, f"{mv_str}_corr_neg.png")

    #     create_glassbrains(roi_values, atlas_path, n_roi, title,output_file)




# Execute script
if __name__ == "__main__":
    main()
