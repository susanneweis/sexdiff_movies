import os
import numpy as np
import pandas as pd
from nilearn import datasets, image, masking

def main():
    # Output path
    output_dir = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "schaefer_to_anatomical_overlap.csv")
    
    # Load Schaefer atlas (400 parcels, 17 networks, MNI 2mm)
    print("Loading Schaefer atlas...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17)
    schaefer_img = image.load_img(schaefer.maps)
    schaefer_labels = schaefer['labels']
    
    # schaefer_labels = [label.decode('utf-8') for label in schaefer['labels']]
    
    # Load Harvard-Oxford Cortical and Subcortical atlases
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_subcort = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    ho_cort_img = image.load_img(ho_cort.maps)
    ho_subcort_img = image.load_img(ho_subcort.maps)
    
    # Resample Harvard-Oxford atlases to Schaefer atlas space
    ho_cort_resampled = image.resample_to_img(
        ho_cort_img,
        schaefer_img,
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    ho_subcort_resampled = image.resample_to_img(
        ho_subcort_img,
        schaefer_img,
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )

    ho_cort_data = ho_cort_resampled.get_fdata()
    ho_subcort_data = ho_subcort_resampled.get_fdata()
    schaefer_data = schaefer_img.get_fdata()
    
    results = []
    total_voxels_per_parcel = {}
    
    # Calculate total voxels per Schaefer parcel
    unique_parcels = np.unique(schaefer_data)
    unique_parcels = unique_parcels[unique_parcels != 0] # exclude background
    
    for parcel_val in unique_parcels:
        total_voxels_per_parcel[parcel_val] = np.sum(schaefer_data == parcel_val)
    
    # Loop through each Schaefer parcel
    for i, schaefer_label in enumerate(schaefer_labels):
        parcel_val = i + 1 # parcel indices start at 1 in the atlas
        parcel_mask = (schaefer_data == parcel_val)
        
        # Check overlap with cortical Harvard-Oxford atlas
        unique_ho_cort_vals = np.unique(ho_cort_data[parcel_mask])
        for val in unique_ho_cort_vals:
            if val == 0:
                continue # skip background
            overlap_voxels = np.sum(parcel_mask & (ho_cort_data == val))
            if overlap_voxels > 0:
                anat_label = ho_cort.labels[int(val)]
                results.append({
                    'Schaefer Parcel': schaefer_label,
                    'Anatomical Region': anat_label,
                    'Overlap Voxels': int(overlap_voxels),
                    'Total Parcel Voxels': int(total_voxels_per_parcel[parcel_val]),
                    'Percent Overlap': (overlap_voxels / total_voxels_per_parcel[parcel_val]) * 100
                })
        # Check overlap with subcortical Harvard-Oxford atlas
        unique_ho_subcort_vals = np.unique(ho_subcort_data[parcel_mask])
        for val in unique_ho_subcort_vals:
            if val == 0:
                continue # skip background
            overlap_voxels = np.sum(parcel_mask & (ho_subcort_data == val))
            if overlap_voxels > 0:
                anat_label = ho_subcort.labels[int(val)]
                results.append({
                    'Schaefer Parcel': schaefer_label,
                    'Anatomical Region': anat_label,
                    'Overlap Voxels': int(overlap_voxels),
                    'Total Parcel Voxels': int(total_voxels_per_parcel[parcel_val]),
                    'Percent Overlap': (overlap_voxels / total_voxels_per_parcel[parcel_val]) * 100
                })

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f" Overlap table saved to: '{output_file}'")
    
if __name__ == '__main__':
    main()
