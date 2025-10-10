import pandas as pd
import os

# Define paths
base_path = "/Users/kbauer/Desktop/master thesis/codes/fMRIdata/"
hormone_data_path = "/Users/kbauer/Desktop/master thesis/Dataset_movie_project_completeness.csv"
phenotype_path = f"{base_path}/movies_phenotype_results.csv"

# Load hormone and phenotype data
hormone_df = pd.read_csv(hormone_data_path, sep=";")
phenotypes = pd.read_csv(phenotype_path, sep=";")

# Ensure subject IDs are in uppercase for consistency
hormone_df["PCode"] = hormone_df["PCode"].str.upper()
phenotypes["subject_ID"] = phenotypes["subject_ID"].str.upper()

# Define movie timepoint parameters
movies = {
    "dd": {"min_timepoint": 6, "max_timepoint": 463},
    "s": {"min_timepoint": 6, "max_timepoint": 445},
    "dps": {"min_timepoint": 6, "max_timepoint": 479},
    "fg": {"min_timepoint": 6, "max_timepoint": 591},
    "dmw": {"min_timepoint": 6, "max_timepoint": 522},
    "lib": {"min_timepoint": 6, "max_timepoint": 454},
    "tgtbtu": {"min_timepoint": 6, "max_timepoint": 512}
}

# Manually exclude subjects
manual_exclusions = {"P000099", "P000119"}

# Initialize sets to track valid and excluded subjects
excluded_subjects = {}
complete_subjects = set()

# Function to check the validity of a subject for a given movie
def check_subject_validity(subject, df, movie_params):
    valid_subject = True
    subject_data = df[df["subject"] == subject]

    # Check if required timepoints are available
    available_timepoints = set(subject_data["timepoint"])
    required_timepoints = set(range(movie_params["min_timepoint"], movie_params["max_timepoint"] + 1))
    if not required_timepoints.issubset(available_timepoints):
        missing_tps = required_timepoints - available_timepoints
        excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
        excluded_subjects[subject]["movies"].add(movie)
        excluded_subjects[subject]["reasons"].add(f"Missing timepoints: {sorted(missing_tps)}")
        valid_subject = False

    # Check for missing ROI data
    if subject_data.iloc[:, 2:].isna().any().any():  # Check if any ROI data is NaN
        excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
        excluded_subjects[subject]["movies"].add(movie)
        excluded_subjects[subject]["reasons"].add("Missing data in brain regions")
        valid_subject = False

    # Check if subject is in hormone dataset
    if subject not in hormone_df["PCode"].values:
        excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
        excluded_subjects[subject]["movies"].add(movie)
        excluded_subjects[subject]["reasons"].add("Not found in hormone dataset")
        valid_subject = False

    # Check if subject is in phenotype dataset
    if subject not in phenotypes["subject_ID"].values:
        excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
        excluded_subjects[subject]["movies"].add(movie)
        excluded_subjects[subject]["reasons"].add("Not found in phenotype dataset")
        valid_subject = False

    return valid_subject

# Process each movie dataset
for movie, params in movies.items():
    print(f"\n--- Checking Movie: {movie} ---")
    bold_path = f"{base_path}BOLD_Schaefer400_subcor36_mean_task-{movie}_MOVIES_INM7.csv"

    if not os.path.exists(bold_path):
        print(f"File not found: {bold_path}")
        continue

    try:
        df = pd.read_csv(bold_path, usecols=[1, 2] + list(range(3, 439)))
    except Exception as e:
        print(f"Error reading {bold_path}: {e}")
        continue

    expected_rois = 436
    actual_rois = df.shape[1] - 2  # First two columns are "subject" and "timepoint"
    if actual_rois != expected_rois:
        print(f"Warning: {movie} contains {actual_rois} ROIs instead of {expected_rois}.")

    valid_subjects = set()
    required_timepoints = set(range(params["min_timepoint"], params["max_timepoint"] + 1))
    actual_subjects = set(df["subject"].unique())

    # First check if the subject is in the BOLD dataset, then exclude immediately if not
    missing_from_bold = (set(hormone_df["PCode"]) - manual_exclusions) - actual_subjects
    for subject in missing_from_bold:
        excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
        excluded_subjects[subject]["movies"].add(movie)
        excluded_subjects[subject]["reasons"].add("Not in BOLD dataset")

    # Check valid subjects and exclude if necessary
    for subject in actual_subjects:
        if check_subject_validity(subject, df, params):
            valid_subjects.add(subject)

    complete_subjects.update(valid_subjects)

# Manually exclude subjects
for subject in manual_exclusions:
    excluded_subjects.setdefault(subject, {"movies": set(), "reasons": set()})
    excluded_subjects[subject]["movies"].update(movies.keys())
    excluded_subjects[subject]["reasons"].add("Manually excluded")
    if subject in complete_subjects:
        complete_subjects.remove(subject)

# REMOVE EXCLUDED SUBJECTS FROM THE COMPLETE LIST
# Ensure no excluded subject is in the complete_subjects list
complete_subjects = complete_subjects - set(excluded_subjects.keys())

# Save valid participants (those in complete_subjects)
complete_path = f"{base_path}complete_participants.csv"
pd.DataFrame(list(complete_subjects), columns=["subject"]).to_csv(complete_path, index=False)
print(f"\n List of valid participants saved to: {complete_path}")

# Save exclusion log (those who were excluded)
excluded_data = []
for subject, data in excluded_subjects.items():
    excluded_data.append([
        subject,
        ", ".join(sorted(data["movies"])),
        ", ".join(sorted(data["reasons"]))
    ])
exclusion_df = pd.DataFrame(excluded_data, columns=["subject", "movies", "reason"])
exclusion_log_path = f"{base_path}excluded_participants_log.csv"
exclusion_df.to_csv(exclusion_log_path, index=False)
print(f" Exclusion log saved to: {exclusion_log_path}")
