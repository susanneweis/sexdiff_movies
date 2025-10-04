import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define reference ranges for hormones
def get_hormone_reference_ranges():
    return {
        'Cortisol': ('Cortisol_µg/dl_Mean', (0.116, 0.478)),  # Non-sex-specific
        'Estradiol': ('Estradiol_pg/ml_Mean', {'female': (3.1, 11.9), 'male': (2.1, 4.1)}),
        'Progesterone': ('Progesterone_pg/ml_Mean', {'female': (30.3, 544.3), 'male': (0, 58)}),
        'Testosterone': ('Testosterone_pg/ml_Mean', {'female': (7.1, 42.5), 'male': (28.57, 117.91)})
    }

# SD-based outlier detection function
def detect_sd_outliers(series, num_std_dev=2):
    mean = series.mean()
    std_dev = series.std()
    lower_bound = mean - num_std_dev * std_dev
    upper_bound = mean + num_std_dev * std_dev
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index, lower_bound, upper_bound

# Function to clean invalid characters in filenames
def sanitize_filename(filename):
    return filename.replace("µ", "u").replace("/", "_").replace(" ", "_")

# Function to create and save a boxplot for each hormone
def plot_boxplot_with_data_points(hormone_data, hormone_name, sex_label, reference_range, output_path, sd_outliers_index=None, unit=None):
    plt.figure(figsize=(8, 6))

    # Boxplot
    plt.boxplot(hormone_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))

    # Plot all data points as scatter
    plt.scatter(hormone_data, np.ones_like(hormone_data) * 1, color='red', alpha=0.5, label='Data Points')

    # Highlight the reference range with color
    min_val, max_val = reference_range
    plt.axvspan(min_val, max_val, color='green', alpha=0.2, label='Reference Range')

    # Mark SD outliers in blue
    if sd_outliers_index is not None:
        sd_outliers = hormone_data.loc[sd_outliers_index]
        plt.scatter(sd_outliers, np.ones_like(sd_outliers), color='blue', edgecolor='black', alpha=0.5, label='SD Outliers')

    # Set title and labels with unit included
    plt.title(f'{hormone_name} Boxplot ({sex_label.capitalize()})')
    plt.xlabel(f'Value ({unit})')  # Add the unit to the x-axis label
    plt.yticks([1], [hormone_name])
    plt.legend()

    # Clean filename and save the boxplot
    sanitized_name = sanitize_filename(hormone_name)
    plot_path = os.path.join(output_path, f"{sanitized_name}_{sex_label}_boxplot.png")
    plt.savefig(plot_path)

    # Show the plot in Spyder as well
    plt.show()
    plt.close()

# Function to filter data by sex group and apply SD-based outlier detection
def filter_by_sex_group(df, sex_label, reference_ranges, output_path):
    sex_code = 2 if sex_label == 'female' else 1
    df = df[df['sex'] == sex_code].copy()

    exclusion_map = {}

    for hormone_name, (hormone_column, value) in reference_ranges.items():
        if isinstance(value, dict):
            min_val, max_val = value[sex_label]
        else:
            min_val, max_val = value

        # Determine the unit
        unit = 'µg/dL' if hormone_name == 'Cortisol' else 'pg/ml'

        # Get hormone values for the group and hormone
        hormone_data = df[hormone_column].dropna()

        # Detect SD outliers only
        sd_outliers_index, _, _ = detect_sd_outliers(hormone_data)

        for idx in sd_outliers_index:
            pcode = df.loc[idx, 'PCode']
            value = df.loc[idx, hormone_column]

            # Determine if the value is within the reference range
            in_range = 'Yes' if min_val <= value <= max_val else 'No'

            # Store the outliers and relevant details
            exclusion_map[(pcode, hormone_name)] = {
                'PCode': pcode,
                'Hormone': hormone_name,
                'Value': value,
                'Unit': unit,
                'Method': "SD",
                'Reference Range': f"{min_val} - {max_val} {unit}",
                'In Reference Range': in_range
            }

        # Generate plot for the hormone
        plot_boxplot_with_data_points(
            hormone_data, hormone_name, sex_label,
            (min_val, max_val), output_path,
            sd_outliers_index, unit
        )

    # Return the exclusion data (subject information to be manually confirmed for exclusion)
    exclusion_df = pd.DataFrame(exclusion_map.values())
    exclusion_df = exclusion_df[['PCode', 'Hormone', 'Value', 'Unit', 'Method', 'Reference Range', 'In Reference Range']]  # Rearrange columns
    return df, exclusion_df

# Function to manually decide whether to remove subjects from the excluded list
def manual_exclusion_decision(exclusion_df):
    updated_exclusions = exclusion_df.copy()  # Create a copy of the excluded subjects

    # Loop through each excluded subject to make a decision
    for idx, row in updated_exclusions.iterrows():
        pcode = row['PCode']
        decision = input(f"Subject {pcode} is excluded. Do you want to keep this subject? (y/n): ")
        
        if decision.lower() == 'y':  # If the answer is 'y', remove the subject from the exclusion list
            updated_exclusions = updated_exclusions[updated_exclusions['PCode'] != pcode]

    # Return the updated list of excluded subjects
    return updated_exclusions

# Function to load and process the data
def load_and_filter_data(base_path):
    hormone_df = pd.read_csv(f"{base_path}/data/Hormone_data.csv", sep="\t", encoding="ISO-8859-1")
    complete_df = pd.read_csv(f"{base_path}/data/complete_participants.csv")

    # Keep only subjects present in both datasets
    hormone_df = hormone_df[hormone_df['PCode'].isin(complete_df['subject'])]

    # Retrieve hormone reference ranges
    reference_ranges = get_hormone_reference_ranges()

    # Create output directory if it doesn't exist
    output_path = os.path.join(base_path,"results","outlier_results")
    os.makedirs(output_path, exist_ok=True)

    # Filter and create boxplots for both sexes
    female_df, female_exclusion = filter_by_sex_group(hormone_df, 'female', reference_ranges, output_path)
    male_df, male_exclusion = filter_by_sex_group(hormone_df, 'male', reference_ranges, output_path)

    # Combine excluded subjects from both sexes
    all_exclusions = pd.concat([female_exclusion, male_exclusion]).drop_duplicates()

    # Manual decision for exclusion list xx
    all_exclusions = manual_exclusion_decision(all_exclusions)

    # Save the updated list of excluded subjects after manual confirmation
    all_exclusions.to_csv(os.path.join(output_path, 'excluded_subjects.csv'), index=False)

    # Return filtered data
    filtered_df = pd.concat([female_df, male_df])

    return filtered_df, all_exclusions

# Main function
def main():
    base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/"
    filtered_df, excluded_subjects_df = load_and_filter_data(base_path)

    print(f"Boxplots have been created and saved for all hormones and groups.")

    if not excluded_subjects_df.empty:
        print(f"\nNumber of excluded subjects: {len(excluded_subjects_df)}")
        print(excluded_subjects_df)
    else:
        print("\nNo subjects were excluded.")

if __name__ == "__main__":
    main()
