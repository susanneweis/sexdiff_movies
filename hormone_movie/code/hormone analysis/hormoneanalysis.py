import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ks_2samp, mannwhitneyu, spearmanr

# Function to determine the significance level based on p-value
def get_significance(p_value):
    """
    Returns the significance level as asterisks based on p-value
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

# Dictionary containing the measurement units for each hormone
hormone_units = {
    'Estradiol': r'$\mathrm{pg} \cdot \mathrm{mL}^{-1}$',
    'Progesterone': r'$\mathrm{pg} \cdot \mathrm{mL}^{-1}$',
    'Testosterone': r'$\mathrm{pg} \cdot \mathrm{mL}^{-1}$',
    'Cortisol': r'$\mu\mathrm{g} \cdot \mathrm{dL}^{-1}$'
}

# Load hormone data and filter valid participants
def load_and_prepare_data(base_path):
    """
    Loads hormone and participant data from CSV files,
    filters the hormone data to include only validated participants
    """
    hormone_path = f"{base_path}data/Hormone_data.csv"
    complete_path = f"{base_path}data/complete_participants.csv" # Participants which completed all movies
    exclude_path = f"{base_path}results/outlier_results/excluded_subjects.csv" # Participants excluded due to outliers
    
    # Check if files exist
    for path in [hormone_path, complete_path, exclude_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load datasets
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
    
    return hormone_df

# Split hormone data by sex
def split_hormones_by_sex(hormone_df, hormone_columns):
    """
    Splits the hormone values into male and female dataframes,
    stores them into a dictionary for convenient access by hormone and sex.
    """
    matrices = {}
    for col in hormone_columns:
        hormone_name = col.split('_')[0]  # Extract hormone name (e.g. Cortisol)
        df = hormone_df[['PCode', 'sex', col]].copy()
        df.rename(columns={col: hormone_name}, inplace=True)

        # Separate by sex (1 = male, 2 = female)
        matrices[f"{hormone_name}_male"] = df[df['sex'] == 1].reset_index(drop=True)
        matrices[f"{hormone_name}_female"] = df[df['sex'] == 2].reset_index(drop=True)

    return matrices

# Run statistical tests and return structured rows
def run_statistical_tests(hormones, hormone_matrices):
    """
    Runs the Shapiro-Wilk test for normality separately for each sex and 
    the Kolmogorov-Smirnov test for group distribution differences
    """
    results = []

    for hormone in hormones:
        male = hormone_matrices[f'{hormone}_male'][hormone].dropna()
        female = hormone_matrices[f'{hormone}_female'][hormone].dropna()

        # Shapiro-Wilk Normality test for each group
        for sex_val, values, label in [(1, male, 'male'), (2, female, 'female')]:
            stat, p = shapiro(values)
            interpretation = f"Data is {'normally' if p > 0.05 else 'NOT normally'} distributed (p = {p:.4f})"
            significance = get_significance(p)  # Get significance
            results.append({
                'Hormone': hormone, 'Test': 'Shapiro-Wilk Normality',
                'Sex': sex_val, 'Statistic': stat, 'Z-score': '', 'p-value': p,
                'Significance': significance,'Interpretation': interpretation
            })

        # Kolmogorov-Smirnov test between male & female
        ks = ks_2samp(male, female)
        ks_interpretation = ("The distributions differed between both groups, KS p < .05."
                      if ks.pvalue < 0.05 else "No significant distribution difference between groups.")
        significance = get_significance(ks.pvalue)  # Get significance
        results.append({
            'Hormone': hormone, 'Test': 'Kolmogorov-Smirnov',
            'Sex': '', 'Statistic': '', 'Z-score': '', 'p-value': ks.pvalue,
            'Significance': significance,'Interpretation': ks_interpretation
        })
        results.append({}) 
    return results

# Mann-Whitney Test
def run_mannwhitney_tests(hormones, hormone_matrices):
    """
    Compares hormone levels between sexes using the non-parametric 
    Mann-Whitney U test (including z-score and p-value)
    """
    results = []

    for hormone in hormones:
        # Retrieve hormone values for males and females, dropping NaN values
        male = hormone_matrices[f'{hormone}_male'][hormone].dropna()
        female = hormone_matrices[f'{hormone}_female'][hormone].dropna()

        # Perform the Mann-Whitney U test
        u, p = mannwhitneyu(male, female, alternative='two-sided')
        n1, n2 = len(male), len(female) # Sample sizes for male and female
        z = (u - (n1 * n2) / 2) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

        interpretation = (f"There was a significant difference in {hormone} levels "
                   f"between males and females, U = {u:.2f}, Z = {z:.3f}, p < .001."
                   if p < 0.05 else f"No significant difference in {hormone} levels between males and females.")
        significance = get_significance(p)  # Get significan
        results.append({
            'Hormone': hormone, 'Test': 'Mann-Whitney U-Test',
            'Sex': '', 'Statistic': u, 'Z-score': z, 'p-value': p,
            'Significance': significance,'Interpretation': interpretation
        })
        results.append({})  
    return results

# Calculate Cohen's d for Mann-Whitney U Test results
def calculate_cohens_d(u, n1, n2):
    """
    Calculate Cohen's d for Mann-Whitney U test results.
    Cohen's d provides an estimate of effect size
    """
    return (2 * u / (n1 * n2)) - 1

# Spearman Correlations
def run_spearman_correlations(hormones, hormone_matrices):
    """
    Calculates the Spearman correlation between all pairs
    of hormones separately for males and females
    """
    results = []
    for i, h1 in enumerate(hormones):
        for h2 in hormones[i+1:]:
            for sex_val, label in [(1, 'male'), (2, 'female')]:
                df1 = hormone_matrices[f'{h1}_{label}'][h1].dropna()
                df2 = hormone_matrices[f'{h2}_{label}'][h2].dropna()

                # Align lengths by removing rows with NaN in either variable
                df1, df2 = df1.align(df2, join='inner')
                
                # Perform the Spearman correlation test
                rho, p = spearmanr(df1, df2)

                interpretation = (f"Significant correlation between {h1} and {h2} in {label}s, "
                           f"Spearman’s rho = {rho:.3f}, p = {p:.3f}."
                           if p < 0.05 else
                           f"No significant correlation between {h1} and {h2} in {label}s, "
                           f"rho = {rho:.3f}, p = {p:.3f}.")
                
                significance = get_significance(p)  # Get significance
                
                results.append({
                    'Hormone': f"{h1} & {h2}", 'Test': 'Spearman Correlation',
                    'Sex': sex_val, 'Statistic': rho, 'Z-score': '', 'p-value': p,
                    'Significance': significance,'Interpretation': interpretation
                })
            results.append({})
    return results

def generate_hormone_summary_table(hormones, hormone_matrices):
    """
    Generates a table with hormone levels, means, standard deviations, 
    Mann-Whitney U p-value, and effect size (Cohen's d).
    """
    table_data = []
    
    for hormone in hormones:
        # Extract male and female data for the current hormone, removing any NaN values
        male_values = hormone_matrices[f'{hormone}_male'][hormone].dropna()
        female_values = hormone_matrices[f'{hormone}_female'][hormone].dropna()

        # Calculate mean and standard deviation for male and female groups
        male_mean = male_values.mean()
        male_std = male_values.std()
        female_mean = female_values.mean()
        female_std = female_values.std()

        # Perform Mann-Whitney U test for significance between males and females
        u, p = mannwhitneyu(male_values, female_values, alternative='two-sided')

        # Calculate Cohen's d for the effect size
        n1, n2 = len(male_values), len(female_values)  # Sample sizes for male and female
        cohens_d = calculate_cohens_d(u, n1, n2)
        
        unit = hormone_units.get(hormone, '')

        # Format the results in the table
        table_data.append([
            hormone, 
            f"{female_mean:.2f} ± {female_std:.2f}  {unit}",
            f"{male_mean:.2f} ± {male_std:.2f}  {unit}",
            f"{p:.3f}", 
            f"{cohens_d:.2f}" if cohens_d is not None else '-'
        ])
    #Convert table data to DataFrame with the required columns
    hormone_table = pd.DataFrame(table_data, columns=[
        'Hormone', 'Female (Mean ± SD)', 'Male (Mean ± SD)', 'p-value', 'Cohen\'s d'
    ])

     # Ensure that the DataFrame is returned
    if hormone_table is None:
        print("Error: hormone_table is None!")
    else:
        return hormone_table


# Boxplot visualization for hormones
def plot_boxplots(hormone_matrices, hormones, output_dir, mannwhitney_results):
    """
    Creates boxplots to visualize the distribution of hormone levels for male and female participants.
    """
    plt.rcParams['font.family'] = 'Arial' # Set font
    
    for hormone in hormones:
        male_key = f"{hormone}_male"
        female_key = f"{hormone}_female"

        # Check if data exists for male and female participants
        if male_key not in hormone_matrices or female_key not in hormone_matrices:
            print(f"Skipping {hormone} – data missing.")
            continue
        
        # Extract hormone data for males and females, removing any NaN values
        male_data = hormone_matrices[male_key][hormone].dropna()
        female_data = hormone_matrices[female_key][hormone].dropna()

        # Skip hormone if any of the data is empty
        if male_data.empty or female_data.empty:
            print(f"Skipping {hormone} – empty data.")
            continue
        
        # First, filter out empty dictionaries before processing
        filtered_mannwhitney_results = [item for item in mannwhitney_results if item]

        # Search for the relevant hormone result
        mannwhitney_result = next(
            (item for item in filtered_mannwhitney_results if item.get('Hormone') == hormone and item.get('Test') == 'Mann-Whitney U-Test'), 
            None
        )

        # If Mann-Whitney result exists, extract significance (otherwise, default to 'N/A')
        if mannwhitney_result:
            significance = mannwhitney_result.get('Significance', '')
        else:
            significance = 'N/A'
        
        # Create the boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot([male_data, female_data], tick_labels=['Male', 'Female'], patch_artist=True, 
                    boxprops=dict(facecolor='#005293', color='black'), # Facecolor= FZJ blue
                    whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black',linewidth=2),
                    medianprops=dict(color='white', linewidth=3),
                    flierprops=dict(marker='o', markersize=8, markerfacecolor='none',markeredgecolor='black', markeredgewidth=2, linestyle='none'))
        
        plt.ylim(bottom=0)
        
        # If there's significance, annotate the plot with stars
        if significance:
            y_position = max(male_data.max(), female_data.max()) * 0.95 # Position above the boxplots
            x_position = 1.5 # X-position between the two boxplots
        
            # Add the significance stars in the plot
            plt.text(x_position, y_position, significance, 
                     horizontalalignment='center', verticalalignment='bottom', 
                     fontsize=26, fontname='Arial', color='black', fontweight='bold')
        
        # Add labels and title
        unit = hormone_units.get(hormone, '')
        plt.title(f'{hormone}', fontsize=20, fontname='Arial')
        plt.ylabel(f'{hormone} [{unit}]', fontsize=20, fontname='Arial')
        plt.xlabel('Sex', fontsize=20, fontname='Arial')
        plt.xticks(fontsize=20, fontname='Arial')
        plt.yticks(fontsize=20, fontname='Arial')
        
        output_path = os.path.join(output_dir, f"{hormone}_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # Save as PNG file
        plt.close()

# Save the summary table
def save_hormone_summary_table(hormones, hormone_matrices, output_path):
    """
    Generates and saves a summary table of hormone levels, Mann-Whitney U p-values, 
    significance, and effect size to a CSV file.
    """
    hormone_table = generate_hormone_summary_table(hormones, hormone_matrices)
    hormone_table.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Hormone summary table saved to: {output_path}")

# Save results to CSV
def save_results(results, output_path):
    """
    Saves the collected results from all statistical tests to a single CSV file
    """
    df = pd.DataFrame(results)
    
    # Rename 'Sex' column for better clarity in the output file
    df.rename(columns={'Sex': 'Sex (1 = male, 2 = female)'}, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"\n results saved to: {output_path}")

# Save Spearman correlation results in a formatted table
def save_spearman_results(results, output_path):
    """
    Saves the Spearman correlation results to a CSV file in a simplified format
    """
    sex_mapping = {1: 'male', 2: 'female'} # Only show Cohen's d for significant results
    
    male_results = []
    female_results = []
    
    for result in results:
        if result.get('Test') == 'Spearman Correlation':
            sex = sex_mapping.get(result['Sex'])

            simplified_result = {
                'Hormone Pair': result['Hormone'],
                'Sex': sex,
                'Spearman\'s rho': result['Statistic'],
                'p-value': result['p-value']
            }

            if sex == 'male':
                male_results.append(simplified_result)
            elif sex == 'female':
                female_results.append(simplified_result)
    
    # Convert to DataFrame and save
    df_male = pd.DataFrame(male_results)
    df_female = pd.DataFrame(female_results)
   
    # Save to CSV
    male_path = f"{output_path}_male.csv"
    female_path = f"{output_path}_female.csv"

    df_male.to_csv(male_path, index=False)
    df_female.to_csv(female_path, index=False)

    print(f"Spearman Correlation results saved to:\n- {male_path}\n- {female_path}")
    
# Main Execution: Load Data, run all tests and save results
def main():
    base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/hormone_movie/"
    
    # Hormone column names from the dataset
    hormone_columns = [
        'Cortisol_µg/dl_Mean',
        'Estradiol_pg/ml_Mean',
        'Progesterone_pg/ml_Mean',
        'Testosterone_pg/ml_Mean'
    ]
    
    # Define the list of hormone names used for analysis
    hormones = ['Cortisol', 'Estradiol', 'Progesterone', 'Testosterone']

    # Load and process data
    hormone_df = load_and_prepare_data(base_path)
    
    # Split the data into male and female hormone matrices for each hormone
    hormone_matrices = split_hormones_by_sex(hormone_df, hormone_columns)

    # Define the output path for the summary table
    output_file = os.path.join(base_path,"results","hormone_summary_table.csv")
    
    # Save the summary table to CSV
    save_hormone_summary_table(hormones, hormone_matrices, output_file)
    
    # Run all statistical tests and append the results to the all_results list
    all_results = []
    all_results += run_statistical_tests(hormones, hormone_matrices)
    all_results += run_mannwhitney_tests(hormones, hormone_matrices)
    all_results += run_spearman_correlations(hormones, hormone_matrices)

    # Run the Mann-Whitney U tests separately to get the results (for use in boxplots)
    mannwhitney_results = run_mannwhitney_tests(hormones, hormone_matrices)
    
    # Generate and save boxplots for the hormone data (with significance stars)
    boxplot_output_dir = os.path.join(base_path, 'boxplots')
    os.makedirs(boxplot_output_dir, exist_ok=True)
    plot_boxplots(hormone_matrices, hormones, boxplot_output_dir, mannwhitney_results)

    output_spearman_base = os.path.join(base_path, "spearman_results")
    save_spearman_results(all_results, output_spearman_base)

    # Define the output path for saving all the statistical results as CSV
    output_file = os.path.join(base_path, "hormone_statistical_results.csv")
    save_results(all_results, output_file)

# Execute script
if __name__ == "__main__":
    main()
