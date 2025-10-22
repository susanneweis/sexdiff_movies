import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

def main():
    # Define file paths
    base_path = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies"
    hormone_path = f"{base_path}/data_pipeline/Hormone_data.csv"
    results_path = f"{base_path}/results_pipeline"
    complete_path = f"{results_path}/complete_participants.csv"
    exclude_path = f"{results_path}/excluded_subjects.csv"
    demographic_path = f"{results_path}/demographics_table.csv"
    hist_path = f"{results_path}/cycle_day_histogram_females.png"

    # Load datasets
    hormone_df = pd.read_csv(hormone_path, sep="\t", encoding="windows-1252")
    complete_df = pd.read_csv(complete_path)

    # Exclude for now, becomes relevant when hormones are included
    # exclude_df = pd.read_csv(exclude_path)

    # Clean column names
    hormone_df.rename(columns={
        'hÃ¶chster Bildungsstand': 'highest education',
        'Nur fÃ¼r Frauen: an welchem Tag des Zyklus befinden Sie sich (gezÃ¤hlt ab dem ersten Tag der Regel)?': 'cycle_day'
    }, inplace=True)

    # Filter valid subjects
    valid_subjects = complete_df['subject'].unique()
    
    # Exclude for now, becomes relevant when hormones are included
    # excluded_subjects = exclude_df['PCode'].unique()
    excluded_subjects = []
    valid_subjects = [subj for subj in valid_subjects if subj not in excluded_subjects]
    hormone_df = hormone_df[hormone_df['PCode'].isin(valid_subjects)]

    # Standardize education and sex
    education_mapping = {
        'Abitur': 'High School',
        'Realschule': 'Secondary School',
        'Hauptschule': 'Lower Secondary',
        'Studium': 'Bachelor\'s',
        'Bachelor': 'Bachelor\'s',
        'Master': 'Master\'s',
        'Promotion': 'PhD'
    }
    hormone_df['highest education'] = hormone_df['highest education'].replace(education_mapping)

    sex_mapping = {1: 'male', 2: 'female'}
    hormone_df['sex'] = hormone_df['sex'].replace(sex_mapping)

    # Generate demographic summary
    summary_rows = []

    for sex in hormone_df['sex'].unique():
        df_sex = hormone_df[hormone_df['sex'] == sex]
        count = len(df_sex)
        mean_age = round(df_sex['age'].mean())  # Round to nearest whole number
        std_age = round(df_sex['age'].std())  # Round to nearest whole number
        min_age = round(df_sex['age'].min())  # Round to nearest whole number
        max_age = round(df_sex['age'].max())  # Round to nearest whole number
        edu_counts = df_sex['highest education'].value_counts().to_dict()
        edu_str = ', '.join([f"{k} ({v})" for k, v in edu_counts.items()])
        
        summary_rows.append({
            'Sex': sex,
            'N': count,
            'Age (Mean ± SD)': f"{mean_age} ± {std_age}",
            'Age Range': f"{min_age}–{max_age}",
            'Highest Education (n)': edu_str
        })

    # Add total summary row
    df_total = hormone_df
    count = len(df_total)
    mean_age = round(df_total['age'].mean())  
    std_age = round(df_total['age'].std())  
    min_age = round(df_total['age'].min())  
    max_age = round(df_total['age'].max())  
    edu_counts = df_total['highest education'].value_counts().to_dict()
    edu_str = ', '.join([f"{k} ({v})" for k, v in edu_counts.items()])

    summary_rows.append({
        'Sex': 'Total',
        'N': count,
        'Age (Mean ± SD)': f"{mean_age} ± {std_age}",
        'Age Range': f"{min_age}–{max_age}",
        'Highest Education (n)': edu_str
    })

    # Save demographic summary to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(demographic_path, index=False, encoding='utf-8-sig')

    # Analyze and plot cycle day data

    # Filter valid female participants with numeric cycle day
    female_df = hormone_df[hormone_df['sex'] == 'female'].copy()
    female_df['cycle_day'] = pd.to_numeric(female_df['cycle_day'], errors='coerce')
    female_df = female_df.dropna(subset=['cycle_day'])

    total_female = hormone_df[hormone_df['sex'] == 'female'].shape[0]
    valid_cycle_days = female_df.shape[0]
    excluded_count = total_female - valid_cycle_days

    # Rename column to 'HC_use' for clarity and easier referencing
    hormone_df.rename(columns={
        'Nur fÃ¼r Frauen: nehmen Sie die Pille (oder andere HormonprÃ¤parate)?': 'HC_use'
    }, inplace=True)

    # Filter dataset for female participants and ensure 'cycle_day' is numeric
    female_df = hormone_df[hormone_df['sex'] == 'female'].copy()
    female_df['cycle_day'] = pd.to_numeric(female_df['cycle_day'], errors='coerce')
    female_df = female_df.dropna(subset=['cycle_day'])

    # Subdivide female participants into naturally cycling and HC users
    naturally_cycling = female_df[female_df['HC_use'].str.lower() == 'nein']
    HC_users = female_df[female_df['HC_use'].str.lower() == 'ja']

    # Plot histogram
    plt.figure(figsize=(8, 3))
    plt.hist(
        [naturally_cycling['cycle_day'], HC_users['cycle_day']],
        bins=np.arange(0.5, 28.5 + 1, 1),  # Center bins on integers 1–32
        color=['#005293', '#80C1EA'], # FZJ blue
        edgecolor='black',
        label=['Naturally Cycling Women', 'HC Users'],
        stacked='True',
        align='mid'
    )
    plt.title('Menstrual Cycle Day Distribution', fontname='Arial')
    plt.xlabel('Cycle Day [d]', fontname='Arial')
    plt.ylabel('Number of Participants [ - ]', fontname='Arial')
    plt.xticks(range(1,29 ), fontname='Arial')
    plt.grid(axis='y', linestyle='--', alpha=1)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend(title='')
    plt.tight_layout()

    # Save histogram
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # Print summary info
    print(f"Cycle day histogram saved to: {hist_path}")
    print(f"Demographic summary table saved to: {demographic_path}\n")
    print(f"- all female participants: {total_female}")
    print(f"- female participants with cycle_day: {valid_cycle_days}")
    print(f"- excluded: {excluded_count}")

    print("\nIncluded female participants with valid cycle_day values:")
    print(female_df[['PCode', 'cycle_day']])

# Execute script
if __name__ == "__main__":
    main()