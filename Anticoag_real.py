import pandas as pd

# Load the data
df = pd.read_csv('Data/dataREanonymized_long.csv')

show_nat = True

df_filtered = df[df['site_id'].isin(["Vitality"])]

# Filter the DataFrame to keep only the variables of interest
filtered_df = df_filtered[df_filtered['variable'].isin(['discharge_warfarin', 'discharge_heparin', 'discharge_dabigatran',
                                      'discharge_rivaroxaban', 'discharge_apixaban', 'discharge_edoxaban', 'discharge_cilostazol',
'discharge_clopidrogel',
'discharge_ticagrelor',
'discharge_ticlopidine',
'discharge_prasugrel',
'discharge_dipyridamol',
])].copy()

# Convert 'Value' column to numeric
filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')

# Replace NaN values with 0
filtered_df['Value'].fillna(0, inplace=True)

# Mark patients with 1 in any of the variables
filtered_df['any_discharge_medication'] = (filtered_df['Value'] == 1).astype(int)

# Count patients with 1 in any discharge medication for each quarter
count_any_discharge_medication = filtered_df.groupby('YQ')['any_discharge_medication'].sum()

# Count total number of patients for each quarter
total_patients = df_filtered.groupby('YQ')['subject_id'].nunique()

# Calculate percentage of patients with 1 in any discharge medication for each quarter
percentage_any_discharge_medication = (count_any_discharge_medication / total_patients) * 100

# Convert percentage_any_discharge_medication Series to a DataFrame
percentage_any_discharge_medication_df = percentage_any_discharge_medication.reset_index()
percentage_any_discharge_medication_df.columns = ['YQ', 'Value']

if show_nat == True:
    df_filtered_nat = df[~df['site_id'].isin(["Vitality"])]

    # Filter the DataFrame to keep only the variables of interest
    filtered_df_nat = df_filtered_nat[
        df_filtered_nat['variable'].isin(['discharge_warfarin', 'discharge_heparin', 'discharge_dabigatran',
                                      'discharge_rivaroxaban', 'discharge_apixaban', 'discharge_edoxaban',
                                      'discharge_cilostazol',
                                      'discharge_clopidrogel',
                                      'discharge_ticagrelor',
                                      'discharge_ticlopidine',
                                      'discharge_prasugrel',
                                      'discharge_dipyridamol',
                                      ])].copy()

    # Convert 'Value' column to numeric
    filtered_df_nat['Value'] = pd.to_numeric(filtered_df_nat['Value'], errors='coerce')

    # Replace NaN values with 0
    filtered_df_nat['Value'].fillna(0, inplace=True)

    # Mark patients with 1 in any of the variables
    filtered_df_nat['any_discharge_medication'] = (filtered_df_nat['Value'] == 1).astype(int)

    # Count patients with 1 in any discharge medication for each quarter
    count_any_discharge_medication_nat = filtered_df_nat.groupby('YQ')['any_discharge_medication'].sum()

    # Count total number of patients for each quarter
    total_patients_nat = df_filtered_nat.groupby('YQ')['subject_id'].nunique()

    # Calculate percentage of patients with 1 in any discharge medication for each quarter
    percentage_any_discharge_medication_nat = (count_any_discharge_medication_nat / total_patients_nat) * 100

    # Convert percentage_any_discharge_medication Series to a DataFrame
    percentage_any_discharge_medication_df_nat = percentage_any_discharge_medication_nat.reset_index()
    percentage_any_discharge_medication_df_nat.columns = ['YQ', 'Value']
    percentage_any_discharge_medication_df['nat_value'] = percentage_any_discharge_medication_df_nat['Value']

# Display the new DataFrame
print(percentage_any_discharge_medication_df)

