import pandas as pd

# Load the data
df = pd.read_csv('Data/dataREanonymized_long.csv')
show_nat = True

df_filtered = df[df['site_id'].isin(["Vitality"])]
# Filter the DataFrame to keep only 'dysphagia_screening_type' variable
filtered_df = df_filtered[df_filtered['variable'] == 'dysphagia_screening_type'].copy()

# Count non-null values for each quarter
non_null_counts = filtered_df.groupby('YQ')['Value'].count()

# Count total number of patients for each quarter
total_patients = df_filtered.groupby('YQ')['subject_id'].nunique()

# Calculate percentage of patients with non-null values for each quarter
percentage_filled_in = (non_null_counts / total_patients) * 100

# Convert percentage_filled_in Series to a DataFrame
percentage_filled_in_df = percentage_filled_in.reset_index()
percentage_filled_in_df.columns = ['YQ', 'Value']


if show_nat == True:
    df_filtered_nat = df[~df['site_id'].isin(["Vitality"])]
    # Filter the DataFrame to keep only 'dysphagia_screening_type' variable
    filtered_df_nat = df_filtered_nat[df_filtered_nat['variable'] == 'dysphagia_screening_type'].copy()

    # Count non-null values for each quarter
    non_null_counts_nat = filtered_df_nat.groupby('YQ')['Value'].count()

    # Count total number of patients for each quarter
    total_patients_nat = df_filtered_nat.groupby('YQ')['subject_id'].nunique()

    # Calculate percentage of patients with non-null values for each quarter
    percentage_filled_in_nat = (non_null_counts_nat / total_patients_nat) * 100

    # Convert percentage_filled_in Series to a DataFrame
    percentage_filled_in_df_nat = percentage_filled_in_nat.reset_index()
    percentage_filled_in_df_nat.columns = ['YQ', 'Value']

# Display the new DataFrame
percentage_filled_in_df['nat_value'] = percentage_filled_in_df_nat['Value']
print(percentage_filled_in_df)

# Save the new DataFrame to a CSV fil