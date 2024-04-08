import pandas as pd

# Load the data
df = pd.read_csv('Data/dataREanonymized_long.csv')
show_nat = True

df_filtered = df[df['site_id'].isin(["Vitality"])]

# Filter the DataFrame to keep only 'door_to_needle' variable
filtered_df = df_filtered[df_filtered['variable'] == 'door_to_needle'].copy()

filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')
filtered_df = filtered_df.dropna(subset=['Value'])

# Calculate the percentage of patients with door to needle time below 60 minutes in each quarter
filtered_df['door_to_needle_below_60'] = filtered_df['Value'] < 45
percentage_below_60 = filtered_df.groupby('YQ')['door_to_needle_below_60'].mean() * 100

percentage_below_60_df = percentage_below_60.reset_index()
percentage_below_60_df.columns = ['YQ', 'Value']

if show_nat == True:
    df_filtered_nat = df[~df['site_id'].isin(["Vitality"])]
    filtered_df_nat = df_filtered_nat[df_filtered_nat['variable'] == 'door_to_needle'].copy()

    filtered_df_nat['Value'] = pd.to_numeric(filtered_df_nat['Value'], errors='coerce')
    filtered_df_nat = filtered_df_nat.dropna(subset=['Value'])

    # Calculate the percentage of patients with door to needle time below 60 minutes in each quarter
    filtered_df_nat['door_to_needle_below_60'] = filtered_df_nat['Value'] < 45
    percentage_below_60_nat = filtered_df_nat.groupby('YQ')['door_to_needle_below_60'].mean() * 100

    percentage_below_60_df_nat = percentage_below_60_nat.reset_index()
    percentage_below_60_df_nat.columns = ['YQ', 'Value']
    #print(percentage_below_60_nat)


# Display the new DataFrame
#print(percentage_below_60_df)


percentage_below_60_df['nat_value'] = percentage_below_60_df_nat['Value']

print(percentage_below_60_df)