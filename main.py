import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Data/dataREanonymized.csv', low_memory=False)
qi_info_df = pd.read_csv('Data/QI_info.csv')


site_id_mapping = {
    'exzoqdsmqbfszgx': 'Riverside',
    'hrqlztlyambobmw': 'Horizon',
    'tehhzyosktscide': 'Evergreen',
    'gapqdomfiblqzyi': 'Summit',
    'zeuvygzylbumoir': 'Pineview',
    'mijcujmylwkrtvs': 'WellSpring',
    'uorpmrdlnunhhfz': 'Vitality',
    'zkelthufwochhtd': 'Alpha',
    'vlagwtastpawvgj': 'Aurora'
}

site_country_mapping = {
    'vrprkigsxydwgni': 'Novaria',
    'ejzryutdacxnhgl': 'Celestria',
    'mpjtenmiflfkbid': 'Solara'
}

df['site_id'] = df['site_id'].map(site_id_mapping)
df['site_country'] = df['site_country'].map(site_country_mapping)

df.drop('site_name', axis=1, inplace=True)

# Specify the columns to maintain
id_vars = ['subject_id', 'site_id', 'discharge_quarter', 'discharge_year', "site_country"]

# Specify the columns to melt
value_vars = [col for col in df.columns if col not in id_vars]

# Reshape the DataFrame into long format
long_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')


# Merge the two DataFrames based on the "variable" column in long_df and the "COLUMN" column in qi_info_df
merged_df = pd.merge(long_df, qi_info_df, left_on='variable', right_on='COLUMN', how='inner')


# Select the desired columns
desired_columns = ['TAB', 'INDICATOR', 'ATTRIBUTE_TYPE', 'SUMMARIZE_BY']

# Keep only the desired columns in the merged DataFrame
final_df = merged_df[desired_columns + ['subject_id', 'site_id', 'site_country', 'discharge_quarter', 'discharge_year', 'variable', 'value']]

final_df = final_df[final_df['discharge_year'] >= 2018]

final_df['YQ'] = final_df['discharge_year'].astype(str) + ' ' + final_df['discharge_quarter']

unique_variables = final_df['variable'].unique()
formatted_variables = ', '.join([f'"{variable}"' for variable in unique_variables])
print("Formatted Variables:")
print(formatted_variables)

final_df = final_df.rename(columns={'value': 'Value'})
final_df = final_df.replace({'TRUE': 1, 'FALSE': 0})
final_df = final_df.replace({'True': 1, 'False': 0})
final_df = final_df.replace({True: 1, False: 0})
final_df = final_df.replace({'yes': 1, 'no': 0})

# Save the long format DataFrame to a new CSV file
final_df.to_csv('Data/dataREanonymized_long.csv', index=False)