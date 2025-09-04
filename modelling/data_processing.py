import pandas as pd
import boto3

# Setting up AWS connection
s3_input = boto3.resource('s3', region_name='ap-southeast-1')
input_bucket = s3_input.Bucket('hdb-resale-pred-raw')

# Load data
df_list = []
for obj in input_bucket.objects.all():
    key = obj.key
    body = obj.get()['Body']

    df = pd.read_csv(body, dtype=str)
    df_list.append(df)
    
print(len(df_list), "files loaded from S3")

# Combine dataframes into a single dataframe
df_combined_col = pd.concat(df_list, ignore_index=True, sort=False)

# Convert variables to appropriate types
df_combined_col['resale_price'] = pd.to_numeric(df_combined_col['resale_price'], errors='coerce')
df_combined_col['lease_commence_date'] = pd.to_numeric(df_combined_col['lease_commence_date'], errors='coerce').astype('Int64')
df_combined_col['floor_area_sqm'] = pd.to_numeric(df_combined_col['floor_area_sqm'], errors='coerce')
df_combined_col['month'] = pd.to_datetime(df_combined_col['month'], format='%Y-%m')

# Calculate months from earliest month in dataset (Mar-2012)
df_combined_col['flat_age_years'] = df_combined_col['month'].dt.year - df_combined_col['lease_commence_date']
earliest_date = pd.to_datetime('2012-03-01')
df_combined_col['days_from_earliest_data'] = (df_combined_col['month'] - earliest_date).dt.days

# Drop remaining_lease column
df_combined_col = df_combined_col.drop('remaining_lease', axis=1, errors='ignore')

# Cleaning flat model: Combine Maisonettes
df_combined_col['flat_model_revised'] = df_combined_col['flat_model'].apply(
    lambda x: 'Maisonette' if 'Maisonette' in str(x) else x
)

# Check unique values
print(df_combined_col['flat_model_revised'].unique())

# Create min max value for stories, then group every 15 storeys since a part of the data groups every 5 floors instead of 3 floors
df_combined_col['storey_max'] = df_combined_col['storey_range'].str.slice(-2, None).astype(int)

def categorize_storey(storey_max):
    if storey_max <= 15:
        return "1-15"
    elif 16 <= storey_max <= 30:
        return "16-30"
    else:
        return "31+"

df_combined_col['storey_range_grouped'] = df_combined_col['storey_max'].apply(categorize_storey)

# Group by storey_range_grouped to check the results
storey_summary = df_combined_col.groupby('storey_range_grouped')['storey_max'].agg([
    ('min_storey', 'min'),
    ('max_storey', 'max'),
    ('counts', 'count')
]).reset_index().sort_values('storey_range_grouped')
print(storey_summary)

# Convert to title case for town, flat type and flat model
df_combined_col['town'] = df_combined_col['town'].str.title()
df_combined_col['flat_type'] = df_combined_col['flat_type'].str.title()
df_combined_col['flat_model_revised'] = df_combined_col['flat_model_revised'].str.title()

# TODO: Other factors to KIV: Distance from MRT/amenities, lease years left, supply and demand of surrounding areas

# Keep relevant columns
df_output = df_combined_col[[
    'month',
    'town',
    'flat_type',
    'flat_model_revised',
    'flat_age_years',
    'floor_area_sqm',
    'days_from_earliest_data',
    'storey_range_grouped',
    'resale_price'
]]

# Split into train (2012-2023), test(2024) and deploy sets (2025)
train = df_output[df_output['month'].dt.year.between(2012, 2023)].drop('month', axis=1)
test = df_output[df_output['month'].dt.year == 2024].drop('month', axis=1)
deploy = df_output[df_output['month'].dt.year > 2024].drop('month', axis=1)

# Output to AWS S3
s3_output = boto3.resource('s3', region_name='ap-southeast-1')
output_bucket = s3_output.Bucket('hdb-resale-pred-processed')
output_bucket.put_object(Key='train.csv', Body=train.to_csv(index=False))
output_bucket.put_object(Key='test.csv', Body=test.to_csv(index=False))
output_bucket.put_object(Key='deploy.csv', Body=deploy.to_csv(index=False))

print("Data processing complete.")