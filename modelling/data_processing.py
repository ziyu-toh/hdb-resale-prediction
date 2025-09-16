import pandas as pd
import boto3


def connect_to_s3():
    # Setting up AWS connection
    s3_input = boto3.resource('s3', region_name='ap-southeast-1')
    input_bucket = s3_input.Bucket('hdb-resale-pred-raw')
    
    return input_bucket

def load_concat_df(input_bucket):
    # Load data
    df_list = []
    for obj in input_bucket.objects.all():
        key = obj.key
        body = obj.get()['Body']

        df = pd.read_csv(body, dtype=str)
        df_list.append(df)
        
    print(len(df_list), "files loaded from S3")

    # Combine dataframes into a single dataframe
    df_all = pd.concat(df_list, ignore_index=True, sort=False)
    
    return df_all 

def convert_variable_type(df_all):
    # Convert variables to appropriate types
    df_all['resale_price'] = pd.to_numeric(df_all['resale_price'], errors='coerce')
    df_all['lease_commence_date'] = pd.to_numeric(df_all['lease_commence_date'], errors='coerce').astype('Int64')
    df_all['floor_area_sqm'] = pd.to_numeric(df_all['floor_area_sqm'], errors='coerce')
    df_all['month'] = pd.to_datetime(df_all['month'], format='%Y-%m')
    
    return df_all

def calculate_time_variables(df_all):
    # Calculate months from earliest month in dataset (Mar-2012), and flat age
    df_all['flat_age_years'] = df_all['month'].dt.year - df_all['lease_commence_date'] # lease_commence_date is actually just the year
    
    earliest_date = pd.to_datetime('2012-03-01')
    df_all['days_from_earliest_data'] = (df_all['month'] - earliest_date).dt.days
    
    return df_all

def clean_flat_model(df_all):
    # Cleaning flat model: Combine Maisonettes
    df_all['flat_model_revised'] = df_all['flat_model'].apply(
        lambda x: 'Maisonette' if 'Maisonette' in str(x) else x
    )
    
    return df_all


def categorise_stories(df_all):
    # group every 15 storeys since a part of the data groups every 5 floors, while another groups every 3 floors
    df_all['storey_max'] = df_all['storey_range'].str.slice(-2, None).astype(int)
    df_all['storey_range_grouped'] = pd.cut(df_all['storey_max'], 
                                            bins=[0, 15, 30, 99], 
                                            labels=["1-15", "16-30", "31+"])

    # Group by storey_range_grouped to check the results
    storey_summary = df_all.groupby('storey_range_grouped')['storey_max'].agg([
        ('min_storey', 'min'),
        ('max_storey', 'max'),
        ('counts', 'count')
    ]).reset_index().sort_values('storey_range_grouped')
    print("Min max storeys in storey group:", storey_summary)

    return df_all


def convert_to_title_case(df_all):
    # Convert to title case for town, flat type and flat model
    df_all['town'] = df_all['town'].str.title()
    df_all['flat_type'] = df_all['flat_type'].str.title()
    df_all['flat_model_revised'] = df_all['flat_model_revised'].str.title()

    return df_all

# TODO: Other factors to KIV: Distance from MRT/amenities, lease years left, supply and demand of surrounding areas

def split_dataset(df_all):
    # Keep relevant columns
    df_output = df_all[[
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
    
    return (train, test, deploy)

def print_missing_counts(df_list):
    df_list_names = ['train', 'test', 'deploy']
    for i, df in enumerate(df_list):
        missing_counts = df.isnull().sum(axis=0)
        print(f"DataFrame {df_list_names[i]} missing value counts:\n{missing_counts}\n")

# Output to AWS S3
def output_to_s3(train, test, deploy):
    s3_output = boto3.resource('s3', region_name='ap-southeast-1')
    output_bucket = s3_output.Bucket('hdb-resale-pred-processed')
    output_bucket.put_object(Key='train.csv', Body=train.to_csv(index=False))
    output_bucket.put_object(Key='test.csv', Body=test.to_csv(index=False))
    output_bucket.put_object(Key='deploy.csv', Body=deploy.to_csv(index=False))

# These should be under "def lambda_handler(event, context)" if using AWS Lambda
if __name__ == "__main__":
    input_bucket = connect_to_s3()
    df_all = load_concat_df(input_bucket)
    df_all = convert_variable_type(df_all)
    df_all = calculate_time_variables(df_all)
    df_all = clean_flat_model(df_all)
    df_all = categorise_stories(df_all)
    df_all = convert_to_title_case(df_all)
    train, test, deploy = split_dataset(df_all)
    print_missing_counts([train, test, deploy])
    output_to_s3(train, test, deploy)

    print("Data processing complete.")