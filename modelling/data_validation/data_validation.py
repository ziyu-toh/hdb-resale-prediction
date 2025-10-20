import pandas as pd
import boto3
import numpy as np

def connect_to_s3():
    # Setting up AWS connection
    s3_input = boto3.resource('s3', region_name='ap-southeast-1')
    input_bucket = s3_input.Bucket('hdb-resale-pred-processed')
    
    return input_bucket

# Load train and test data from s3
def load_data_from_s3(input_bucket):
    print("Loading training data...")
    train_df = pd.read_csv(input_bucket.Object('train.csv').get()['Body'])
    test_df = pd.read_csv(input_bucket.Object('test.csv').get()['Body'])
    
    return train_df, test_df

# Check if distribution in numerical values are not too different using PSI
def calculate_psi(expected_array, actual_array, buckets=10):
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected_array, breakpoints)

    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
    print(actual_percents)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

    return psi_value

def check_numerical_distribution(train_df, test_df):
    print("Checking numerical distribution differences using PSI...")
    numerical_cols = ["flat_age_years", "floor_area_sqm"]
    trigger_training = 0
    for col in numerical_cols:
        psi_value = calculate_psi(train_df[col], test_df[col])
        if psi_value > 0.2:
            print(f"Warning: Significant distribution difference in column '{col}' with PSI value: {psi_value}")
            trigger_training += 1
        else:
            print(f"Column '{col}' has acceptable distribution difference with PSI value: {psi_value}")
    
    return trigger_training

# Check if categorical values in test set are present in train set
def check_categorical_values(train_df, test_df):
    print("Checking categorical values in test set against training set...")
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    trigger_training = 0
    for col in categorical_cols:
        train_unique = set(train_df[col].unique())
        test_unique = set(test_df[col].unique())
        if not test_unique.issubset(train_unique):
            print(f"Warning: Test set contains unseen categories in column '{col}'")
            unseen_categories = test_unique - train_unique
            print(f"Unseen categories: {unseen_categories}")
            trigger_training += 1
        else:
            print(f"All categories in column '{col}' are present in the training set.")
    
    return trigger_training

def lambda_handler(event, context):
    print("Starting data validation process...")
    input_bucket = connect_to_s3()
    train_df, test_df = load_data_from_s3(input_bucket)
    trigger_training_cat = check_categorical_values(train_df, test_df)
    trigger_training_num = check_numerical_distribution(train_df, test_df)
    if trigger_training_cat + trigger_training_num > 0:
        print("At least one data validation check failed. Triggering model retraining.")
        lambda_client = boto3.client('lambda', region_name='ap-southeast-1')
        response = lambda_client.invoke(
            FunctionName='hdb-resale-retraining',
            InvocationType='Event'  # Asynchronous invocation bc no need to wait for response
        )
        print("Retraining Lambda response:", response)
        
    print("Data validation process completed.")

if __name__ == "__main__":
    lambda_handler(None, None)