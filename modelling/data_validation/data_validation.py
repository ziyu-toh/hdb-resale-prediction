import pandas as pd
import boto3

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

# Check if distribution in numerical values are not too different

# Check if categorical values in test set are present in train set
def check_categorical_values(train_df, test_df):
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        train_unique = set(train_df[col].unique())
        test_unique = set(test_df[col].unique())
        if not test_unique.issubset(train_unique):
            print(f"Warning: Test set contains unseen categories in column '{col}'")
            unseen_categories = test_unique - train_unique
            print(f"Unseen categories: {unseen_categories}")
        else:
            print(f"All categories in column '{col}' are present in the training set.")

# Check if distribution in categorical values are not too different

if __name__ == "__main__":
    input_bucket = connect_to_s3()
    train_df, test_df = load_data_from_s3(input_bucket)
    check_categorical_values(train_df, test_df)

