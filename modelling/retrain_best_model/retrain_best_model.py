import pandas as pd
import boto3
import joblib
from sklearn.metrics import root_mean_squared_error
from copy import deepcopy

def lambda_handler(event, context):
    print("Starting retraining process...")
    # Load train, test, retraining test data  and model from S3 - Save to tmp then load from there
    s3 = boto3.client('s3', region_name='ap-southeast-1')
    s3.download_file('hdb-resale-pred-processed', 'train.csv', '/tmp/train.csv')
    s3.download_file('hdb-resale-pred-processed', 'test.csv', '/tmp/test.csv')
    s3.download_file('hdb-resale-pred-processed', 'retrain_test.csv', '/tmp/retrain_test.csv')
    s3.download_file('hdb-resale-best-model', 'champion_model.joblib', '/tmp/champion_model.joblib')

    # Combine train and test data --> New training data
    df_train = pd.read_csv('/tmp/train.csv')
    df_test = pd.read_csv('/tmp/test.csv')
    df_combined = pd.concat([df_train, df_test], axis=0)
    df_retrain_test = pd.read_csv('/tmp/retrain_test.csv')
    champ_model = joblib.load('/tmp/champion_model.joblib')

    # Retrain best model on combined data
    challenger_model = deepcopy(champ_model)
    challenger_model.fit(df_combined.drop(columns=['resale_price']), df_combined['resale_price'])

    # Test on retraining test data
    challenger_pred = challenger_model.predict(df_retrain_test.drop(columns=['resale_price']))
    champ_pred = champ_model.predict(df_retrain_test.drop(columns=['resale_price']))

    # Check if have 5% difference in RMSE
    challenger_rmse = root_mean_squared_error(df_retrain_test['resale_price'], challenger_pred)
    champion_rmse = root_mean_squared_error(df_retrain_test['resale_price'], champ_pred)

    print("Challenger RMSE: ", challenger_rmse)
    print("Champion RMSE: ", champion_rmse)

    # if challenger_rmse < champion_rmse * 0.95:
    #     print("Challenger model is better with score of: ", challenger_rmse, " vs ", champion_rmse)
    #     # Save challenger model and metrics to S3
    #     joblib.dump(challenger_model, '/tmp/challenger_model.joblib')
    #     s3.upload_file("/tmp/challenger_model.joblib", 
    #                   'hdb-resale-best-model', 
    #                   'champion_model.joblib')
    #     print("Challenger model uploaded successfully.")

    # else:
    #     print("Champion model is better with score of: ", champion_rmse, " vs ", challenger_rmse)