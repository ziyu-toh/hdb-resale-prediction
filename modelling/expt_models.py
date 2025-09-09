import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import yaml
import boto3

# Defining MLFlow server URI
TRACKING_URI = "http://127.0.0.1:8080/" # Use this link to access MLFlow logs
EXPT_NAME = "HDB Resale Price"
RUN_NAME = "Elastic Net Regression"
RANDOM_STATE = 42

# Define the model hyperparameters
params = {
    "reg__l1_ratio": [0.5],
    "reg__random_state": [RANDOM_STATE],
}

# # Setting up AWS connection
# s3_input = boto3.resource('s3', region_name='ap-southeast-1')
# input_bucket = s3_input.Bucket('hdb-resale-pred-processed')

# # Load data
# train_df = pd.read_csv(input_bucket.Object('train.csv').get()['Body'])

# Use local training file instead while touching up on modelling
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Define allowable values for flat model, flat type, and town
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# create sklearn pipeline with pre-processing of numerical and categorical features
def define_pipeline(train_df):
    """Create a machine learning pipeline with preprocessing and model"""
    numeric_features = train_df.drop("resale_price", axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('reg', ElasticNet())
    ])
    
    return pipeline


# train model
def train_model(pipeline, train_df):
    """Train the model using cross-validation"""
    print("Training model")
    X = train_df.drop(columns=['resale_price'])
    y = train_df['resale_price']
    
    grid_search = GridSearchCV(pipeline, 
                          cv=TimeSeriesSplit(n_splits=5), 
                          param_grid=params, 
                          scoring='neg_root_mean_squared_error',
                          verbose=1, 
                          n_jobs=-1,)
    
    
    # Train set 
    trained_pipeline = grid_search.fit(X, y)
    
    return trained_pipeline

# define mlflow experiment and trigger autologging
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPT_NAME)

# Training model with the pipeline
with mlflow.start_run(run_name=RUN_NAME) as run:
    print("Training pipeline") 
    mlflow.sklearn.autolog() # Need to run this before training
    trained_pipeline = train_model(pipeline=define_pipeline(train_df), train_df=train_df)
    
    # Calculate test score
    result = mlflow.models.evaluate(
        model="runs:/{}/model".format(mlflow.active_run().info.run_id),
        data=test_df,
        targets="resale_price",
        model_type="regressor",
        evaluators=["default"],
    )
    

mlflow.end_run()

