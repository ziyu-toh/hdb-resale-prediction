import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import boto3

# Defining MLFlow server URI
TRACKING_URI = "http://127.0.0.1:8080/" # Use this link to access MLFlow logs
EXPT_NAME = "HDB Resale Price - Elastic Net Regression"
RUN_NAME = "Baseline"
RANDOM_STATE = 42
K_FOLDS = 2
VAR_NAMES = yaml.safe_load(open("config.yaml"))['data_feature_names']

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
train_df = pd.read_csv("data/processed/train.csv")[VAR_NAMES]
test_df = pd.read_csv("data/processed/test.csv")[VAR_NAMES]

# Define allowable values for flat model, flat type, and town
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# create sklearn pipeline with pre-processing of numerical and categorical features
def define_pipeline(df):
    """Create a machine learning pipeline with preprocessing and model"""
    numeric_features = df.drop("resale_price", axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
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
def train_model(pipeline, df):
    """Train the model using cross-validation"""
    print("Training model")
    X = df.drop(columns=['resale_price'])
    y = df['resale_price']

    grid_search = GridSearchCV(pipeline, 
                          cv=TimeSeriesSplit(n_splits=K_FOLDS), 
                          param_grid=params, 
                          scoring='neg_root_mean_squared_error',
                          refit=True,
                          verbose=1, 
                          n_jobs=-1,)

    grid_search.fit(X, y)
    
    # Train set 
    trained_pipeline = grid_search.fit(X, y)
    
    return trained_pipeline

# Log feature importance on best model
def get_feature_importance(trained_grid_search, test_df):
    """Get feature importance using permutation importance"""
    best_pipeline = trained_grid_search.best_estimator_
    r = permutation_importance(best_pipeline, test_df.drop(columns=['resale_price']), test_df['resale_price'],
                                n_repeats=10,
                                random_state=RANDOM_STATE)

    # Soring permutation importance results
    sorted_idx = r.importances_mean.argsort()[::-1]
    importance_df = pd.DataFrame({
        'feature': best_pipeline[:-1].get_feature_names_out()[sorted_idx],
        'importance_mean': r.importances_mean[sorted_idx],
        'importance_std': r.importances_std[sorted_idx]
    })

    print("Feature importances:\n", importance_df)
    return importance_df

# Plot
def plot_feature_importance(importance_df):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.barplot(x="importance_mean", y="feature", ax=ax,
                data=importance_df, palette="viridis", 
               # xerr=importance_df["importance_std"].to_list()
                )
    ax.set_title("Feature Importances", fontsize=14)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.close(fig)
    
    return fig

# define mlflow experiment and trigger autologging
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPT_NAME)

# Training model with the pipeline
with mlflow.start_run(run_name=RUN_NAME, nested=True) as run:
    print("Training pipeline") 
    mlflow.sklearn.autolog(max_tuning_runs=None, silent=True) # Need to run this before training
    trained_grid_search = train_model(pipeline=define_pipeline(train_df), df=train_df)
     
    # Plot feature importances
    importance_df = get_feature_importance(trained_grid_search, test_df)
    fi_plot = plot_feature_importance(importance_df)
    mlflow.log_figure(fi_plot, "feature_importance.png")
     
    # Calculate test score
    result = mlflow.models.evaluate(
        model="runs:/{}/model".format(mlflow.active_run().info.run_id),
        data=test_df,
        targets="resale_price",
        model_type="regressor",
        evaluators=["default"]
    )   
    

