import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_parquet('data/processed/train.parquet')
test_df = pd.read_parquet('data/processed/test.parquet')

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}


# create sklearn pipeline with pre-processing of numerical and categorical features
def pipeline(train_df):
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
        ('model', LinearRegression())
    ])
    return pipeline


# train model
def train_model(pipeline, train_df):
    """Train the model using cross-validation"""
    print("Training model")
    X = train_df.drop(columns=['resale_price'])
    y = train_df['resale_price']
    
    kf = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error',
                             verbose=1, n_jobs=-1)

    print(f"Cross-validated RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")
    
    pipeline.fit(X, y)
    
    return pipeline

# define mlflow experiment and trigger autologging
mlflow.set_tracking_uri("http://127.0.0.1:8080/")
mlflow.set_experiment("HDB Resale Price")
print("Run 'mlflow server --host 127.0.0.1 --port 8080' in one terminal") 

# Training model with the pipeline
with mlflow.start_run() as run:
    print("Training pipeline") 
    mlflow.sklearn.autolog() # Need to run this before training
    trained_pipeline = train_model(pipeline=pipeline(train_df), train_df=train_df)

mlflow.end_run()

# By right still need to implement final test on test set, but skipping for now
print("Access mlflow dashboard at http://127.0.0.1:8080 after that")
