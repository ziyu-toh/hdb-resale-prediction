import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import boto3


# Initialisation
OUTPUT_BEST_MODEL = False
MODEL_NAME = "RandomForestRegressor"
RANDOM_STATE = 42
K_FOLDS = 5
MODEL = RandomForestRegressor()
PARAM_DICT = {
    "reg__n_estimators": [50, 100, 250],
    "reg__max_features": [0.25, 0.5, 0.75],
    "reg__max_depth": [5, 10, 25],
    "reg__min_samples_split": [10, 20],
    "reg__min_samples_leaf": [5, 10],
    "reg__criterion": ["squared_error"],
    "reg__n_jobs": [-1],
    
    "reg__random_state": [RANDOM_STATE],
} # model hyperparameters

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
        ('reg', MODEL)
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
                    param_grid=PARAM_DICT, 
                    scoring='neg_root_mean_squared_error',
                    refit=True,
                    verbose=2, 
                    n_jobs=-1,
                    )

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
        'feature': trained_grid_search.feature_names_in_[sorted_idx],
        'importance_mean': r.importances_mean[sorted_idx],
        'importance_std': r.importances_std[sorted_idx]
    })
    
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
    plt.savefig("fi_plot.png")
    plt.close(fig)
    
    return fig

if __name__ == "__main__":
    # Setting up AWS connection
    print("Setting up AWS connection...")
    s3_input = boto3.resource('s3', region_name='ap-southeast-1')
    input_bucket = s3_input.Bucket('hdb-resale-pred-processed')

    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(input_bucket.Object('train.csv').get()['Body'])
    test_df = pd.read_csv(input_bucket.Object('test.csv').get()['Body'])

    # Training model with the pipeline
    print("Training model with the pipeline...")
    trained_grid_search = train_model(pipeline=define_pipeline(train_df), df=train_df)
    cv_results = trained_grid_search.cv_results_

    # Plot feature importances
    print("Plotting feature importances...")
    fi_plot = plot_feature_importance(get_feature_importance(trained_grid_search, test_df))
        
    # Calculate test score
    print("Calculating test score...")
    test_score = trained_grid_search.score(test_df.drop(columns=['resale_price']), test_df['resale_price'])

    # Output for CML
    with open("train_metrics.txt", "w") as outfile:
        outfile.write(str(round(trained_grid_search.best_score_, 3)) + "\n")
    
    with open("test_metrics.txt", "w") as outfile:
        outfile.write(str(round(-test_score, 3)) + "\n")
        
    with open("hyperparams_tried.txt", "w") as outfile:
        outfile.write(str(PARAM_DICT) + "\n")

    with open("best_hyperparams.txt", "w") as outfile:
        outfile.write(str(trained_grid_search.best_params_) + "\n")

    if OUTPUT_BEST_MODEL:
        # Output champion model
        print("Saving champion model...")
        joblib.dump(trained_grid_search.best_estimator_, '/tmp/champion_model.joblib')

        # Output best model to s3
        s3_output = boto3.client('s3')

        # Upload best model to destination bucket
        print("Uploading best model to destination bucket...")
        s3_output.upload_file("/tmp/champion_model.joblib", 
                              'hdb-resale-best-model', 
                              'champion_model.joblib')

        print("Best model uploaded successfully.")

