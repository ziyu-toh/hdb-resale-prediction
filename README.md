# HDB Resale Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project was my take on an end-to-end machine learning solution to predict HDB apartment resale prices in Singapore. Continuous training is also implemented to keep the model updated with the data, when necessary. 

The data is obtained from [data.gov](https://data.gov.sg/collections/189/view), where data is updated on a daily basis. Only data from March-2012 will be used, since the definition of the "month" variable is consistent from then. Variables used at the moment can be broadly categorised as type of flat, characteristics of flat, location, lease details, and resale price. 

## Main structure of solution
1. Data processing: Pulls data from data.gov, process the data afterwards, storing datasets using AWS S3
    -  Location: modelling/data_pulling, modelling/data_processing
    -  Usage: No manual action required. Eventbridge triggers every 7 days activating data_pulling Lambda > Dump into S3 for raw data > triggers data_processing Lambda > Dump into S3 for processed data
    -  CI/CD: .github/workflows/cicd_data_processing.yml
    -  Notes: Written as AWS Lambda functions (given the right execution and resource permissions). Lambdas were created on AWS, adding on the default pandas layer provided by AWS. CI/CD workflow pushes the Python scripts to AWS Lambda. 

2. Retraining: Performs retraining every time there is new data
    -  Location: modelling/retrain_best_model
    -  Usage: No manual action required. Dumping processed data into S3 triggers the retraining Lambda function.
    -  CI/CD: .github/workflows/cicd_retraining.yml
    -  Notes: Lambda function which will be Dockerised and pushed to private AWS ECR upon CI/CD.

3. Webapp (FastAPI for prediction + Streamlit): Respecively, they form the backend and frontend of the prediction model webapp. 
    -  Location: app
    -  Usage: Spin up the AWS ECS to deploy the prediction model when needed (AWS resources required for this step costs money!)
    -  CI/CD: .github/workflows/cicd_apps.yml
    -  Notes: Streamlit requests prediction from FastAPI via localhost, since they are in the same task in the ECS. Both microservices are automatically Dockerised and pushed to Docker Hub using the CI/CD workflow. AWS ECS reads from Docker Hub for changes. 

4. Hyperparameter tuning: Python function which utilises CML in tandem with GitHub actions, to perform hyperparameter tuning for just one model type, while specifying  hyperparameter values to try. 
    -  Location: modelling/hyperparam_search
    -  Usage: Adjust hyperparam_search.py to select model of interest and the respective hyperparameters > Push to GitHub > GitHub Actions workflow runs automatically > Compare results on GitHub > Set "OUTPUT_BEST_MODEL" parameter to True when retraining the best model with the best hyperparameters identified.
    -  CI/CD: .github/workflows/hyperparam_search.yml
    -  Notes: This step exists independently of the previous 3 components. 

Overview of tech stack used: Python 3.12, AWS (S3, Lambda, Eventbridge, ECS, ECR), Docker, Github Actions, CML

## To-Do
- Engineer more features and include feature selection mechanisms
- Implement logging
