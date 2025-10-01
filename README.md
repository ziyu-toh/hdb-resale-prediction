# HDB Resale Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project was my take on an end-to-end machine learning solution to predict HDB apartment resale prices in Singapore. Continuous training is also implemented to keep the model updated with the data, when necessary. 

The data is obtained from [data.gov](https://data.gov.sg/collections/189/view), where the latest data is updated on a daily basis. Only data from March-2012 will be used, since the definition of the date column is consistent from then. Features used at the moment can be broadly categorised as characteristics of house apartment, location, related to lease of house, and resale price. 

## Main structure of solution
1. Data processing: Written as AWS Lambda functions (given the right execution and resource permissions), to pull data from data.gov, process the data afterwards, with any storage required using AWS S3
    -  Location: modelling/data_pulling, modelling/data_processing
    -  Sequence of events: Eventbridge trigger > data_pulling lambda > Dump into S3 for raw data >  data_processing lambda > Dump into S3 for processed data
    -  CI/CD: .github/workflows/cicd_data_processing.yml
3. Hyperparameter tuning: Python function which utilises CML in tandem with GitHub actions. Execute function using Github Actions when pushing
    -  Location: modelling/hyperparam_search
    -  CI/CD: .github/workflows/hyperparam_search.yml
5. Retraining:
    -  Location: modelling/retrain_best_model
6. Webapp (FastAPI + Streamlit):
-  Location: app

Overview of tech stack used: Python, AWS (S3, Lambda, Eventbridge, ECS), Docker, Github Actions, CML

## To-Do
- Reinforce CI/CD workflows and tests.
- Add data/concept drift thresholds on top of 5% performance difference for continuous training script.
- Engineer more features and include feature selection mechanisms
- Implement logging
