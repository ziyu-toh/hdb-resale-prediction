FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY modelling/hyperparam_search/requirements.txt  ${LAMBDA_TASK_ROOT} 

# install dependencies
RUN pip install -r requirements.txt

# Copy function code
COPY modelling/retrain_best_model/retrain_best_model.py ${LAMBDA_TASK_ROOT} 

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "retrain_best_model.lambda_handler" ]