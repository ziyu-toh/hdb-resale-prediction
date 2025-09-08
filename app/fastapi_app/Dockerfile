FROM python:3.12

WORKDIR /app

# Copy from specified file path (in docker build -t) -> to target file path
# Use relative paths on the docker side so can align file path with pytest
COPY /app/fastapi_app/requirements.txt ./app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./app/requirements.txt

# Copy from specified file path (in docker build -t) -> to target file path
# Do not save as fastapi as it will lead to module not found error
COPY /app/fastapi_app ./app/fastapi_app

# Exec form: CMD ["executable","param1","param2"]
# Make sure to always use the exec form to ensure that FastAPI can shutdown gracefully and lifespan events are triggered.
CMD ["fastapi", "run", "./app/fastapi_app/main.py", "--port", "80"]

# To build container, use docker compose up from parent directory