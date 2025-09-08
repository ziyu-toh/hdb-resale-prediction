FROM python:3.12

WORKDIR /app

# Copy from specified file path (in docker build -t) -> to target file path
COPY /app/streamlit/requirements.txt /app/requirements.txt

# Copy yaml file as well
COPY /config.yaml /app/config.yaml

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy from specified file path (in docker build -t) -> to target file path
COPY /app/streamlit /app/streamlit

# informs Docker that the container listens on the specified network ports at runtime. Your container needs to listen to Streamlit’s (default) port 8501:
EXPOSE 8501

# HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Exec form: CMD ["executable","param1","param2"]
# Make sure to always use the exec form to ensure that FastAPI can shutdown gracefully and lifespan events are triggered.
CMD ["streamlit", "run", "streamlit/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# To build container, use docker compose up from parent directory