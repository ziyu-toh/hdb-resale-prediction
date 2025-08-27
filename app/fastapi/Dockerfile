FROM python:3.12

WORKDIR /code

# Copy from specified file path (in docker build -t) -> to target file path
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy from specified file path (in docker build -t) -> to target file path
COPY . /code/fastapi

# Exec form: CMD ["executable","param1","param2"]
# Make sure to always use the exec form to ensure that FastAPI can shutdown gracefully and lifespan events are triggered.
CMD ["fastapi", "run", "fastapi/main.py", "--proxy-headers", "--port", "80"]