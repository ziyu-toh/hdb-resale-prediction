FROM python:3.12

WORKDIR /app

# Copy from specified file path (in docker build -t) -> to target file path
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy from specified file path (in docker build -t) -> to target file path
COPY . /app/fastapi

# Exec form: CMD ["executable","param1","param2"]
# Make sure to always use the exec form to ensure that FastAPI can shutdown gracefully and lifespan events are triggered.
CMD ["fastapi", "run", "fastapi/main.py", "--proxy-headers", "--port", "80"]

# To build the container: docker build -t fastapi_image ./app/fastapi
# To run the container: docker run -d --name fastapi_container -p 80:80 fastapi_image