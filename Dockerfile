FROM python:3
COPY input/testing.tar.gz /app/input/testing.tar.gz
COPY input/training.tar.gz /app/input/training.tar.gz
WORKDIR "/app"
COPY requirements.txt /app/
RUN pip install -r requirements.txt