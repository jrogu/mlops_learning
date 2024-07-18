FROM python:3.9-slim

COPY requirements.txt .

RUN apt-get update \
    && pip install --upgrade pip \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2 \
    && pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
