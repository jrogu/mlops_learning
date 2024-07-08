FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime


COPY requirements.txt .

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2 \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000
#RUN python -m unittest discover -s tests

CMD ["python", "main.py"]
