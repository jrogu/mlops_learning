FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel


COPY requirements.txt .
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

RUN pip install -r requirements.txt

COPY . .

#RUN python -m unittest discover -s tests

CMD ["python", "main.py"]
