FROM python:3.12.1


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python -m unittest discover -s tests

CMD ["python", "main.py"]
