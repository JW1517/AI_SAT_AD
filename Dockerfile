FROM python:3.12.9-buster


COPY AI_SAT_AD /aisatad
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn aisatad.api.fast:app --host 0.0.0.0 --port $PORT
