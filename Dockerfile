FROM python:3.12-bookworm

COPY aisatad /aisatad
COPY requirements.txt /requirements.txt
COPY raw_data /raw_data
COPY training_outputs /training_outputs


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn aisatad.api.fast:app --host 0.0.0.0 --port $PORT
