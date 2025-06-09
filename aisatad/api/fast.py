from aisatad.params import *
from aisatad.function_files.preprocessor import api_preprocess
from aisatad.function_files.registry import load_model
import pandas as pd

#import for FastAPI
from fastapi import FastAPI


# a fast api
app = FastAPI()
app.state.model = load_model()
app.state.scaler = load_scaler()


@app.post("/predict")
def predict(data: dict) :

    df = pd.DataFrame(data)

    X_scaled = api_preprocess(df, app.state.scaler)

    y_pred = app.state.model.predict(X_scaled)

    return {"anomaly": y_pred}
