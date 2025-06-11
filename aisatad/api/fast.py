from aisatad.params import *
from aisatad.function_files.preprocessor import api_preprocess
from aisatad.function_files.registry import load_model, load_scaler
import pandas as pd

#import for FastAPI
from fastapi import FastAPI

# a fast api
app = FastAPI()
app.state.model = load_model()
app.state.scaler = load_scaler()


@app.post("/predict")
async def predict(json: dict) :

    df = pd.DataFrame(json)
    X_scaled = api_preprocess(df, app.state.scaler)
    y_pred = app.state.model.predict(X_scaled)

    return dict(zip(df["segment"].unique().tolist(), y_pred.tolist()))
