from aisatad.params import *
from aisatad.function_files.preprocessor import api_preprocess
from aisatad.function_files.registry import load_model, load_scaler
import pandas as pd
from io import StringIO

#import for FastAPI
from fastapi import FastAPI, UploadFile, File


# a fast api
app = FastAPI()
app.state.model = load_model()
app.state.scaler = load_scaler()


@app.post("/predict")
async def predict(json: dict) :

    df = pd.DataFrame(json)

    X_scaled = api_preprocess(df, app.state.scaler)

    #y_pred = app.state.model.predict(X_scaled)

    return {"anomaly": 0}


