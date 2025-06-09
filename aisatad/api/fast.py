from aisatad.params import *
from aisatad.function_files.preprocessor import api_preprocess
from aisatad.function_files.registry import load_model,load_scaler
import pandas as pd

#import for FastAPI
from fastapi import FastAPI, File, UploadFile


# a fast api
app = FastAPI()

model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    raise RuntimeError("❌ Model or scaler could not be loaded. Check registry.")


app.state.model = load_model()
app.state.scaler = load_scaler()


# @app.get("/predict")
# def predict(data: dict) :

#     df = pd.DataFrame(data)

#     X_scaled = api_preprocess(df, app.state.scaler)

#     y_pred = app.state.model.predict(X_scaled)

#     return {"anomaly": y_pred}

@app.get("/health")
def health_check():
    if app.state.model is None:
        return {"status": "❌ No model loaded"}
    return {"status": "✅ API is healthy"}




@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        file.file.close()
        #return {"filename": file.filename}

        X_scaled = api_preprocess(df, app.state.scaler)

        y_pred = app.state.model.predict(X_scaled)

        return {"anomaly": y_pred.tolist()}
    except Exception as e:
        return {"error": str(e)}



    #return {"filename": file.filename}
