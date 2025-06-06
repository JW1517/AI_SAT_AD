from aisatad.params import *
from aisatad.function_files.preprocessor import preprocess, gen_data_scaled
from aisatad.function_files.registry import load_model
import pandas as pd

#import for FastAPI
from fastapi import FastAPI


# a fast api
app = FastAPI()
app.state.model = load_model()
app.state.scaler = load_scaler()


@app.get("/predict")
def predict(dico):

    df = pd.DataFrame(dico)

    df_preproc = gen_data_scaled(df, app.state.scaler)

    y_pred = app.state.model.predict(df_preproc)

    return {"anomaly": y_pred}


    # X_pred = pd.DataFrame({
    #     "pickup_datetime": [pd.Timestamp(pickup_datetime, tz='UTC')],
    #     "pickup_longitude": [pickup_longitude],
    #     "pickup_latitude": [pickup_latitude],
    #     "dropoff_longitude": [dropoff_longitude],
    #     "dropoff_latitude": [dropoff_latitude],
    #     "passenger_count": [passenger_count]
    # })

    # X_pred_preprocessed = preprocess_features(X_pred)

    # result = app.state.model.predict(X_pred_preprocessed)

    # return {"fare": float(result[0][0])}

