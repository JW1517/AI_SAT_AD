import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from aisatad.params import *
from aisatad.function_files.data import get_data_with_cache #clean_data, load_data_to_bq
# from aisatad.function_files.model import initialize_model, compile_model, train_model, evaluate_model
# from aisatad.function_files.preprocessing import preprocess_features
# from mlflow.registry import load_model, save_model, save_results
# from mlflow.registry import mlflow_run, mlflow_transition_model


# data():

# def preprocessing():
#load data





#def preprocess() -> None:

    # # Process data
    # data_clean = clean_data(data_query)

    # X = data_clean.drop("fare_amount", axis=1)
    # y = data_clean[["fare_amount"]]

    # X_processed = preprocess_features(X)

    # # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # # using data.load_data_to_bq()
    # data_processed_with_timestamp = pd.DataFrame(np.concatenate((
    #     data_clean[["pickup_datetime"]],
    #     X_processed,
    #     y,
    # ), axis=1))

    # load_data_to_bq(
    #     data_processed_with_timestamp,
    #     gcp_project=GCP_PROJECT,
    #     bq_dataset=BQ_DATASET,
    #     table=f'processed_{DATA_SIZE}',
    #     truncate=True
    # )

print("✅ preprocess() done \n")


# def model():
    #1. init()
    #2. train()
    #3. evaluate()
    #4. pred()

if __name__ == '__main__':
    #data()
    #preprocessing()
    #model()
    print(BQ_DATASET)
    print(LOCAL_DATA_PATH)
    data_bucket_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{BQ_DATASET}.csv")
    print(data_bucket_cache_path)
    df = get_data_with_cache(cache_path=data_bucket_cache_path)
    pass
