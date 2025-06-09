# imports
from aisatad.function_files.preprocessor import *
from aisatad.function_files.model import *
from aisatad.params import *
from google.cloud import storage

import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras

# import mlflow
# from mlflow.tracking import MlflowClient

def save_scaler(scaler) -> None:
    """
    Persist trained scaler locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/scalers/{timestamp}.pickle"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "scalers/{timestamp}.pickle"
    - if MODEL_TARGET='mlflow', also persist it on MLflow (not implemented here)
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save scaler locally
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, "scalers", f"{timestamp}.pickle")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    print("✅ Scaler saved locally")

    if MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nUpload scaler to GCS..." + Style.RESET_ALL)

        scaler_filename = scaler_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"scalers/{scaler_filename}")
        blob.upload_from_filename(scaler_path)

        print("✅ Scaler saved to GCS")

        return None

    return None


def load_scaler() -> object:
    """
    Return a saved scaler:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    - or from MLflow if MODEL_TARGET=='mlflow'

    Return None (but do not Raise) if no scaler is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest scaler from local registry..." + Style.RESET_ALL)

        local_scaler_directory = os.path.join(LOCAL_REGISTRY_PATH, "scalers")
        local_scaler_paths = glob.glob(f"{local_scaler_directory}/*")

        if not local_scaler_paths:
            print("❌ No scaler found locally")
            return None

        most_recent_scaler_path_on_disk = sorted(local_scaler_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest scaler from disk..." + Style.RESET_ALL)

        with open(most_recent_scaler_path_on_disk, "rb") as file:
            scaler = pickle.load(file)

        print("✅ Scaler loaded from local disk")

        return scaler

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest scaler from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="scalers/"))

        if not blobs:
            print(f"\n❌ No scaler found in GCS bucket {BUCKET_NAME}")
            return None

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_scaler_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(latest_scaler_path_to_save), exist_ok=True)

            latest_blob.download_to_filename(latest_scaler_path_to_save)

            with open(latest_scaler_path_to_save, "rb") as file:
                scaler = pickle.load(file)

            print("✅ Latest scaler downloaded from cloud storage")

            return scaler

        except Exception as e:
            print(f"\n❌ Error loading scaler from GCS: {e}")

            return None

    return None
