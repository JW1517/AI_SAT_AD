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
from google.cloud import storage

# import mlflow
# from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_dir= os.path.join(LOCAL_REGISTRY_PATH, "params")
        params_path = os.path.join(params_dir, timestamp + ".pickle")
        os.makedirs(params_dir, exist_ok=True)
        # if os.path.isdir(params_path):
        #     os.rmdir(params_path)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        metrics_path = os.path.join(metrics_dir, timestamp + ".pickle")

        os.makedirs(metrics_dir, exist_ok=True)
        # if os.path.isdir(metrics_path):
        #     os.rmdir(metrics_path)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results params ans metrics saved locally")


def save_model(model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    model_path = os.path.join(model_dir, f"{timestamp}.h5")

    os.makedirs(model_dir, exist_ok=True)
    #model.save(model_path)

        # Export Pipeline as pickle file
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None



    return None

# load_model
def load_model(stage="Production"):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
        #print(blobs)

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            print(latest_blob)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,latest_blob.name)
            os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
            #print(latest_model_path_to_save)

            latest_blob.download_to_filename(latest_model_path_to_save)
            print("ok")
            latest_model = pickle.load(open(latest_model_path_to_save,"rb"))
            print(f"⬇️ Downloading: {latest_blob.name}")

            print("✅ Latest model downloaded from cloud storage")


            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None


# Arsène, Yves, 20250609

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
