import pandas as pd


from google.cloud import storage
from colorama import Fore, Style
from pathlib import Path

from aisatad.params import *


def get_data_with_cache():

    # print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # query to download data

    """
    Retrieve `bucket` data from storage, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from storage for future use
    """
    # CHANGED TO SIMPLIFIED GET DATA WITHOUT FUNCTION ARGUMENTS REQUIRED
    
    cache_path = Path(LOCAL_DATA_PATH).joinpath("raw_data", f"{BUCKET_DATASET}.csv")

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)

    else:
        print(Fore.BLUE + "\nLoad data from storage server..." + Style.RESET_ALL)

        storage_filename = "raw_data.csv"
        local_filename = "raw_data.csv"

        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        #Chercher le fichier storage_filename dans le bucket
        blob = bucket.blob(storage_filename)
        #Stocker le fichier en local sous le nom local_filename
        blob.download_to_filename(cache_path)

        print(f"✅ Fichier téléchargé dans {local_filename}")


    df = pd.read_csv(cache_path, parse_dates=["timestamp"])

    return df
