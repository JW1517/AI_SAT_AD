import pandas as pd


from google.cloud import storage
from colorama import Fore, Style
from pathlib import Path

from aisatad.params import *


def get_data_with_cache(cache_path: Path,
                        storage_filename: str = "raw_data.csv") -> pd.DataFrame:


    # print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # query to download data



    """
    Retrieve `bucket` data from storage, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from storage for future use
    """


    if cache_path.is_file():
        print(Fore.BLUE + f"\nLoad data from local CSV:{cache_path.name}..." + Style.RESET_ALL)

    else:
        print(Fore.BLUE + f"\nLoad data from storage server:{cache_path.name}..." + Style.RESET_ALL)

        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        #Chercher le fichier storage_filename dans le bucket
        blob = bucket.blob(storage_filename)
        #creat le dossier si cela n'existe pas
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        #Stocker le fichier en local sous le nom local_filename
        blob.download_to_filename(cache_path)

        print(f"✅ Fichier téléchargé dans {cache_path}")


    df = pd.read_csv(cache_path, parse_dates=["timestamp"])

    return df
