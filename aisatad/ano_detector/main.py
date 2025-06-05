import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from aisatad.params import *
from aisatad.function_files.data import get_data_with_cache
from aisatad.function_files.model import model_stacking

#scaler
from sklearn.preprocessing import StandardScaler

# pipiline
from sklearn.pipeline import make_pipeline

#model
from sklearn.model_selection import cross_val_score,cross_validate

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



#load raw_data.csv

data_bucket_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw_data", f"{BUCKET_DATASET}.csv")
df = get_data_with_cache(cache_path=data_bucket_cache_path)

# preprocess=> à remplacer par les vrais preprocess
path_csv = data_bucket_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw_data", "feature_data.csv")
print(path_csv)
df_featurs = pd.read_csv(path_csv)
X = df_featurs.iloc[:, -18:]
y = df_featurs['anomaly']
def preproc(df_featurs):
    # prepare data
    #df_featurs = get_data_with_cache(cache_path=data_bucket_cache_path)
    X_train = df_featurs[df_featurs['train'] == 1].iloc[:, -18:]
    X_test = df_featurs[df_featurs['train'] == 0].iloc[:, -18:]
    y_train = df_featurs[df_featurs['train'] == 1]['anomaly']
    y_test = df_featurs[df_featurs['train'] == 0]['anomaly']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test,y_test
X_train, y_train, X_test,y_test = preproc(df_featurs)

print("✅ preprocess() done \n")



# modeliser and score
df_results_stacking = model_stacking(X_train, y_train, X_test,y_test)
print(df_results_stacking)





if __name__ == '__main__':
    #data()
    #preprocessing()
    #model()
    pass
