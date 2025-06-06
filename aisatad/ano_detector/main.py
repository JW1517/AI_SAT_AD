import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from aisatad.params import *
from aisatad.function_files.data import get_data_with_cache
from aisatad.function_files.plot import plot_seg_ax
from aisatad.function_files.preprocessor import preprocess
from aisatad.function_files.model import model_stacking

#scaler
from sklearn.preprocessing import StandardScaler


# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



#load raw_data.csv

data_bucket_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw_data", f"{BUCKET_DATASET}.csv")
df = get_data_with_cache(cache_path=data_bucket_cache_path)

# preprocess
X_train, X_test, y_train, y_test = preprocess(df)

print("✅ preprocess() done \n")



# modeliser and score
df_results_stacking = model_stacking(X_train, X_test, y_train, y_test)
print(df_results_stacking)







if __name__ == '__main__':
    #data()
    #preprocessing()
    #model()
    pass
