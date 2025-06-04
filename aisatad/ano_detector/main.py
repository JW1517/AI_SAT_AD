import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from aisatad.params import *
from aisatad.function_files.data import get_data_with_cache #clean_data, load_data_to_bq
# Model Imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,StackingRegressor,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#scaler
from sklearn.preprocessing import StandardScaler

# pipiline
from sklearn.pipeline import make_pipeline

#model
from sklearn.model_selection import cross_val_score,cross_validate

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# data():

# def preprocessing():
#load data

data_bucket_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{BQ_DATASET}.csv")
df = get_data_with_cache(cache_path=data_bucket_cache_path)


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






# #pipe_stacking = make_pipeline(preproc(df), model, memory=cachedir)
# scoring = ['accuracy', 'f1', 'roc_auc']

# #scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
# # Résultats
# print("Accuracy (mean ± std): {:.4f} ± {:.4f}".format(scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
# print("F1-score (mean ± std): {:.4f} ± {:.4f}".format(scores['test_f1'].mean(), scores['test_f1'].std()))
# print("ROC AUC (mean ± std): {:.4f} ± {:.4f}".format(scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()))





if __name__ == '__main__':
    #data()
    #preprocessing()
    #model()
    pass
