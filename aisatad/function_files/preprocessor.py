# Imports
from aisatad.params import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
import scipy.signal as sig
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', 60)

# Fonctions de feature engineering sur des array (1D)
def number_of_peaks_finding(array):
    prominence = 0.1 * (np.max(array) - np.min(array))
    peaks = sig.find_peaks(array, prominence=prominence)[0]
    return len(peaks)

def smooth10_n_peaks(array):
    kernel = np.ones(10) / 10
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)

def smooth20_n_peaks(array):
    kernel = np.ones(20) / 20
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)

def diff_peaks(array):
    array_diff = np.diff(array)
    return number_of_peaks_finding(array_diff)

def diff2_peaks(array):
    array_diff = np.diff(array, n=2)
    return number_of_peaks_finding(array_diff)

def diff_var(array):
    array_diff = np.diff(array)
    return np.var(array_diff)

def diff2_var(array):
    array_diff = np.diff(array, n=2)
    return np.var(array_diff)

# Fonctions de feature engineering sur des DataFrames
def duration(df):
    t1 = pd.Timestamp(df.head(1).timestamp.values[0])
    t2 = pd.Timestamp(df.tail(1).timestamp.values[0])
    return (t2 - t1).seconds

def gaps_squared(df):
    df = df.copy()
    df['timestamp2'] = df['timestamp'].shift(1)
    df = df.reset_index().iloc[1:, :]
    df['time_delta'] = (df.timestamp - df.timestamp2).dt.seconds
    df['time_delta_squared'] = df['time_delta'] ** 2
    return df.time_delta_squared.sum()

def zero_deriv_duration(df):
    array = df["value"].values
    timestamps = df["timestamp"].values
    diffs = np.diff(array)
    zero_diff_indices = np.where(diffs == 0)[0]
    if len(zero_diff_indices) == 0:
        return 0
    zero_timestamps = np.array(timestamps)[zero_diff_indices + 1]
    zero_timestamps = pd.to_datetime(zero_timestamps)
    stretches = []
    start = zero_timestamps[0]
    for i in range(1, len(zero_timestamps)):
        gap = (zero_timestamps[i] - zero_timestamps[i - 1]).total_seconds()
        if gap > 1:
            end = zero_timestamps[i - 1]
            stretches.append((start, end))
            start = zero_timestamps[i]
    stretches.append((start, zero_timestamps[-1]))
    total_duration = sum((end - start).total_seconds() for start, end in stretches)
    return total_duration

# Dictionnaire des features array
array_transformations = {
    "len": len,
    "mean": np.mean,
    "var": np.var,
    "std": np.std,
    "kurtosis": kurtosis,
    "skew": skew,
    "n_peaks": number_of_peaks_finding,
    "smooth10_n_peaks": smooth10_n_peaks,
    "smooth20_n_peaks": smooth20_n_peaks,
    "diff_peaks": diff_peaks,
    "diff2_peaks": diff2_peaks,
    "diff_var": diff_var,
    "diff2_var": diff2_var,
}

# Dictionnaire des features DataFrame
df_transformations = {
    "duration": duration,
    "gaps_squared": gaps_squared,
    "zero_deriv_duration": zero_deriv_duration,
}

# Génération du dataset
def generate_dataset(source_df):
    dataset = []
    target_name = "features_dataset"
    for segment_id in tqdm(source_df.segment.unique()):
        tdf = source_df[source_df.segment == segment_id]

        ligne = [
            segment_id,
            int(tdf["anomaly"].iloc[0]),
            tdf["train"].iloc[0],
            tdf["channel"].iloc[0],
            tdf["sampling"].iloc[0],
        ]

        for func in array_transformations.values():
            ligne.append(func(tdf["value"].values))

        for func in df_transformations.values():
            ligne.append(func(tdf))

        dataset.append(ligne)

    columns = ["segment", "anomaly", "train", "channel", "sampling"]
    columns += list(array_transformations.keys())
    columns += list(df_transformations.keys())

    dataset = pd.DataFrame(dataset, columns=columns)

    dataset["len_weighted"] = dataset["sampling"] * dataset["len"]
    dataset["var_div_duration"] = dataset["var"] / dataset["duration"]
    dataset["var_div_len"] = dataset["var"] / dataset["len"]

    full_path = f"{LOCAL_DATA_PATH}/{target_name}.csv"
    dataset.to_csv(full_path, index=False)

    return dataset, full_path

# Fonction principale de preprocessing
def preprocess(df):
    dataset, _ = generate_dataset(df)

    X_train = dataset[dataset['train'] == 1].iloc[:, 5:]
    X_test = dataset[dataset['train'] == 0].iloc[:, 5:]
    y_train = dataset[dataset['train'] == 1]['anomaly']
    y_test = dataset[dataset['train'] == 0]['anomaly']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns = X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns = X_test.columns)

    return X_train, X_test, y_train, y_test, scaler


def api_preprocess(df, fitted_scaler):
    dataset, _ = generate_dataset(df)

    X = dataset.iloc[:, 5:]

    X_scaled = fitted_scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled
