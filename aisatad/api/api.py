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

#import for FastAPI
from fastapi import FastAPI

# a fast api
app = FastAPI()
