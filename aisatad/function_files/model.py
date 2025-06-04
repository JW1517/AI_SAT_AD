# Data Manipulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, roc = (accuracy_score(y_test, y_pred),
                                            precision_score(y_test, y_pred),
                                            recall_score(y_test, y_pred),
                                            f1_score(y_test, y_pred),
                                            roc_auc_score(y_test, y_pred)
                                            )

    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1 : {f1:.3f} - roc : {roc:3f}"


def SVC_model(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear',
                C=1000,gamma=0.0001,
                coef0=0,
                class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, roc = (accuracy_score(y_test, y_pred),
                                            precision_score(y_test, y_pred),
                                            recall_score(y_test, y_pred),
                                            f1_score(y_test, y_pred),
                                            roc_auc_score(y_test, y_pred)
                                            )
    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1 : {f1:.3f} - roc : {roc:3f}"


def ada_boost(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(algorithm='SAMME',
                               estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                max_depth=10,
                                                                min_samples_leaf=2,
                                                                min_samples_split=10))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1, roc = (accuracy_score(y_test, y_pred),
                                            precision_score(y_test, y_pred),
                                            recall_score(y_test, y_pred),
                                            f1_score(y_test, y_pred),
                                            roc_auc_score(y_test, y_pred)
                                            )

    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1 : {f1:.3f} - roc : {roc:3f}"


def xgboost_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        colsample_bytree=0.8,
        eval_metric='logloss',
        gamma=0.3,
        learning_rate=0.2,
        max_depth=5,
        min_child_weight=1,
        n_estimators=50,
        use_label_encoder=False,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        tree_method='hist'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1 : {f1:.3f} - roc : {roc:.3f}"



# Juan
df = pd.read_csv("/Users/juanborron/code/JW1517/AI_SAT_AD/raw_data/feature_data.csv")
X = df.iloc[:, -18:]
y = df['anomaly']

# def model():
    #1. init()
    #2. train()
    #3. evaluate()
    #4. pred()

# stacking

lreg = LogisticRegression(class_weight="balanced")
svc = SVC(kernel='linear',
                C=1000,gamma=0.0001,
                coef0=0,
                class_weight="balanced")
adabstc = AdaBoostClassifier(algorithm='SAMME',
                               estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                max_depth=10,
                                                                min_samples_leaf=2,
                                                                min_samples_split=10))
xgbc = XGBClassifier(
        colsample_bytree=0.8,
        eval_metric='logloss',
        gamma=0.3,
        learning_rate=0.2,
        max_depth=5,
        min_child_weight=1,
        n_estimators=50,
        use_label_encoder=False,
        #scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        tree_method='hist'
    )

model = StackingClassifier(
    estimators=[("lreg", lreg),("svc", svc), ("adabstc", adabstc), ("xgbc", xgbc)],
    final_estimator=LogisticRegression(max_iter= 1000),
    cv=5,
    n_jobs=-1
)


