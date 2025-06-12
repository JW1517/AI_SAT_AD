# base pour manipuler les data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Preprocessing
from sklearn.preprocessing import StandardScaler


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# calcul time
import time

#plot le ROC
from aisatad.function_files.plot import plot_roc_curves_model_staking




# def function of Arsene et Joss 0602

def logistic_regression(X_train, X_test, y_train, y_test):
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


def SVC_model(X_train, X_test, y_train, y_test):
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


def ada_boost(X_train, X_test, y_train, y_test):
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


def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        colsample_bytree=0.9720296698433066,
        gamma=0.0788536227370118,
        learning_rate=0.05594937721224293,
        max_depth=9,
        min_child_weight=1,
        n_estimators=600,
        use_label_encoder=False,
        eval_metric='auc',
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

    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, roc: {roc:.3f}"


# Juan0603:  def model_stacking() qui compare tous les models de mardi(par Arsene et Joss), plus model_stacking
# et return les scores of comparason dans un df_results_stacking

def model_stacking(X_train, X_test, y_train, y_test):
    """
    1. lister les models déjà fait gridsearch
    2. stacking les models
    3. entrainer les models plus celui de stacking, et mis les results de scores dans un dataframe
    """

    """
    1. lister les models déjà fait gridsearch
    """

    #supervised model à stacking: LogisticRegression
    lreg = LogisticRegression(class_weight="balanced")
    #supervised model à stacking: SVC
    svc = SVC(kernel='linear',
                    C=1000,gamma=0.0001,
                    coef0=0,
                    class_weight="balanced",
                    probability=True)
    #supervised model à stacking: AdaBoostClassifier
    adabstc = AdaBoostClassifier(estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                    max_depth=10,
                                                                    min_samples_leaf=2,
                                                                    min_samples_split=10)
                                )
    #supervised model à stacking: XGBClassifier
    xgbc = XGBClassifier(
        colsample_bytree=0.9720296698433066,
        gamma=0.0788536227370118,
        learning_rate=0.05594937721224293,
        max_depth=9,
        min_child_weight=1,
        n_estimators=600,
        use_label_encoder=False,
        eval_metric='auc',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        tree_method='hist'
        )

    """
    2. stacking les models
    """

    mdl_stacking = StackingClassifier(
        estimators=[("lreg", lreg),("svc", svc), ("adabstc", adabstc), ("xgbc", xgbc)],
        final_estimator=LogisticRegression(max_iter= 1000),
        cv=5,
        n_jobs=-1
    )

    """
    3. entrainer les models plus celui de stacking, et mis les results de scores dans un dataframe
    """

    # dictionary of models

    model_lists = {
            "lreg": lreg,
            "svc": svc,
            "adabstc": adabstc,
            "xgbc": xgbc,
            "mdl_stack" :mdl_stacking
    }

    results_y_pred = []
    results_metrics =[]
    results_model = []
    #for loop iterate model to get scores
    for model_nm, model_t in model_lists.items():

        start_time = time.time()
        #train model
        model_t.fit(X_train, y_train)
        # model predict
        y_pred = model_t.predict(X_test)
        #calcul les metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        #calul time
        cal_time = time.time()- start_time


        #print les metrics 
        print(f" results for model {model_nm}:  [accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f},f1 : {f1:.3f} - roc : {roc:.3f}, elapsed_time: {cal_time:.3f}]")
        # save results dans une list
        results_metrics.append({
            "model_nm": model_nm,
            "accuracy": accuracy,
            "precision":precision,
            "recall":recall,
            "f1":f1,
            "roc":roc,
            "elapsed_time": cal_time
            })
        results_y_pred.append({
            "y_pred": y_pred,
            })
        results_model.append({
            "model": model_t
        }

        )


    #transformer en pd
    df_results_y_pred = pd.DataFrame(results_y_pred)
    df_results_metrics = pd.DataFrame(results_metrics)
    df_results_stacking= pd.concat([df_results_metrics, df_results_y_pred], axis=1)
    df_results_model = pd.DataFrame(results_model)

    # #plot le ROC
    # plot_roc_curves_model_staking(model_lists, X_test, y_test)

    return df_results_model, df_results_y_pred,df_results_metrics,df_results_stacking
