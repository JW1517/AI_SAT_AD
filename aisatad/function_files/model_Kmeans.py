import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from aisatad.function_files.preprocessor import preprocess
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def anomaly_detection_kmeans_eval(X_train, X_test, y_train, y_test, n_clusters=10, threshold_percentile=95):

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)

    # Calcul distances au centre
    def dist_to_center(X, model):
        centers = model.cluster_centers_
        labels = model.predict(X)
        distances = np.linalg.norm(X - centers[labels], axis=1)
        return distances

    train_distances = dist_to_center(X_train, kmeans)
    test_distances = dist_to_center(X_test, kmeans)

    # Seuil = percentile sur distances train
    threshold = np.percentile(train_distances, threshold_percentile)
    print(f"Seuil distance pour anomalie : {threshold:.3f}")

    # Prédiction : distance > seuil → anomalie (1)
    y_pred_train = (train_distances > threshold).astype(int)
    y_pred_test = (test_distances > threshold).astype(int)

    # Scores sur test
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, test_distances)

    print("=== Résultats Test ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC AUC : {roc_auc:.3f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.xlabel("Prédiction")
    plt.ylabel("Vrai label")
    plt.title("Matrice de confusion - Anomalies détectées")
    plt.show()

    return y_pred_test, threshold, kmeans
