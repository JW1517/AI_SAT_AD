import matplotlib.pyplot as plt

def plot_seg(df_raw, list_seg):
    for segment in list_seg:
        df_filtered = df_raw[df_raw["segment"] == segment]
        plt.figure(figsize=(15, 5))
        plt.scatter(df_filtered["timestamp"], df_filtered["value"])
        plt.title(f"Segment {segment} - Anomaly {df_filtered['anomaly'].iloc[0]} - Channel {df_filtered['channel'].iloc[0]} - Sampling {df_filtered['sampling'].iloc[0]}")
        plt.tight_layout()
        print(plt.show())

# ajouter un foncton avec ax dans le plot_seg: 20250605
def plot_seg_ax(ax, df_raw, segment):
    df_filtered = df_raw[df_raw["segment"] == segment]
    ax.scatter(df_filtered["timestamp"], df_filtered["value"])
    ax.set_title(f"Segment {segment}\nAnomaly {df_filtered['anomaly'].iloc[0]} | Channel {df_filtered['channel'].iloc[0]}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")

# pour plot le ROC
from sklearn.metrics import roc_curve, auc

def plot_roc_curves_model_staking(model_lists, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model_name, model in model_lists.items():
        # Vérifie si le modèle supporte predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]  #[0.3,0.7], proba chaque ligne soit dans la classe 1 (la classe positive)
        else:
            print(f"Le modèle {model_name} ne supporte pas predict_proba ni decision_function.")
            continue

        # Extract associated metrics and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        # Compute AUC score: L’aire sous la courbe ROC
        # The larger the AUC, the greater the overall general performance.
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire") #  correspond à un modèle aléatoire, sans pouvoir prédictif.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbes ROC comparées")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
