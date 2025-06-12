import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import Sequential, Input, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from sklearn.metrics import classification_report


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from aisatad.params import *
from aisatad.function_files.data import get_data_with_cache
from aisatad.function_files.plot import plot_seg_ax
from aisatad.function_files.preprocessor import preprocess
from aisatad.function_files.model import model_stacking

from sklearn.metrics import f1_score, roc_auc_score
from tensorflow.keras.metrics import AUC




def Dense_Neural_model(X_train, X_test, y_train, y_test):

    model = Sequential()

    model.add(layers.Dense(128, activation='gelu', input_shape=(X_train.shape[1],)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation='gelu'))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation='gelu'))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='sigmoid'))

    es = EarlyStopping(patience=10,restore_best_weights=True)

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy', Precision(), Recall(),AUC(name='auc')],
                )

    y_train = y_train.astype(int)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)

    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    history = model.fit(X_train, y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks = [es],
            validation_split=0.2,
            class_weight=class_weight_dict
            )

    results = model.evaluate(X_test, y_test, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int").flatten()
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", cm)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Prédiction")
    plt.ylabel("Valeur réelle")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {results[1]:.3f}, Test Precision: {results[2]:.3f}, Test Recall: {results[3]:.3f}, F1-score: {f1:.3f}, ROC AUC: {roc_auc:.3f}")

    return history
