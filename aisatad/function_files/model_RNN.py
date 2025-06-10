import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import Sequential, Input, layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

def preprocess_RNN(df) :
#Créer un X_train, X_test, y_train, y_test utilisable en RNN depuis de la raw_data dataframe#

    # Fit scaler sur le train
    scaler = StandardScaler()
    train_data = df[df['train'] == 1]
    scaler.fit(train_data[['value']])

    # Transform scaler
    df['value_scaled'] = scaler.transform(df[['value']])

    # Group le DF par segment
    grouped_segments = df.groupby('segment')

    X_train, y_train = [], []
    X_test, y_test = [], []

    for id, group in grouped_segments:
        values = group['value_scaled'].values
        label = group['anomaly'].iloc[0]
        is_train = group['train'].iloc[0]

        if is_train == 1:
            X_train.append(values)
            y_train.append(label)
        else:
            X_test.append(values)
            y_test.append(label)

    # Padding
    X_train = pad_sequences(X_train, padding='post', maxlen=350, truncating='post', dtype='float32',value=-10.0)
    X_test  = pad_sequences(X_test, padding='post', maxlen=350, truncating='post', dtype='float32',value=-10.0)

    # Créer les bonnes shape pour un RNN
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    return X_train, X_test, y_train, y_test



def RNN_model(X_train, X_test, y_train, y_test):
#Créer un modèle RNN et afficher les résultats#

    # Création du modèle
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.Masking(mask_value=-10.0))
    model.add(layers.Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True)))
    model.add(layers.LSTM(units=32, activation='tanh', return_sequences=False))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))


    # Compilation
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=1e-3),
                metrics=['accuracy', Precision(), Recall()],
                )

    # Gestion du déséquilibre
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)

    class_weight_dict = dict(enumerate(class_weights))

    # Entraînement
    history = model.fit(X_train, y_train,
            epochs=100,
            batch_size=16,
            verbose=0,
            callbacks = [EarlyStopping(patience=10, restore_best_weights=True)],
            validation_split=0.2,
            class_weight=class_weight_dict
            )

    # Évaluation
    results = model.evaluate(X_test, y_test, verbose=0)

    # Confusion matrix
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int").flatten()
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", cm)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité réelle")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    return print(f"Test Accuracy: {results[1]:.3f}, Test Precision: {results[2]:.3f}, Test Recall: {results[3]:.3f}")
