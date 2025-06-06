import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import Sequential, Input, layers
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



def grid_search(X_train, y_train):

    param_grid = {
        'units_1': [32, 64],
        'units_2': [16, 32],
        'dense_units': [32, 64],
        'dropout_rate': [0.2, 0.3, 0.5],
        'learning_rate': [1e-3, 1e-4],
        'epochs': [30, 50, 100],
        'batch_size': [16, 32]
    }

    best_val_acc = 0
    best_params = None
    best_model = None

    for units_1 in param_grid['units_1']:
        for units_2 in param_grid['units_2']:
            for dense_units in param_grid['dense_units']:
                for dropout_rate in param_grid['dropout_rate']:
                    for learning_rate in param_grid['learning_rate']:
                        for epochs in param_grid['epochs']:
                            for batch_size in param_grid['batch_size']:
                                params = {
                                    'units_1': units_1,
                                    'units_2': units_2,
                                    'dense_units': dense_units,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': learning_rate,
                                    'epochs': epochs,
                                    'batch_size': batch_size
                                }
                                print(f"Testing params: {params}")

                                model = Sequential()
                                model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
                                model.add(layers.Masking(mask_value=-10.0))
                                model.add(layers.Bidirectional(layers.LSTM(units=units_1, activation='tanh', return_sequences=True)))
                                model.add(layers.LSTM(units=units_2, activation='tanh', return_sequences=False))
                                model.add(layers.Dense(dense_units, activation='relu'))
                                model.add(layers.Dropout(dropout_rate))
                                model.add(layers.Dense(1, activation='sigmoid'))

                                optimizer = Adam(learning_rate=learning_rate)
                                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                                es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)

                                history = model.fit(X_train, y_train,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    validation_split=0.2,
                                                    callbacks=[es],
                                                    verbose=0)

                                val_acc = max(history.history['val_accuracy'])
                                print(f"Validation accuracy: {val_acc:.4f}")

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_params = params
                                    best_model = model

    print(f"Best validation accuracy: {best_val_acc:.4f} with params: {best_params}")
    return best_model, best_params, best_val_acc
