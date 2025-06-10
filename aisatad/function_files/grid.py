from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras import Sequential, Input, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight


def ada_boost_GC(X_train, y_train, X_test, y_test):
    base_estimator = DecisionTreeClassifier(class_weight='balanced')

    param_grid = {
        'n_estimators': [10, 25, 50, 75, 100, 150, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0],
        'algorithm': ['SAMME', 'SAMME.R'],
        'estimator__max_depth': [1, 2, 3, 4, 5, 8, 10, None],
        'estimator__min_samples_split': [2, 5, 10, 20, 50],
        'estimator__min_samples_leaf': [1, 2, 4, 10, 20],
        'estimator__max_features': ['sqrt', 'log2', None],
        'estimator__criterion': ['gini', 'entropy', 'log_loss']
    }

    model = AdaBoostClassifier(estimator=base_estimator)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=150,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        cv=5,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    return f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, roc: {roc:.3f}", best_model



def xgboost_GC(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    )

    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500, 700, 1000],
        'learning_rate': [
            0.0005, 0.001, 0.003, 0.005,
            0.01, 0.02, 0.03, 0.04, 0.05,
            0.07, 0.1, 0.2, 0.3, 0.5
        ],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
        'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
        'reg_lambda': [0.01, 0.1, 1, 5, 10, 20, 50],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'importance_type': ['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=200,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        cv=5,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    return (
        f"accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, "
        f"f1: {f1:.3f}, roc: {roc:.3f}",
        best_model
    )


def grid_search_RNN(X_train, y_train):

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
    return best_model, best_params, best_val_ac
