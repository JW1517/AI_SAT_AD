


def SVC_model(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear',C=1000,gamma=0.0001, coef0=0, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall = (accuracy_score(y_test, y_pred),
                                   precision_score(y_test, y_pred),
                                   recall_score(y_test, y_pred)
                                   )
    return f"accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}"
