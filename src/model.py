from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(X_train, y_train):
    """
        Inputs:
            - X_train
            - y_train
        Return:
            - A trained RF model on X_train and y_train with 100 estimators
    """
    model = RandomForestRegressor(n_estimators=100, random_state=2)
    model.fit(X_train, y_train)
    return model

def model_evaluation(model, X_test, y_test):
    """
        Inputs:
            - A trained model
            - X_test
            - y_test
        Return:
            - predictions y_pred
            - MAE.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mae
