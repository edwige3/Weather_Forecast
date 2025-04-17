"""
    Link to the project: https://github.com/edwige3/Weather_Forecast.git
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def feature_engineering(df, target, n_lags=7, horizon= [1, 7]):
    """
        Inputs:
            - A dataframe df.
            - target: variable of interest.
            - n_lags: number of lags in interger.
            - horizon: a list of horizons (number of day ahead).

        Returns:
            - X: Input data with lags features.
            - y_1d: target values for 1-day ahead
            - y_7d: target values for 7-day ahead
    """
    features= list(df.columns)
    n_lags= 4
    # Create lag features (up to n_lags previous days)
    for lag in range(1, n_lags+1):
        for col in features:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)


    # target = 'maxtp'  # Température max
    df['target_1d'] = df[target].shift(-horizon[0])
    df['target_7d'] = df[target].shift(-horizon[1])

    df.drop(columns=features, inplace=True)
    df.dropna(inplace=True)

    X = df.iloc[:, :-2]
    # Convert features to numeric, handling errors
    for col in list(df.columns[:-2]):
        X[col] = pd.to_numeric(X[col], errors='coerce')  # Convert to numeric, invalid parsing set to NaN
    
    y_1d = df['target_1d']
    y_7d = df['target_7d']
    return X, y_1d, y_7d

def get_data():
    df = pd.read_csv("./data/dly1475.csv",skiprows=24)
    df = df.dropna(axis=1, how='all')

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    df.set_index('date', inplace=True)

    return df

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
def plot(y_true, y_pred, horizon):
    """
        Input: 
            - horizon
            - y_pred
            - y_true
    """

    plt.figure(figsize=(12, 5))
    plt.plot(y_true.values, label="Truth data")
    plt.plot(y_pred, label="Prediction")

    plt.legend()
    plt.title("Weather forecast {}-day ahead".format(horizon))
    plt.grid()
    plt.savefig("./images/pred_{}.png".format(str(horizon)))
    plt.show()
target= "maxtp"
# 1. Load Data
df = get_data()

# 2. Create Features
n_lags = 7
horizon = [1, 7]
X, y_1d,  y_7d= feature_engineering(df, target, n_lags, horizon)

# 3. Split into train and test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_1d, test_size=0.2, shuffle=False)
X_train7, X_test7, y_train7, y_test7 = train_test_split(X, y_7d, test_size=0.2, shuffle=False)

# 4. Train & Predict

## 1-day ahead:
model1 = train_model(X_train1, y_train1)
y_pred1, mae1 = model_evaluation(model1, X_test1, y_test1)

## 7-day ahead:
model7 = train_model(X_train7, y_train7)
y_pred7, mae7 = model_evaluation(model7, X_test7, y_test7)

print("\n" + "-" * 30)
print("          MAE Report")
print("-" * 30)
print(f" MAE for 1-day prediction : {mae1:.5f}")
print(f" MAE for 7-day prediction : {mae7:.5f}")
print("-" * 30 + "\n")

# Save our MAEs
# Create a small DataFrame
df_metrics = pd.DataFrame({
    'Metric': ['mae1', 'mae7'],
    'Value': [mae1, mae7]
})
# Save to CSV
df_metrics.to_csv('./data/maes.csv', index=False)
# 7. Plot Forecasts
plot(y_test1, y_pred1, horizon=horizon[0])
plot(y_test7, y_pred7, horizon=horizon[1])

