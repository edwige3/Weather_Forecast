import numpy as np
import pandas as pd

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