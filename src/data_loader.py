import pandas as pd

def get_data():
    df = pd.read_csv("./data/dly1475.csv",skiprows=24)
    df = df.dropna(axis=1, how='all')

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    df.set_index('date', inplace=True)

    return df