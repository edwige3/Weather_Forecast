import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


from src.data_loader import get_data
from src.features import feature_engineering
from src.model import train_model, model_evaluation
from src.plot import plot

# Collect the selected target from the user:
parser = ArgumentParser()
parser.add_argument('-t','--target', help='target.', default="maxtp", required=True)
main_args = vars(parser.parse_args())

target= main_args["target"].lower()

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



