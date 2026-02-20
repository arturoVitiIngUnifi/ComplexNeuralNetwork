import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from network.ComplexNetwork import ComplexNetwork

train_file = "dataset/train_def.csv"
test_file = "dataset/test_def.csv"

# Load datasets
df_train = pd.read_csv(train_file, sep=';', header=None, dtype=str)
df_test = pd.read_csv(test_file, sep=';', header=None, dtype=str)

# --- Clean numeric strings ---
def clean_numeric_string(s):
    s = str(s).strip()
    if '.' in s:
        first, *rest = s.split('.')
        s = first + '.' + ''.join(rest)
    return s

for col in df_train.columns:
    df_train[col] = df_train[col].apply(clean_numeric_string)
    df_test[col] = df_test[col].apply(clean_numeric_string)

# --- Features and target ---
X_train = df_train.iloc[:, :50].astype(np.float64).values
Y_train = df_train.iloc[:, 50].astype(np.float64).values.reshape(-1, 1)

X_test = df_test.iloc[:, :50].astype(np.float64).values
Y_test = df_test.iloc[:, 50].astype(np.float64).values.reshape(-1, 1)

# --- SCALERS ---
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test  = X_scaler.transform(X_test)

Y_train = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

# --- Network ---
net = ComplexNetwork([50, 30, 1])

net.train(
    X_train,
    Y_train,
    epochs=1000,
    errorThresholdToStop=1e-4
)

# --- Prediction ---
Y_pred_scaled = np.array([
    net.feedforward(x)[0].real for x in X_test
]).reshape(-1, 1)

# 🔁 Inverse scaling
Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)

print("Y_test mean (original):", Y_test.mean())
print("Y_pred mean (inverse scaled):", Y_pred.mean())

# --- Metrics ---
MSE = np.mean((Y_test - Y_pred) ** 2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(Y_test - Y_pred))
R2 = 1 - np.sum((Y_test - Y_pred) ** 2) / np.sum(
    (Y_test - np.mean(Y_test)) ** 2
)

print("Y mean:", Y_train.mean())
print("Y std:", Y_train.std())
print("\n=== Risultati Test ===")
print(f"MSE:  {MSE:.6f}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE:  {MAE:.6f}")
print(f"R²:   {R2:.6f}")
