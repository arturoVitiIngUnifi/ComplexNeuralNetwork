import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from network.ComplexNetwork import ComplexNetwork

train_file = "dataset/train_def.csv"
test_file = "dataset/test_def.csv"

df_train = pd.read_csv(train_file, sep=';', header=None, dtype=str)
df_test = pd.read_csv(test_file, sep=';', header=None, dtype=str)


def clean_numeric_string(s):
    s = str(s).strip()
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    return s


for col in df_train.columns:
    df_train[col] = df_train[col].apply(clean_numeric_string)
for col in df_test.columns:
    df_test[col] = df_test[col].apply(clean_numeric_string)

X = df_train.iloc[:, :50].astype(np.float64).values
Y = df_train.iloc[:, 50].astype(np.float64).values.reshape(-1, 1)

# Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== Fold {fold} ===")

    # Split train/validation
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_fold)
    X_val_scaled = scaler_X.transform(X_val_fold)

    # Scale dependent variable
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train_fold)
    Y_val_scaled = scaler_Y.transform(Y_val_fold)

    n_input = X_train_scaled.shape[1]
    n_output = Y_train_scaled.shape[1]
    net = ComplexNetwork([n_input, 30, 20, n_output])

    # Training
    history = net.train(X_train_scaled, Y_train_scaled, epochs=1000, errorThresholdToStop=1e-4)

    # Prediction
    Y_pred_scaled = np.array([net.feedforward(x)[0].real for x in X_val_scaled]).reshape(-1, 1)
    Y_val_real = scaler_Y.inverse_transform(Y_val_scaled)
    Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)

    # Accuracy Metrics
    MSE = np.mean((Y_val_real - Y_pred_real) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y_val_real - Y_pred_real))
    R2 = 1 - np.sum((Y_val_real - Y_pred_real) ** 2) / np.sum((Y_val_real - np.mean(Y_val_real)) ** 2)

    print(f"MSE: {MSE:.6f}, RMSE: {RMSE:.6f}, MAE: {MAE:.6f}, R²: {R2:.6f}")

    fold_results.append((MSE, RMSE, MAE, R2))

fold_results = np.array(fold_results)
print("\n=== Risultati medi K-Fold ===")
print(f"MSE: {fold_results[:, 0].mean():.6f}")
print(f"RMSE: {fold_results[:, 1].mean():.6f}")
print(f"MAE: {fold_results[:, 2].mean():.6f}")
print(f"R²: {fold_results[:, 3].mean():.6f}")
