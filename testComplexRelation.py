import numpy as np
import pandas as pd
from network.ComplexNetwork import ComplexNetwork


train_file = "dataset_complesso.csv"
df = pd.read_csv(train_file, sep=',', header=0, dtype=str)

# Convert string to complex
def str_to_complex(s):
    return complex(s.replace(' ', ''))

X = df.iloc[:, :-1].apply(lambda col: col.map(str_to_complex)).values
Y = df.iloc[:, -1].map(str_to_complex).values.reshape(-1, 1)  # array 2D

n_samples = X.shape[0]
split_index = int(n_samples * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

net = ComplexNetwork([X.shape[1], 3, 1])
net.train(
    X_train,
    Y_train,
    epochs=1000,
    errorThresholdToStop=1e-4
)

Y_pred = []
for x in X_test:
    Y_pred.append(net.feedforward(list(x)))

Y_pred = np.array(Y_pred).reshape(-1, 1)

# --- Metrics
errors = np.abs(Y_test - Y_pred)

rmse = np.sqrt(np.mean(errors ** 2))
mae = np.mean(errors)
mse = np.mean(errors ** 2)
max_error = np.max(errors)

# R² Score
y_mean = np.mean(np.abs(Y_test))
ss_tot = np.sum((np.abs(Y_test) - y_mean) ** 2)
ss_res = np.sum(errors ** 2)
r2 = 1 - ss_res / ss_tot

# Mean Absolute Percentage Error
mape = np.mean(errors / (np.abs(Y_test) + 1e-8)) * 100

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Max Error: {max_error:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")