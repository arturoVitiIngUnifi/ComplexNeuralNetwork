from typing import List
import numpy as np
import cmath

from network.ComplexNetwork import ComplexNetwork
from sklearn.preprocessing import MinMaxScaler


def generate_mackey_glass(length=5000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    max_delay = tau
    x = np.zeros(length + max_delay + 1)
    x[0] = 1.2

    for t in range(1, length + max_delay):
        x_tau = x[t - tau] if t - tau >= 0 else 0.0
        x[t + 1] = x[t] + dt * (beta * x_tau / (1 + x_tau ** n) - gamma * x[t])

    return x[max_delay + 1:]


mg_series = generate_mackey_glass(length=5000)
print("Serie Mackey-Glass generata, primi 10 valori:", mg_series[:10])



def build_regression_dataset(series, delays=[18, 12, 6, 0], horizon=6):
    X, Y = [], []
    max_delay = max(delays)
    for t in range(max_delay, len(series) - horizon):
        X.append([series[t - d] for d in delays])
        Y.append(series[t + horizon])
    return np.array(X), np.array(Y)


X_mg, Y_mg = build_regression_dataset(mg_series)
print("X shape:", X_mg.shape, "Y shape:", Y_mg.shape)
print("Esempio X[0]:", X_mg[0], "Y[0]:", Y_mg[0])


train_size = int(len(X_mg) * 0.7)
X_train_mg, X_test_mg = X_mg[:train_size], X_mg[train_size:]
Y_train_mg, Y_test_mg = Y_mg[:train_size], Y_mg[train_size:]

print("Training:", X_train_mg.shape, Y_train_mg.shape)
print("Test:", X_test_mg.shape, Y_test_mg.shape)


scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_Y = MinMaxScaler(feature_range=(-1, 1))


X_train_scaled = scaler_X.fit_transform(X_train_mg)
X_test_scaled = scaler_X.transform(X_test_mg)

Y_train_scaled = scaler_Y.fit_transform(Y_train_mg.reshape(-1, 1))
Y_test_scaled = scaler_Y.transform(Y_test_mg.reshape(-1, 1))


def to_complex(matrix: np.ndarray) -> List[List[complex]]:
    return [[complex(x, 0) for x in row] for row in matrix]

X_train_complex: List[List[complex]] = to_complex(X_train_scaled)
X_test_complex: List[List[complex]] = to_complex(X_test_scaled)
Y_train_complex: List[List[complex]] = [[complex(y[0], 0)] for y in Y_train_scaled]
Y_test_complex: List[List[complex]] = [[complex(y[0], 0)] for y in Y_test_scaled]

n_input = X_train_complex[0].__len__()
n_hidden = 30
n_output = 1

net = ComplexNetwork([n_input, 30, 20, n_output])
print(net)


history = net.train(X_train_complex, Y_train_complex, epochs=1000, errorThresholdToStop=1e-4)
print("✅ Training completato")


Y_pred_complex = [net.feedforward(x)[0] for x in X_test_complex]

Y_pred_scaled = np.array([y.real for y in Y_pred_complex]).reshape(-1, 1)
Y_true_scaled = np.array([y[0].real for y in Y_test_complex]).reshape(-1, 1)

Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)
Y_true_real = scaler_Y.inverse_transform(Y_true_scaled)


MSE = np.mean((Y_true_real - Y_pred_real)**2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(Y_true_real - Y_pred_real))
R2 = 1 - np.sum((Y_true_real - Y_pred_real)**2) / np.sum((Y_true_real - np.mean(Y_true_real))**2)

print(f"MSE: {MSE:.6f}, RMSE: {RMSE:.6f}, MAE: {MAE:.6f}, R²: {R2:.6f}")


print("Prime 5 predizioni vs valori reali:")
for i in range(5):
    print(f"Pred: {Y_pred_real[i][0]:.6f}, True: {Y_true_real[i][0]:.6f}")