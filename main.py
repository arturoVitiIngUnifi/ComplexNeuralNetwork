import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from network.ComplexNetwork import ComplexNetwork
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

train_file = "dataset/train_def.csv"
test_file  = "dataset/test_def.csv"

df_train = pd.read_csv(train_file, sep=';', header=None, dtype=str)
df_test  = pd.read_csv(test_file,  sep=';', header=None, dtype=str)

def clean_numeric_string(s):
    s = str(s).strip()
    if '.' in s:
        first, *rest = s.split('.')
        s = first + '.' + ''.join(rest)
    return s

for col in df_train.columns:
    df_train[col] = df_train[col].apply(clean_numeric_string)
    df_test[col]  = df_test[col].apply(clean_numeric_string)

X_train_raw = df_train.iloc[:, :50].astype(np.float64).values
Y_train_raw = df_train.iloc[:, 50].astype(np.float64).values.reshape(-1, 1)
X_test_raw  = df_test.iloc[:,  :50].astype(np.float64).values
Y_test      = df_test.iloc[:,  50].astype(np.float64).values.reshape(-1, 1)

X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train      = X_scaler.fit_transform(X_train_raw)
X_test       = X_scaler.transform(X_test_raw)
Y_train      = Y_scaler.fit_transform(Y_train_raw)
Y_test_scaled = Y_scaler.transform(Y_test)

EPOCHS    = 1000
THRESHOLD = 1e-6
ARCH      = [50, 30, 1]

results = {}

for use_qr in [False, True]:
    label = "QR=True" if use_qr else "QR=False"
    print(f"\n--- Training with {label} ---")

    net = ComplexNetwork(ARCH, use_qr_batch=use_qr)

    start = time.perf_counter()
    history = net.train(
        X_train,
        Y_train,
        epochs=EPOCHS,
        errorThresholdToStop=THRESHOLD,
    )
    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.4f}s")

    Y_pred_scaled = np.array([
        net.feedforward(x)[0].real for x in X_test
    ]).reshape(-1, 1)

    Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)

    errors = np.abs(Y_test - Y_pred)
    mse    = np.mean((Y_test - Y_pred) ** 2)
    rmse   = np.sqrt(mse)
    mae    = np.mean(errors)
    max_err = np.max(errors)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2     = 1 - np.sum((Y_test - Y_pred) ** 2) / ss_tot
    mape   = np.mean(errors / (np.abs(Y_test) + 1e-8)) * 100

    print(f"  MSE={mse:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}  R²={r2:.6f}  MAPE={mape:.2f}%")

    results[label] = {
        "history" : history,
        "Y_pred"  : Y_pred,
        "errors"  : errors,
        "time"    : elapsed,
        "metrics" : dict(MSE=mse, RMSE=rmse, MAE=mae, MaxErr=max_err, R2=r2, MAPE=mape),
    }


# ─────────────────────────────  PLOTS  ──────────────────────────────────────
labels    = list(results.keys())
colors    = {"QR=False": "#E05C5C", "QR=True": "#4C9BE8"}
n_test    = Y_test.shape[0]
sample_idx = np.arange(n_test)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("ComplexNetwork — QR Decomposition: True vs False", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── 1. Loss curves ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for lbl in labels:
    h = results[lbl]["history"]
    if h:
        ax1.plot(h, label=lbl, color=colors[lbl], linewidth=1.5)
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)


# ── 2. Training time bar ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
times = [results[lbl]["time"] for lbl in labels]
bars  = ax2.bar(labels, times, color=[colors[l] for l in labels], width=0.4)
ax2.set_title("Training Time (s)")
ax2.set_ylabel("Seconds")
for bar, val in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
ax2.grid(axis="y", alpha=0.3)


# ── 3. Predicted vs True ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(sample_idx, Y_test.flatten(), "k--", label="True", linewidth=1.2, alpha=0.7)
for lbl in labels:
    ax3.plot(sample_idx, results[lbl]["Y_pred"].flatten(),
             label=f"Pred {lbl}", color=colors[lbl], linewidth=1.2, alpha=0.85)
ax3.set_title("Predicted vs True (original scale)")
ax3.set_xlabel("Test Sample")
ax3.set_ylabel("y")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)


# ── 4. Absolute error per sample ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
for lbl in labels:
    ax4.plot(sample_idx, results[lbl]["errors"].flatten(),
             label=lbl, color=colors[lbl], linewidth=1.1, alpha=0.85)
ax4.set_title("Absolute Error per Test Sample")
ax4.set_xlabel("Test Sample")
ax4.set_ylabel("|y_true - y_pred|")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)


# ── 5. Scatter: True vs Predicted ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
y_flat = Y_test.flatten()
for lbl in labels:
    ax5.scatter(y_flat, results[lbl]["Y_pred"].flatten(),
                label=lbl, color=colors[lbl], alpha=0.5, s=15)
lims = [y_flat.min(), y_flat.max()]
ax5.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect fit")
ax5.set_title("Scatter: True vs Predicted")
ax5.set_xlabel("True y")
ax5.set_ylabel("Predicted y")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)


# ── 6. Metrics bar chart ──────────────────────────────────────────────────
metric_names = ["MSE", "RMSE", "MAE", "MaxErr", "MAPE"]
ax6 = fig.add_subplot(gs[2, 2])
x     = np.arange(len(metric_names))
width = 0.35
for i, lbl in enumerate(labels):
    vals = [results[lbl]["metrics"][m] for m in metric_names]
    ax6.bar(x + i * width, vals, width, label=lbl, color=colors[lbl], alpha=0.85)
ax6.set_xticks(x + width / 2)
ax6.set_xticklabels(metric_names, fontsize=8)
ax6.set_title("Test Metrics Comparison")
ax6.set_yscale("log")
ax6.legend(fontsize=8)
ax6.grid(axis="y", alpha=0.3)


plt.savefig("qr_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafico salvato come 'qr_comparison.png'")