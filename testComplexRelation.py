import numpy as np
import pandas as pd
from network.ComplexNetwork import ComplexNetwork
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


train_file = "dataset_complesso.csv"
df = pd.read_csv(train_file, sep=',', header=0, dtype=str)

def str_to_complex(s):
    return complex(s.replace(' ', ''))

X = df.iloc[:, :-1].apply(lambda col: col.map(str_to_complex)).values
Y = df.iloc[:, -1].map(str_to_complex).values.reshape(-1, 1)

n_samples = X.shape[0]
split_index = int(n_samples * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

X_train_list = X_train.tolist()
Y_train_list = Y_train.tolist()

EPOCHS = 1000
ARCH   = [X.shape[1], 3, 1]
THRESHOLD = 1e-6

results = {}

for use_qr in [False, True]:
    label = "QR=True" if use_qr else "QR=False"
    print(f"\n--- Training with {label} ---")

    net = ComplexNetwork(ARCH, use_qr_batch=use_qr)

    start_time = time.perf_counter()
    history = net.train(
        X_train_list,
        Y_train_list,
        epochs=EPOCHS,
        errorThresholdToStop=THRESHOLD,
    )
    elapsed = time.perf_counter() - start_time
    print(f"Training time: {elapsed:.4f}s")

    Y_pred = np.array([net.feedforward(list(x)) for x in X_test]).reshape(-1, 1)
    errors = np.abs(Y_test - Y_pred)

    rmse      = np.sqrt(np.mean(errors ** 2))
    mae       = np.mean(errors)
    mse       = np.mean(errors ** 2)
    max_err   = np.max(errors)
    y_mean    = np.mean(np.abs(Y_test))
    ss_tot    = np.sum((np.abs(Y_test) - y_mean) ** 2)
    r2        = 1 - np.sum(errors ** 2) / ss_tot
    mape      = np.mean(errors / (np.abs(Y_test) + 1e-8)) * 100

    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  MSE={mse:.4f}  MaxErr={max_err:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    results[label] = {
        "history"  : history,
        "Y_pred"   : Y_pred,
        "errors"   : errors,
        "time"     : elapsed,
        "metrics"  : dict(RMSE=rmse, MAE=mae, MSE=mse, MaxErr=max_err, R2=r2, MAPE=mape),
    }


# ─────────────────────────────  PLOTS  ──────────────────────────────────────
labels   = list(results.keys())
colors   = {"QR=False": "#E05C5C", "QR=True": "#4C9BE8"}
n_test   = Y_test.shape[0]
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
bars = ax2.bar(labels, times, color=[colors[l] for l in labels], width=0.4)
ax2.set_title("Training Time (s)")
ax2.set_ylabel("Seconds")
for bar, val in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
ax2.grid(axis="y", alpha=0.3)


# ── 3. Predicted vs True (real part) ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(sample_idx, Y_test.real.flatten(), "k--", label="True (Re)", linewidth=1.2, alpha=0.7)
for lbl in labels:
    ax3.plot(sample_idx, results[lbl]["Y_pred"].real.flatten(),
             label=f"Pred {lbl} (Re)", color=colors[lbl], linewidth=1.2, alpha=0.85)
ax3.set_title("Predicted vs True — Real Part")
ax3.set_xlabel("Test Sample")
ax3.set_ylabel("Re(y)")
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


# ── 5. Metrics bar chart ──────────────────────────────────────────────────
metric_names = ["RMSE", "MAE", "MSE", "MaxErr", "MAPE"]
ax5 = fig.add_subplot(gs[2, :2])
x   = np.arange(len(metric_names))
width = 0.35
for i, lbl in enumerate(labels):
    vals = [results[lbl]["metrics"][m] for m in metric_names]
    ax5.bar(x + i * width, vals, width, label=lbl, color=colors[lbl], alpha=0.85)
ax5.set_xticks(x + width / 2)
ax5.set_xticklabels(metric_names)
ax5.set_title("Test Metrics Comparison")
ax5.set_yscale("log")
ax5.legend()
ax5.grid(axis="y", alpha=0.3)


# ── 6. R² Score bar ───────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
r2_vals = [results[lbl]["metrics"]["R2"] for lbl in labels]
bars6 = ax6.bar(labels, r2_vals, color=[colors[l] for l in labels], width=0.4)
ax6.set_title("R² Score")
ax6.set_ylabel("R²")
ax6.set_ylim(min(0, min(r2_vals)) - 0.05, 1.05)
for bar, val in zip(bars6, r2_vals):
    ax6.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005 * np.sign(bar.get_height()),
             f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax6.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax6.grid(axis="y", alpha=0.3)


plt.savefig("qr_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafico salvato come 'qr_comparison.png'")