import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from network.ComplexNetwork import ComplexNetwork
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


train_file = "dataset/Filter/sallen_keyBP_cartesian.csv"
df = pd.read_csv(train_file, sep=',', header=0, dtype=str)

# ── Parsing ───────────────────────────────────────────────────────────────────
# V1..V10 → complessi  |  X1..X7 → reali (variabili dipendenti / target)

def str_to_complex(s):
    s = s.replace(' ', '')
    s = re.sub(r'([+-])j([0-9eE.+-]+)', r'\1\2j', s)
    return complex(s)

X_cols = [c for c in df.columns if c.startswith('V')]   # V1..V10  (input)
Y_cols = [c for c in df.columns if c.startswith('X')]   # X1..X7   (target)

X = df[X_cols].apply(lambda col: col.map(str_to_complex)).values          # (N,10) complex
Y = df[Y_cols].apply(lambda col: pd.to_numeric(col)).values.astype(float) # (N,7)  real

n_samples = X.shape[0]
split_index = int(n_samples * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# ── StandardScaler per-colonna sui target ─────────────────────────────────────
# Fit solo su train → niente data leakage
scaler = StandardScaler()
Y_train_n = scaler.fit_transform(Y_train)

print("Normalizzazione target (StandardScaler):")
for k, (m, s) in enumerate(zip(scaler.mean_, scaler.scale_)):
    print(f"  X{k+1}: mean={m:.4e}  std={s:.4e}")

n_inputs  = X.shape[1]   # 10
n_targets = Y.shape[1]   # 7

# Architettura più profonda: più neuroni e più layer
# per catturare la relazione non lineare risposta-in-frequenza → parametri
ARCH      = [n_inputs, 32, 16, 8, n_targets]
EPOCHS    = 2000
THRESHOLD = 1e-6

X_train_list   = X_train.tolist()
X_test_list    = X_test.tolist()
Y_train_n_list = Y_train_n.tolist()

results = {}

for use_qr in [False, True]:
    label = "QR=True" if use_qr else "QR=False"
    print(f"\n{'='*55}")
    print(f"  Training with {label}  —  arch={ARCH}")
    print(f"{'='*55}")

    net = ComplexNetwork(ARCH, use_qr_batch=use_qr)

    t_start = time.perf_counter()
    history = net.train(
        X_train_list,
        Y_train_n_list,
        epochs=EPOCHS,
        errorThresholdToStop=THRESHOLD,
    )
    elapsed = time.perf_counter() - t_start

    # Predici normalizzato → de-normalizza nello spazio originale
    Y_pred_n   = np.array([net.feedforward(list(x)) for x in X_test_list]).real  # (n_test, 7)
    Y_pred_mat = scaler.inverse_transform(Y_pred_n)
    errors_mat = np.abs(Y_test - Y_pred_mat)

    # Metriche per target
    for k in range(n_targets):
        rmse_k = np.sqrt(np.mean(errors_mat[:, k]**2))
        mape_k = np.mean(errors_mat[:, k] / (np.abs(Y_test[:, k]) + 1e-12)) * 100
        print(f"  X{k+1}: RMSE={rmse_k:.4e}  MAPE={mape_k:.1f}%")

    # Metriche globali
    rmse    = np.sqrt(np.mean(errors_mat**2))
    mae     = np.mean(errors_mat)
    mse     = np.mean(errors_mat**2)
    max_err = np.max(errors_mat)
    y_mean_ = np.mean(np.abs(Y_test), axis=0)
    ss_tot  = np.sum((np.abs(Y_test) - y_mean_)**2)
    r2      = 1 - np.sum(errors_mat**2) / ss_tot
    mape    = np.mean(errors_mat / (np.abs(Y_test) + 1e-12)) * 100

    print(f"\n  [Globale] RMSE={rmse:.4e}  MAE={mae:.4e}  R²={r2:.4f}  MAPE={mape:.2f}%  time={elapsed:.2f}s")

    results[label] = {
        "history" : history,
        "Y_pred"  : Y_pred_mat,
        "errors"  : errors_mat,
        "time"    : elapsed,
        "metrics" : dict(RMSE=rmse, MAE=mae, MSE=mse, MaxErr=max_err, R2=r2, MAPE=mape),
    }


# ─────────────────────────────  PLOTS  ───────────────────────────────────────
labels       = list(results.keys())
colors       = {"QR=False": "#E05C5C", "QR=True": "#4C9BE8"}
n_test       = Y_test.shape[0]
sample_idx   = np.arange(n_test)
target_names = [f"X{k+1}" for k in range(n_targets)]

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f"ComplexNetwork — 7 Regressori (rete unica)  arch={ARCH}  |  QR: True vs False",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── 1. Loss curves ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for lbl in labels:
    h = results[lbl]["history"]
    if h:
        ax1.plot(h, label=lbl, color=colors[lbl], linewidth=1.5)
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_yscale("log"); ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# ── 2. Training time bar ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
times = [results[lbl]["time"] for lbl in labels]
bars  = ax2.bar(labels, times, color=[colors[l] for l in labels], width=0.4)
ax2.set_title("Training Time (s)"); ax2.set_ylabel("Seconds")
for bar, val in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

# ── 3. R² Score ───────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 3])
r2_vals = [results[lbl]["metrics"]["R2"] for lbl in labels]
bars3   = ax3.bar(labels, r2_vals, color=[colors[l] for l in labels], width=0.4)
ax3.set_title("R² Score"); ax3.set_ylabel("R²")
ax3.set_ylim(min(0, min(r2_vals)) - 0.05, 1.05)
for bar, val in zip(bars3, r2_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             max(bar.get_height(), 0) + 0.01, f"{val:.4f}",
             ha="center", va="bottom", fontsize=10)
ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax3.grid(axis="y", alpha=0.3)

# ── 4. Predicted vs True per ogni target ─────────────────────────────────────
for k in range(n_targets):
    row = 1 + k // 4
    col = k % 4
    ax = fig.add_subplot(gs[row, col])
    ax.plot(sample_idx, Y_test[:, k], "k--", linewidth=1.0, alpha=0.6, label="True")
    for lbl in labels:
        ax.plot(sample_idx, results[lbl]["Y_pred"][:, k],
                color=colors[lbl], linewidth=1.0, alpha=0.85, label=lbl)
    ax.set_title(f"Pred vs True — {target_names[k]}", fontsize=9)
    ax.set_xlabel("Sample", fontsize=7); ax.set_ylabel("Value", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# ── 5. RMSE per target ───────────────────────────────────────────────────────
ax_rmse = fig.add_subplot(gs[3, :2])
x     = np.arange(n_targets)
width = 0.35
for i, lbl in enumerate(labels):
    rmse_per = np.sqrt(np.mean(results[lbl]["errors"]**2, axis=0))
    ax_rmse.bar(x + i * width, rmse_per, width, label=lbl, color=colors[lbl], alpha=0.85)
ax_rmse.set_xticks(x + width / 2)
ax_rmse.set_xticklabels(target_names)
ax_rmse.set_title("RMSE per Target")
ax_rmse.set_ylabel("RMSE"); ax_rmse.legend()
ax_rmse.set_yscale("log")
ax_rmse.grid(axis="y", alpha=0.3)

# ── 6. Metriche globali ───────────────────────────────────────────────────────
ax_m = fig.add_subplot(gs[3, 2:])
metric_names = ["RMSE", "MAE", "MSE", "MaxErr", "MAPE"]
x2 = np.arange(len(metric_names))
for i, lbl in enumerate(labels):
    vals = [results[lbl]["metrics"][m] for m in metric_names]
    ax_m.bar(x2 + i * 0.35, vals, 0.35, label=lbl, color=colors[lbl], alpha=0.85)
ax_m.set_xticks(x2 + 0.175)
ax_m.set_xticklabels(metric_names, fontsize=8)
ax_m.set_title("Metriche Globali")
ax_m.set_yscale("log"); ax_m.legend(fontsize=8)
ax_m.grid(axis="y", alpha=0.3)

plt.savefig("qr_comparison_7outputs.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → qr_comparison_7outputs.png")