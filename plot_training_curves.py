# plot_training_curves.py

import pickle
import matplotlib.pyplot as plt
import os

# ======================================
# LOAD TRAINING HISTORIES
# ======================================
MODEL_PATH = "outputs/models"
PLOT_PATH = "outputs/plots"

os.makedirs(PLOT_PATH, exist_ok=True)

print("Loading training histories...")

with open(os.path.join(MODEL_PATH, "baseline_history.pkl"), "rb") as f:
    baseline_history = pickle.load(f)

with open(os.path.join(MODEL_PATH, "attention_history.pkl"), "rb") as f:
    attention_history = pickle.load(f)

with open(os.path.join(MODEL_PATH, "cyclic_history.pkl"), "rb") as f:
    cyclic_history = pickle.load(f)

# ======================================
# PLOT 1: BASELINE LOSS CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(baseline_history["train_loss"], label='Train Loss', linewidth=2, marker='o')
plt.plot(baseline_history["val_loss"], label='Val Loss', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Baseline Model - Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "loss_curve_baseline.png"), dpi=300)
plt.close()

# ======================================
# PLOT 2: BASELINE RMSE CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(baseline_history["train_rmse"], label='Train RMSE', linewidth=2, marker='o')
plt.plot(baseline_history["val_rmse"], label='Val RMSE', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Baseline Model - Training & Validation RMSE', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "rmse_curve_baseline.png"), dpi=300)
plt.close()

# ======================================
# PLOT 3: ATTENTION LOSS CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(attention_history["train_loss"], label='Train Loss', linewidth=2, marker='o')
plt.plot(attention_history["val_loss"], label='Val Loss', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Attention Model - Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "loss_curve_attention.png"), dpi=300)
plt.close()

# ======================================
# PLOT 4: ATTENTION RMSE CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(attention_history["train_rmse"], label='Train RMSE', linewidth=2, marker='o')
plt.plot(attention_history["val_rmse"], label='Val RMSE', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Attention Model - Training & Validation RMSE', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "rmse_curve_attention.png"), dpi=300)
plt.close()

# ======================================
# PLOT 5: CYCLIC DUAL LOSS CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(cyclic_history["train_loss"], label='Train Loss', linewidth=2, marker='o')
plt.plot(cyclic_history["val_loss"], label='Val Loss', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Cyclic Dual Model - Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "loss_curve_cyclic.png"), dpi=300)
plt.close()

# ======================================
# PLOT 6: CYCLIC DUAL RMSE CURVE
# ======================================
plt.figure(figsize=(10, 6))
plt.plot(cyclic_history["train_rmse"], label='Train RMSE', linewidth=2, marker='o')
plt.plot(cyclic_history["val_rmse"], label='Val RMSE', linewidth=2, marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Cyclic Dual Model - Training & Validation RMSE', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "rmse_curve_cyclic.png"), dpi=300)
plt.close()

# ======================================
# PLOT 7: MODEL LOSS COMPARISON
# ======================================
plt.figure(figsize=(12, 7))
epochs_baseline = range(len(baseline_history["val_loss"]))
epochs_attention = range(len(attention_history["val_loss"]))
epochs_cyclic = range(len(cyclic_history["val_loss"]))

plt.plot(epochs_baseline, baseline_history["val_loss"], label='Baseline (Val)', linewidth=2.5, marker='o')
plt.plot(epochs_attention, attention_history["val_loss"], label='Attention (Val)', linewidth=2.5, marker='s')
plt.plot(epochs_cyclic, cyclic_history["val_loss"], label='Cyclic Dual (Val)', linewidth=2.5, marker='^')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss (MSE)', fontsize=12)
plt.title('Model Comparison - Validation Loss Across Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "model_loss_comparison.png"), dpi=300)
plt.close()

# ======================================
# PLOT 8: MODEL RMSE COMPARISON
# ======================================
# ======================================
# FINAL PAPER FIGURE:
# VALIDATION RMSE COMPARISON
# ======================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_baseline,
    baseline_history["val_rmse"],
    label="Neural CF",
    linewidth=2.5,
    marker="o"
)

plt.plot(
    epochs_attention,
    attention_history["val_rmse"],
    label="Metadata Attention",
    linewidth=2.5,
    marker="s"
)

plt.plot(
    epochs_cyclic,
    cyclic_history["val_rmse"],
    label="Cyclic Dual",
    linewidth=2.5,
    marker="^"
)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation RMSE", fontsize=12)

plt.title(
    "Model Comparison - Validation RMSE Across Epochs",
    fontsize=14,
    fontweight="bold"
)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_PATH, "final_validation_rmse_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Final RMSE comparison figure saved.")

# ======================================
# PRINT SUCCESS MESSAGE
# ======================================
print("All plots saved successfully in outputs/plots/")
