import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# LOAD METRICS
# ======================================
metrics_df = pd.read_csv(
    "outputs/models/topk_metrics_comparison.csv"
)

# Metrics to display
metrics = [
    "Precision@10",
    "Recall@10",
    "HitRatio@10",
    "NDCG@10"
]

# Model names
models = metrics_df["Model"].tolist()

# ======================================
# PLOT SETTINGS
# ======================================
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))

# ======================================
# GROUPED BAR CHART
# ======================================
for i, model in enumerate(models):

    values = metrics_df.loc[i, metrics].values

    bars = plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model
    )

    # Add value labels
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.3f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

# ======================================
# FORMATTING
# ======================================
plt.xticks(x, metrics, fontsize=11)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Evaluation Metric", fontsize=12)
plt.title(
    "Grouped Top-N Recommendation Metrics Comparison",
    fontsize=14,
    fontweight='bold'
)

plt.legend(title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# ======================================
# SAVE FIGURE
# ======================================
plt.savefig(
    "outputs/plots/topk_metrics_grouped_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "Grouped metrics plot saved to "
    "outputs/plots/topk_metrics_grouped_comparison.png"
)