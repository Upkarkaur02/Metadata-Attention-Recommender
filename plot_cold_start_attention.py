import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load cold-start movie list
import pandas as pd

cold_items_df = pd.read_csv("processed_data/cold_items.csv")

print(cold_items_df.columns)
print(cold_items_df.head())
# Load top attention features
attention_df = pd.read_csv(
    "outputs/attention_results/top_attention_features.csv"
)

# Merge only cold-start movies
cold_attention_df = attention_df[
    attention_df["item"].isin(cold_items_df["cold_item"])
]
# Top feature columns
feature_cols = [
    "top_feature_1",
    "top_feature_2",
    "top_feature_3",
    "top_feature_4",
    "top_feature_5"
]

# Collect all top features for cold-start movies
all_features = []

for col in feature_cols:
    all_features.extend(
        cold_attention_df[col].dropna().astype(int).tolist()
    )

# Count frequency of each feature
feature_counts = Counter(all_features)

# Convert to dataframe
feature_count_df = pd.DataFrame(
    feature_counts.items(),
    columns=["feature_id", "count"]
)

# Sort descending
feature_count_df = feature_count_df.sort_values(
    by="count",
    ascending=False
)

# Keep top 10
top_features = feature_count_df.head(10)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature_id"].astype(str)[::-1],
    top_features["count"][::-1]
)

plt.xlabel("Frequency in Cold-Start Attention Features")
plt.ylabel("Feature ID")
plt.title("Most Important Features for Cold-Start Movies")
plt.tight_layout()

plt.savefig("outputs/plots/cold_start_attention.png")
plt.close()

print("Cold-start attention plot saved to outputs/plots/cold_start_attention.png")
print(top_features)