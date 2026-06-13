import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load top attention features file
attention_df = pd.read_csv(
    "outputs/attention_results/top_attention_features.csv"
)

# Collect all top feature columns
feature_cols = [
    "top_feature_1",
    "top_feature_2",
    "top_feature_3",
    "top_feature_4",
    "top_feature_5"
]

# Flatten all feature values into one list
all_features = []

for col in feature_cols:
    all_features.extend(attention_df[col].dropna().astype(int).tolist())

# Count frequency of each feature
feature_counts = Counter(all_features)

# Convert to dataframe
feature_count_df = pd.DataFrame(
    feature_counts.items(),
    columns=["feature_id", "count"]
)

# Sort by most frequent
feature_count_df = feature_count_df.sort_values(
    by="count",
    ascending=False
)

# Select top 10 most frequent features
top_k = 10
top_features = feature_count_df.head(top_k).copy()

# Plot
plt.figure(figsize=(10, 6))
label_mapping = {
    807: "surprisingly",
    446: "predictable",
    867: "social commentary",
    469: "transformation",
    787: "great movie",
    86: "culture clash",
    992: "adaptation",
    933: "talky",
    277: "oscar (best picture)",
    464: "plot"
}

top_features["feature_name"] = (
    top_features["feature_id"]
    .map(label_mapping)
    .fillna(top_features["feature_id"].astype(str))
)

print(top_features[["feature_id", "feature_name"]])
plt.barh(
    top_features["feature_name"][::-1],
    top_features["count"][::-1]
)

plt.xlabel("Frequency in Top Attention Features")
plt.ylabel("Metadata Feature")
plt.title("Most Important Metadata Features Based on Attention")
plt.tight_layout()

plt.savefig("outputs/plots/feature_importance.png")
plt.close()

print("Feature importance plot saved to outputs/plots/feature_importance.png")
print(top_features)