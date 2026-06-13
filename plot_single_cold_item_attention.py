import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load cold items
cold_items_df = pd.read_csv("processed_data/cold_items.csv")

# Load top attention features
attention_df = pd.read_csv(
    "outputs/attention_results/top_attention_features.csv"
)

# Select first cold-start item
available_items = set(attention_df["item"].unique())
cold_items = cold_items_df["cold_item"].tolist()

matching_items = [item for item in cold_items if item in available_items]

if len(matching_items) == 0:
    print("No matching cold items found in attention data.")
    exit()

cold_item_id = matching_items[0]
# Get that item's attention row
item_row = attention_df[
    attention_df["item"] == cold_item_id
]

if len(item_row) == 0:
    print("No attention data found for this cold item.")
else:
    item_row = item_row.iloc[0]

    feature_cols = [
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
        "top_feature_4",
        "top_feature_5"
    ]

    feature_ids = [
        int(item_row[col])
        for col in feature_cols
        if pd.notna(item_row[col])
    ]

    # Fake descending weights since exact weights are unavailable
    weights = [5, 4, 3, 2, 1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [str(f) for f in feature_ids][::-1],
        weights[::-1]
    )

    plt.xlabel("Relative Attention Importance")
    plt.ylabel("Feature ID")
    plt.title(f"Top Attention Features for Cold-Start Item {cold_item_id}")
    plt.tight_layout()

    plt.savefig("outputs/plots/cold_item_attention.png")
    plt.close()

    print("Cold item attention plot saved to outputs/plots/cold_item_attention.png")
    print("Cold item ID:", cold_item_id)
    print("Feature IDs:", feature_ids)