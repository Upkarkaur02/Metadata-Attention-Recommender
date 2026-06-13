import numpy as np
import pandas as pd
import os
from sklearn.metrics import ndcg_score
from tensorflow.keras.models import load_model

# ======================================
# UTILITY FUNCTIONS FOR COLUMN DETECTION
# ======================================
def detect_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"None of {possible_names} found in columns: {df.columns.tolist()}")

def get_column_mapping(df, column_type):
    mappings = {
        'user': ['user', 'user_id', 'userId', 'User', 'USER_ID'],
        'item': ['item', 'movie_id', 'movieId', 'movie', 'Item', 'Movie', 'MOVIE_ID'],
        'rating': ['rating', 'actual_rating', 'true_rating', 'Rating', 'RATING'],
        'predicted': ['predicted_rating', 'prediction', 'pred_rating', 'Prediction', 'predicted_score']
    }

    if column_type not in mappings:
        raise ValueError(f"Unknown column type: {column_type}")

    return detect_column(df, mappings[column_type])

# ======================================
# LOAD DATA
# ======================================
print("Loading datasets and detecting column names...\n")

MODEL_PATH = "outputs/models"
DATA_PATH = "processed_data"

test_df = pd.read_csv(os.path.join(DATA_PATH, "test_warm.csv"))

print("Test data columns:", test_df.columns.tolist())

# Encoded columns for model input
encoded_user_col = "user"
encoded_item_col = "item"

# Original columns for reporting/evaluation
raw_user_col = get_column_mapping(test_df, 'user')
raw_item_col = get_column_mapping(test_df, 'item')
rating_col = get_column_mapping(test_df, 'rating')

print(f"Encoded user column used for model input: {encoded_user_col}")
print(f"Encoded item column used for model input: {encoded_item_col}")
print(f"Raw user column used for evaluation: {raw_user_col}")
print(f"Raw item column used for evaluation: {raw_item_col}")
print(f"Rating column used for evaluation: {rating_col}")
print()

# ======================================
# LOAD TRAINED MODELS
# ======================================
print("Loading trained models...\n")

try:
    baseline_model = load_model(
        os.path.join(MODEL_PATH, "baseline_model.h5"),
        compile=False
    )
    baseline_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✓ Baseline model loaded")
except Exception as e:
    print(f"✗ Could not load baseline model: {e}")
    baseline_model = None

try:
    attention_model = load_model(
        os.path.join(MODEL_PATH, "attention_model.keras"),
        compile=False
    )
    attention_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✓ Attention model loaded")
except Exception as e:
    print(f"✗ Could not load attention model: {e}")
    attention_model = None

try:
    cyclic_model = load_model(
        os.path.join(MODEL_PATH, "cyclic_dual_model.h5"),
        compile=False
    )
    cyclic_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✓ Cyclic Dual model loaded")
except Exception as e:
    print(f"✗ Could not load cyclic dual model: {e}")
    cyclic_model = None

print()

# ======================================
# LOAD MOVIE FEATURES
# ======================================
print("Loading metadata features...\n")

movie_features = pd.read_csv(os.path.join(DATA_PATH, "movie_features.csv"))

movie_features = movie_features.sort_values("item").reset_index(drop=True)

feature_columns = [
    col for col in movie_features.columns
    if col not in ["movieId", "item"]
]

metadata_matrix = movie_features[feature_columns].values

print(f"Metadata matrix shape: {metadata_matrix.shape}")
print()

def get_metadata_features(df, item_col="item"):
    item_indices = df[item_col].values.astype(int)

    # Keep indices within valid range
    item_indices = np.clip(item_indices, 0, metadata_matrix.shape[0] - 1)

    return metadata_matrix[item_indices]

# ======================================
# GENERATE PREDICTIONS
# ======================================
print("Generating predictions...\n")

baseline_results = None
attention_results = None
cyclic_results = None

# -------------------------
# Baseline Predictions
# -------------------------
if baseline_model is not None:
    baseline_predictions = baseline_model.predict(
        [
            test_df[encoded_user_col].values,
            test_df[encoded_item_col].values
        ],
        batch_size=1024,
        verbose=0
    ).flatten()

    baseline_results = test_df.copy()
    baseline_results["predicted_rating"] = baseline_predictions

    baseline_results.to_csv(
        os.path.join(MODEL_PATH, "baseline_results_with_predictions.csv"),
        index=False
    )

    print("✓ Baseline predictions generated")

# -------------------------
# Attention Predictions
# -------------------------
if attention_model is not None:
    X_test_metadata = get_metadata_features(test_df, encoded_item_col)

    attention_predictions = attention_model.predict(
        [
            test_df[encoded_user_col].values,
            test_df[encoded_item_col].values,
            X_test_metadata
        ],
        batch_size=1024,
        verbose=0
    ).flatten()

    attention_results = test_df.copy()
    attention_results["predicted_rating"] = attention_predictions

    attention_results.to_csv(
        os.path.join(MODEL_PATH, "attention_results_with_predictions.csv"),
        index=False
    )

    print("✓ Attention predictions generated")

# -------------------------
# Cyclic Dual Predictions
# -------------------------
if cyclic_model is not None:
    X_test_metadata = get_metadata_features(test_df, encoded_item_col)

    cyclic_predictions = cyclic_model.predict(
        [
            test_df[encoded_user_col].values,
            test_df[encoded_item_col].values,
            X_test_metadata
        ],
        batch_size=1024,
        verbose=0
    ).flatten()

    cyclic_results = test_df.copy()
    cyclic_results["predicted_rating"] = cyclic_predictions

    cyclic_results.to_csv(
        os.path.join(MODEL_PATH, "cyclic_dual_results_with_predictions.csv"),
        index=False
    )

    print("✓ Cyclic Dual predictions generated")

print()

# ======================================
# TOP-K METRIC FUNCTION
# ======================================
def calculate_topk_metrics(
    results_df,
    user_col,
    item_col,
    rating_col,
    predicted_col="predicted_rating",
    k=10,
    relevance_threshold=4.0
):
    precisions = []
    recalls = []
    hit_ratios = []
    ndcgs = []

    users = results_df[user_col].unique()

    for user in users:
        user_df = results_df[results_df[user_col] == user]

        relevant_items = set(
            user_df[user_df[rating_col] >= relevance_threshold][item_col]
        )

        if len(relevant_items) == 0:
            continue

        top_k = user_df.sort_values(
            by=predicted_col,
            ascending=False
        ).head(k)

        recommended_items = list(top_k[item_col])

        hits = [1 if item in relevant_items else 0 for item in recommended_items]

        precision = np.sum(hits) / k
        recall = np.sum(hits) / len(relevant_items)
        hit_ratio = 1 if np.sum(hits) > 0 else 0

        dcg_scores = np.array(hits)
        ideal_scores = np.sort(dcg_scores)[::-1]

        # NDCG requires at least 2 items
        if len(dcg_scores) > 1 and np.sum(ideal_scores) > 0:
            ndcg = ndcg_score([ideal_scores], [dcg_scores])
        else:
            ndcg = 0.0
            
        precisions.append(precision)
        recalls.append(recall)
        hit_ratios.append(hit_ratio)
        ndcgs.append(ndcg)

    return {
        "Precision@10": np.mean(precisions) if precisions else 0.0,
        "Recall@10": np.mean(recalls) if recalls else 0.0,
        "HitRatio@10": np.mean(hit_ratios) if hit_ratios else 0.0,
        "NDCG@10": np.mean(ndcgs) if ndcgs else 0.0
    }

# ======================================
# CALCULATE METRICS
# ======================================
print("Calculating Top-K metrics...\n")

results_list = []

if baseline_results is not None:
    baseline_metrics = calculate_topk_metrics(
        baseline_results,
        raw_user_col,
        raw_item_col,
        rating_col
    )
    results_list.append({
        "Model": "Baseline",
        **baseline_metrics
    })
    print("✓ Baseline metrics calculated")

if attention_results is not None:
    attention_metrics = calculate_topk_metrics(
        attention_results,
        raw_user_col,
        raw_item_col,
        rating_col
    )
    results_list.append({
        "Model": "Attention",
        **attention_metrics
    })
    print("✓ Attention metrics calculated")

if cyclic_results is not None:
    cyclic_metrics = calculate_topk_metrics(
        cyclic_results,
        raw_user_col,
        raw_item_col,
        rating_col
    )
    results_list.append({
        "Model": "Cyclic Dual",
        **cyclic_metrics
    })
    print("✓ Cyclic Dual metrics calculated")

print()

# ======================================
# SAVE FINAL METRICS TABLE
# ======================================
metrics_df = pd.DataFrame(results_list)

print("Top-K Metrics Comparison:")
print(metrics_df)

metrics_output_path = os.path.join(MODEL_PATH, "topk_metrics_comparison.csv")
metrics_df.to_csv(metrics_output_path, index=False)

print(f"\nMetrics saved to: {metrics_output_path}")