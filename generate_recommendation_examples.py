import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load data
movies = pd.read_csv("data/movies.csv")
test_df = pd.read_csv("processed_data/test_warm.csv")

# Load trained attention model
model = load_model("outputs/models/attention_model.keras", compile=False)

# Choose a few sample users
sample_users = test_df["userId"].drop_duplicates().sample(5, random_state=42).tolist()
# Get unique movie list
all_movies = test_df[["movieId", "item"]].drop_duplicates()

recommendation_rows = []

for user in sample_users:

    # Encoded user ID
    user_rows = test_df[test_df["userId"] == user]

    if len(user_rows) == 0:
        continue

    user_encoded = user_rows["user"].iloc[0]

    candidate_movies = all_movies.copy()
    candidate_movies["user"] = user_encoded

    # Dummy metadata input if needed
    movie_features = pd.read_csv("processed_data/movie_features.csv")
    movie_features = movie_features.sort_values("item")

    feature_columns = [
        col for col in movie_features.columns
        if col not in ["movieId", "item"]
    ]

    metadata_matrix = movie_features[feature_columns].values

    item_indices = candidate_movies["item"].values.astype(int)
    item_indices = np.clip(item_indices, 0, metadata_matrix.shape[0] - 1)

    metadata_input = metadata_matrix[item_indices]

    predictions = model.predict(
        [
            candidate_movies["user"].values,
            candidate_movies["item"].values,
            metadata_input
        ],
        verbose=0
    ).flatten()

    candidate_movies["predicted_rating"] = predictions

    top_movies = candidate_movies.sort_values(
        by="predicted_rating",
        ascending=False
    ).head(5)

    top_movies = top_movies.merge(
        movies,
        on="movieId",
        how="left"
    )

    # Actual highly rated movies
    actual_movies = test_df[
        (test_df["userId"] == user) &
        (test_df["rating"] >= 4.0)
    ][["movieId"]].drop_duplicates()

    actual_movies = actual_movies.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendation_rows.append({
        "User": user,
        "Top Recommendations": ", ".join(top_movies["title"].tolist()),
        "Actual Highly Rated Movies": ", ".join(actual_movies["title"].head(5).tolist())
    })

recommendation_df = pd.DataFrame(recommendation_rows)

print(recommendation_df)

recommendation_df.to_csv(
    "outputs/models/recommendation_examples.csv",
    index=False
)