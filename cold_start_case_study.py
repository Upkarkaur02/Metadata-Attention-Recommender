import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ======================================
# LOAD DATA
# ======================================
cold_df = pd.read_csv("processed_data/test_cold.csv")
movie_features = pd.read_csv("processed_data/movie_features.csv")

# Replace with actual path if needed
movies = pd.read_csv("data/movies.csv")

# Load trained attention model
model = load_model("outputs/models/attention_model.keras", compile=False)

# Sort movie features by encoded item ID
movie_features = movie_features.sort_values("item").reset_index(drop=True)

feature_columns = [
    col for col in movie_features.columns
    if col not in ["movieId", "item"]
]

metadata_matrix = movie_features[feature_columns].values

# ======================================
# SELECT SAMPLE COLD-START MOVIES
# ======================================
cold_movies = (
    cold_df[["movieId", "item"]]
    .drop_duplicates()
    .sample(5, random_state=42)
)

results = []

for _, row in cold_movies.iterrows():

    movie_id = row["movieId"]
    item_encoded = int(row["item"])

    # Get movie title
    movie_row = movies[movies["movieId"] == movie_id]

    if len(movie_row) == 0:
        continue

    title = movie_row["title"].iloc[0]
    genres = movie_row["genres"].iloc[0]

    # Metadata vector
    item_encoded = min(item_encoded, metadata_matrix.shape[0] - 1)
    metadata_vector = metadata_matrix[item_encoded]

    # Get top metadata features
    top_feature_indices = np.argsort(metadata_vector)[::-1][:5]
    top_features = [feature_columns[i] for i in top_feature_indices]

    # Choose a few users
    sample_users = cold_df["user"].drop_duplicates().sample(10, random_state=42)

    predicted_scores = []

    for user_encoded in sample_users:

        prediction = model.predict(
            [
                np.array([user_encoded]),
                np.array([item_encoded]),
                np.array([metadata_vector])
            ],
            verbose=0
        ).flatten()[0]

        predicted_scores.append((user_encoded, prediction))

    predicted_scores = sorted(predicted_scores, key=lambda x: x[1], reverse=True)

    top_users = predicted_scores[:3]

    results.append({
        "Movie Title": title,
        "Genres": genres,
        "Top Metadata Features": ", ".join(top_features),
        "Top Recommended Users": ", ".join([str(x[0]) for x in top_users]),
        "Predicted Ratings": ", ".join([f"{x[1]:.2f}" for x in top_users])
    })

case_study_df = pd.DataFrame(results)

print(case_study_df)

case_study_df.to_csv(
    "outputs/models/cold_start_case_study.csv",
    index=False
)