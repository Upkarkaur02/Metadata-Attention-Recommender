# improved_preprocessing.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ======================================
# CONFIG
# ======================================
DATA_PATH = "data"
OUTPUT_PATH = "processed_data"
PLOT_PATH = "outputs/plots"

MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 50
MAX_USERS = 8000
TOP_GENOME_TAGS = 100
COLD_ITEM_FRACTION = 0.10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# ======================================
# 1. LOAD DATA
# ======================================
print("Loading data...")

ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
genome_scores = pd.read_csv(os.path.join(DATA_PATH, "genome-scores.csv"))
genome_tags = pd.read_csv(os.path.join(DATA_PATH, "genome-tags.csv"))
tags = pd.read_csv(os.path.join(DATA_PATH, "tags.csv"))

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Genome scores shape:", genome_scores.shape)

# ======================================
# 2. INITIAL VISUALIZATION
# ======================================
print("Creating raw data plots...")

user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()

plt.figure(figsize=(8,5))
plt.hist(user_counts, bins=50)
plt.title("Raw User Activity Distribution")
plt.xlabel("Ratings per User")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "raw_user_activity.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(movie_counts, bins=50)
plt.title("Raw Movie Popularity Distribution")
plt.xlabel("Ratings per Movie")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "raw_movie_popularity.png"))
plt.close()

# ======================================
# 3. FILTER USERS
# ======================================
print("Filtering users...")

valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
ratings = ratings[ratings['userId'].isin(valid_users)]

print("Remaining users:", ratings['userId'].nunique())

# ======================================
# 4. FILTER MOVIES
# ======================================
print("Filtering movies...")

movie_counts = ratings['movieId'].value_counts()
valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
ratings = ratings[ratings['movieId'].isin(valid_movies)]

print("Remaining movies:", ratings['movieId'].nunique())

# ======================================
# 5. KEEP MOVIES AVAILABLE IN GENOME SCORES
# ======================================
print("Aligning with genome data...")

valid_genome_movies = genome_scores['movieId'].unique()
ratings = ratings[ratings['movieId'].isin(valid_genome_movies)]
movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]

# ======================================
# 6. SAMPLE USERS
# ======================================
print("Sampling users...")

unique_users = ratings['userId'].unique()

if len(unique_users) > MAX_USERS:
    sampled_users = np.random.choice(unique_users, MAX_USERS, replace=False)
    ratings = ratings[ratings['userId'].isin(sampled_users)]

print("Sampled users:", ratings['userId'].nunique())

# ======================================
# 7. CREATE GENRE FEATURES
# ======================================
print("Creating genre features...")

movies['genres'] = movies['genres'].fillna('(no genres listed)')

movie_genres = movies[['movieId', 'genres']].copy()
genre_features = movie_genres['genres'].str.get_dummies(sep='|')

genre_matrix = pd.concat(
    [movie_genres[['movieId']], genre_features],
    axis=1
)

print("Genre feature shape:", genre_matrix.shape)

# ======================================
# 8. SELECT TOP GENOME TAGS
# ======================================
print("Selecting top genome tags...")

mean_tag_relevance = genome_scores.groupby('tagId')['relevance'].mean()
top_tag_ids = mean_tag_relevance.sort_values(ascending=False).head(TOP_GENOME_TAGS).index

genome_scores = genome_scores[
    genome_scores['tagId'].isin(top_tag_ids)
]

genome_scores = genome_scores[
    genome_scores['movieId'].isin(ratings['movieId'].unique())
]

print("Selected genome tags:", len(top_tag_ids))

# ======================================
# 9. CREATE GENOME FEATURE MATRIX
# ======================================
print("Creating genome feature matrix...")

genome_matrix = genome_scores.pivot(
    index='movieId',
    columns='tagId',
    values='relevance'
).fillna(0)

genome_matrix.reset_index(inplace=True)

print("Genome matrix shape:", genome_matrix.shape)

# ======================================
# 10. MERGE GENRES + GENOME FEATURES
# ======================================
print("Creating final movie feature matrix...")

movie_feature_matrix = genre_matrix.merge(
    genome_matrix,
    on='movieId',
    how='left'
).fillna(0)

print("Final movie feature matrix shape:", movie_feature_matrix.shape)

# ======================================
# 11. ENCODE USERS AND MOVIES
# ======================================
print("Encoding IDs...")

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['userId'])
ratings['item'] = movie_encoder.fit_transform(ratings['movieId'])

# Keep only movies that exist in encoded ratings
movie_feature_matrix = movie_feature_matrix[
    movie_feature_matrix['movieId'].isin(movie_encoder.classes_)
].copy()

movie_feature_matrix['item'] = movie_encoder.transform(movie_feature_matrix['movieId'])

num_users = ratings['user'].nunique()
num_items = ratings['item'].nunique()

print("Encoded users:", num_users)
print("Encoded items:", num_items)

# ======================================
# 12. SORT CHRONOLOGICALLY
# ======================================
print("Sorting ratings by timestamp...")

ratings = ratings.sort_values('timestamp').reset_index(drop=True)

# ======================================
# 13. CREATE TRAIN / TEST SPLIT
# ======================================
print("Creating train-test split...")

split_index = int(len(ratings) * 0.8)

train_df = ratings.iloc[:split_index].copy()
test_df = ratings.iloc[split_index:].copy()

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# ======================================
# 14. CREATE COLD-START ITEMS
# ======================================
print("Creating cold-start split...")

unique_items = train_df['item'].unique()

cold_items = np.random.choice(
    unique_items,
    size=int(len(unique_items) * COLD_ITEM_FRACTION),
    replace=False
)

# Remove cold items completely from train
train_df = train_df[~train_df['item'].isin(cold_items)]

# Separate warm and cold test
cold_test_df = test_df[test_df['item'].isin(cold_items)].copy()
warm_test_df = test_df[~test_df['item'].isin(cold_items)].copy()

print("Cold items:", len(cold_items))
print("Warm test size:", len(warm_test_df))
print("Cold test size:", len(cold_test_df))

# ======================================
# 15. NORMALIZE MOVIE FEATURES
# ======================================
print("Normalizing movie features...")

feature_columns = [
    col for col in movie_feature_matrix.columns
    if col not in ['movieId', 'item']
]

# Convert all feature column names to string
movie_feature_matrix.columns = movie_feature_matrix.columns.astype(str)

# Rebuild feature column list after conversion
feature_columns = [
    col for col in movie_feature_matrix.columns
    if col not in ['movieId', 'item']
]

scaler = MinMaxScaler()
movie_feature_matrix[feature_columns] = scaler.fit_transform(
    movie_feature_matrix[feature_columns]
)
# ======================================
# 16. CALCULATE SPARSITY
# ======================================
num_possible = num_users * num_items
num_actual = len(ratings)
sparsity = 1 - (num_actual / num_possible)

print(f"Dataset sparsity: {sparsity:.4f}")

# ======================================
# 17. SAVE PLOTS
# ======================================
print("Saving plots...")

plt.figure(figsize=(8,5))
plt.hist(movie_feature_matrix[feature_columns].values.flatten(), bins=50)
plt.title("Metadata Feature Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "metadata_distribution.png"))
plt.close()

sample_heatmap = movie_feature_matrix[feature_columns].sample(30, random_state=42)

plt.figure(figsize=(14,8))
sns.heatmap(sample_heatmap, cmap='viridis')
plt.title("Movie Metadata Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "metadata_heatmap.png"))
plt.close()

# ======================================
# 18. SAVE OUTPUT FILES
# ======================================
print("Saving processed files...")

train_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
warm_test_df.to_csv(os.path.join(OUTPUT_PATH, "test_warm.csv"), index=False)
cold_test_df.to_csv(os.path.join(OUTPUT_PATH, "test_cold.csv"), index=False)
movie_feature_matrix.to_csv(os.path.join(OUTPUT_PATH, "movie_features.csv"), index=False)

pd.DataFrame({
    'original_userId': user_encoder.classes_,
    'encoded_userId': range(len(user_encoder.classes_))
}).to_csv(os.path.join(OUTPUT_PATH, "user_mapping.csv"), index=False)

pd.DataFrame({
    'original_movieId': movie_encoder.classes_,
    'encoded_movieId': range(len(movie_encoder.classes_))
}).to_csv(os.path.join(OUTPUT_PATH, "movie_mapping.csv"), index=False)

genome_tags[genome_tags['tagId'].isin(top_tag_ids)].to_csv(
    os.path.join(OUTPUT_PATH, "tag_mapping.csv"),
    index=False
)

pd.DataFrame({
    'cold_item': cold_items
}).to_csv(os.path.join(OUTPUT_PATH, "cold_items.csv"), index=False)

# ======================================
# 19. FINAL SUMMARY
# ======================================
print("\nDONE")
print("=" * 50)
print("Final Users:", num_users)
print("Final Items:", num_items)
print("Train Ratings:", len(train_df))
print("Warm Test Ratings:", len(warm_test_df))
print("Cold Test Ratings:", len(cold_test_df))
print("Movie Feature Matrix Shape:", movie_feature_matrix.shape)
print(f"Dataset Sparsity: {sparsity:.4f}")
print("=" * 50)
