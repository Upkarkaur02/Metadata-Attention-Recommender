import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = "processed_data"

MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 50
MAX_USERS = 8000
COLD_ITEM_FRACTION = 0.1
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================
# 1. LOAD DATA
# =========================
print("Loading data...")

ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
genome_scores = pd.read_csv(os.path.join(DATA_PATH, "genome-scores.csv"))

# =========================
# 2. INITIAL VISUALIZATION
# =========================

print("Plotting raw data distributions...")

# User activity distribution
user_counts = ratings['userId'].value_counts()

plt.figure()
plt.hist(user_counts, bins=50)
plt.title("Raw User Activity Distribution")
plt.xlabel("Ratings per User")
plt.ylabel("Frequency")
plt.show()

# Movie popularity distribution
movie_counts = ratings['movieId'].value_counts()

plt.figure()
plt.hist(movie_counts, bins=50)
plt.title("Raw Movie Popularity Distribution")
plt.xlabel("Ratings per Movie")
plt.ylabel("Frequency")
plt.show()

# =========================
# 3. FILTER USERS
# =========================

ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= MIN_USER_RATINGS].index)]

# =========================
# 4. FILTER MOVIES
# =========================

movie_counts = ratings['movieId'].value_counts()
ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index)]

# =========================
# 5. ALIGN WITH GENOME TAGS
# =========================

valid_movies = set(genome_scores['movieId'].unique())
ratings = ratings[ratings['movieId'].isin(valid_movies)]

# =========================
# 6. SAMPLING
# =========================

unique_users = ratings['userId'].unique()
if len(unique_users) > MAX_USERS:
    sampled_users = np.random.choice(unique_users, MAX_USERS, replace=False)
    ratings = ratings[ratings['userId'].isin(sampled_users)]

# =========================
# 7. POST-FILTER VISUALIZATION
# =========================

print("Plotting filtered distributions...")

user_counts_filtered = ratings['userId'].value_counts()

plt.figure()
plt.hist(user_counts_filtered, bins=50)
plt.title("Filtered User Activity")
plt.show()

movie_counts_filtered = ratings['movieId'].value_counts()

plt.figure()
plt.hist(movie_counts_filtered, bins=50)
plt.title("Filtered Movie Popularity")
plt.show()

# =========================
# 8. ENCODE IDS
# =========================

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['userId'])
ratings['item'] = item_encoder.fit_transform(ratings['movieId'])

num_users = ratings['user'].nunique()
num_items = ratings['item'].nunique()

# =========================
# 9. SPARSITY CALCULATION
# =========================

num_possible = num_users * num_items
num_actual = len(ratings)
sparsity = 1 - (num_actual / num_possible)

print(f"Sparsity: {sparsity:.4f}")

# =========================
# 10. ITEM FEATURE MATRIX
# =========================

genome_scores = genome_scores[
    genome_scores['movieId'].isin(ratings['movieId'].unique())
].copy()

genome_scores['item'] = item_encoder.transform(genome_scores['movieId'])

item_features = genome_scores.pivot(
    index='item',
    columns='tagId',
    values='relevance'
).fillna(0)

item_features = item_features.reindex(range(num_items), fill_value=0)

# =========================
# 11. FEATURE DISTRIBUTION
# =========================

plt.figure()
plt.hist(item_features.values.flatten(), bins=50)
plt.title("Tag Relevance Distribution")
plt.xlabel("Relevance")
plt.ylabel("Frequency")
plt.show()

# =========================
# 12. HEATMAP (EXPLAINABILITY VISUAL)
# =========================

sample_features = item_features.sample(50)

plt.figure(figsize=(10,6))
sns.heatmap(sample_features, cmap='viridis')
plt.title("Item-Tag Feature Heatmap")
plt.show()

# =========================
# 13. NORMALIZATION
# =========================

scaler = MinMaxScaler()
item_features[:] = scaler.fit_transform(item_features)

# =========================
# 14. TRAIN-TEST SPLIT
# =========================

train_df, test_df = train_test_split(
    ratings,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# =========================
# 15. COLD START SPLIT
# =========================

unique_items = ratings['item'].unique()

cold_items = np.random.choice(
    unique_items,
    size=int(len(unique_items) * COLD_ITEM_FRACTION),
    replace=False
)

train_df = train_df[~train_df['item'].isin(cold_items)]
test_cold = test_df[test_df['item'].isin(cold_items)]
test_warm = test_df[~test_df['item'].isin(cold_items)]

# =========================
# 16. COLD-START VISUALIZATION
# =========================

labels = ['Train Items', 'Cold Items']
sizes = [num_items - len(cold_items), len(cold_items)]

plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Cold-Start Split")
plt.show()

# =========================
# 17. SAVE OUTPUT
# =========================

os.makedirs(OUTPUT_PATH, exist_ok=True)

train_df.to_csv(f"{OUTPUT_PATH}/train.csv", index=False)
test_warm.to_csv(f"{OUTPUT_PATH}/test_warm.csv", index=False)
test_cold.to_csv(f"{OUTPUT_PATH}/test_cold.csv", index=False)
item_features.to_csv(f"{OUTPUT_PATH}/item_features.csv")

print("DONE ✅ Everything ready for modeling")