import pandas as pd

top_features = pd.read_csv("outputs/attention_results/top_attention_features.csv")
tag_mapping = pd.read_csv("processed_data/tag_mapping.csv")
movie_mapping = pd.read_csv("processed_data/movie_mapping.csv")
movies = pd.read_csv("data/movies.csv")

# Create tag dictionary
tag_dict = dict(zip(tag_mapping['tagId'].astype(str), tag_mapping['tag']))

# Merge movie titles
top_features = top_features.merge(
    movie_mapping,
    left_on='movieId',
    right_on='original_movieId',
    how='left'
)

top_features = top_features.merge(
    movies[['movieId', 'title']],
    on='movieId',
    how='left'
)

# Convert feature IDs to readable names
feature_cols = [
    'top_feature_1',
    'top_feature_2',
    'top_feature_3',
    'top_feature_4',
    'top_feature_5'
]

for col in feature_cols:
    top_features[col] = top_features[col].astype(str).map(tag_dict).fillna(top_features[col])

# Keep useful columns
final_output = top_features[
    ['movieId', 'title'] + feature_cols
]

print(final_output.head(10))

final_output.to_csv(
    "outputs/attention_results/decoded_attention_features.csv",
    index=False
)

print("DONE")