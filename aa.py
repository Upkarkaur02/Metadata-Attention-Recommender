import pandas as pd

movie_features = pd.read_csv("processed_data/movie_features.csv")

print(movie_features.columns[:130].tolist())
print("\nTotal columns:", len(movie_features.columns))