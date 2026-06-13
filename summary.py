import pandas as pd

train = pd.read_csv("processed_data/train.csv")
test_warm = pd.read_csv("processed_data/test_warm.csv")
test_cold = pd.read_csv("processed_data/test_cold.csv")

data = pd.concat([train, test_warm, test_cold])

print("Users after filtering =", data.userId.nunique())
print("Movies after filtering =", data.movieId.nunique())
print("Ratings after filtering =", len(data))