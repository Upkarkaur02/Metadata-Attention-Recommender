# dataset_summary.py

import pandas as pd

train = pd.read_csv("processed_data/train.csv")
test_warm = pd.read_csv("processed_data/test_warm.csv")
test_cold = pd.read_csv("processed_data/test_cold.csv")

all_data = pd.concat(
    [train, test_warm, test_cold],
    ignore_index=True
)

print("Users after filtering:",
      all_data["userId"].nunique())

print("Movies after filtering:",
      all_data["movieId"].nunique())

print("Ratings after filtering:",
      len(all_data))

print("Train ratings:",
      len(train))

print("Warm test ratings:",
      len(test_warm))

print("Cold test ratings:",
      len(test_cold))