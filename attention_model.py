# attention_model.py

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Concatenate,
    Dense,
    Dropout,
    Multiply,
    Softmax,
    Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ======================================
# CONFIG
# ======================================
DATA_PATH = "processed_data"
MODEL_PATH = "outputs/models"
ATTENTION_PATH = "outputs/attention_results"

EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(ATTENTION_PATH, exist_ok=True)

# ======================================
# LOAD DATA
# ======================================
print("Loading processed data...")

train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_warm_df = pd.read_csv(os.path.join(DATA_PATH, "test_warm.csv"))
test_cold_df = pd.read_csv(os.path.join(DATA_PATH, "test_cold.csv"))
movie_features = pd.read_csv(os.path.join(DATA_PATH, "movie_features.csv"))
tag_mapping = pd.read_csv(os.path.join(DATA_PATH, "tag_mapping.csv"))
movie_mapping = pd.read_csv(os.path.join(DATA_PATH, "movie_mapping.csv"))

# ======================================
# USER / ITEM COUNTS
# ======================================
num_users = max(
    train_df['user'].max(),
    test_warm_df['user'].max(),
    test_cold_df['user'].max()
) + 1

num_items = max(
    train_df['item'].max(),
    test_warm_df['item'].max(),
    test_cold_df['item'].max()
) + 1

# ======================================
# PREPARE MOVIE FEATURE MATRIX
# ======================================
movie_features = movie_features.sort_values('item')

feature_columns = [
    col for col in movie_features.columns
    if col not in ['movieId', 'item']
]

metadata_matrix = movie_features[feature_columns].values
num_features = metadata_matrix.shape[1]

print("Metadata matrix shape:", metadata_matrix.shape)
print("Number of metadata features:", num_features)

# ======================================
# FUNCTION TO GET METADATA
# ======================================
def get_metadata_features(df):
    return metadata_matrix[df['item'].values]

X_train_metadata = get_metadata_features(train_df)
X_test_warm_metadata = get_metadata_features(test_warm_df)
X_test_cold_metadata = get_metadata_features(test_cold_df)

# ======================================
# MODEL INPUTS
# ======================================
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')
metadata_input = Input(shape=(num_features,), name='metadata_input')

# ======================================
# USER / ITEM EMBEDDINGS
# ======================================
user_embedding = Embedding(
    input_dim=num_users,
    output_dim=EMBEDDING_DIM,
    name='user_embedding'
)(user_input)

item_embedding = Embedding(
    input_dim=num_items,
    output_dim=EMBEDDING_DIM,
    name='item_embedding'
)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

# ======================================
# ATTENTION OVER METADATA
# ======================================
metadata_representation = Dense(
    64,
    activation='relu',
    name='metadata_projection'
)(metadata_input)

# ======================================
# CONCATENATE ALL FEATURES
# ======================================
concat = Concatenate()([
    user_vec,
    item_vec,
    metadata_representation
])

x = Dense(128, activation='relu')(concat)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='linear')(x)

# ======================================
# BUILD MODEL
# ======================================
model = Model(
    inputs=[user_input, item_input, metadata_input],
    outputs=output
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ======================================
# INITIALIZE TRAINING HISTORY LISTS
# ======================================
train_losses = []
val_losses = []
train_rmse_list = []
val_rmse_list = []

# ======================================
# TRAIN MODEL
# ======================================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    x=[
        train_df['user'],
        train_df['item'],
        X_train_metadata
    ],
    y=train_df['rating'],
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)

# ======================================
# EXTRACT AND CALCULATE TRAINING METRICS
# ======================================


# ======================================
# SAVE TRAINING HISTORY
# ======================================
training_history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_rmse": train_rmse_list,
    "val_rmse": val_rmse_list
}

with open(os.path.join(MODEL_PATH, "attention_history.pkl"), "wb") as f:
    pickle.dump(training_history, f)

print("Training history saved to attention_history.pkl")

# ======================================
# EVALUATE WARM TEST
# ======================================
print("Evaluating warm test...")

warm_predictions = model.predict(
    [
        test_warm_df['user'],
        test_warm_df['item'],
        X_test_warm_metadata
    ],
    batch_size=BATCH_SIZE
).flatten()

warm_rmse = np.sqrt(mean_squared_error(test_warm_df['rating'], warm_predictions))
warm_mae = mean_absolute_error(test_warm_df['rating'], warm_predictions)

print(f"Warm RMSE: {warm_rmse:.4f}")
print(f"Warm MAE: {warm_mae:.4f}")

# ======================================
# EVALUATE COLD TEST
# ======================================
print("Evaluating cold test...")

cold_predictions = model.predict(
    [
        test_cold_df['user'],
        test_cold_df['item'],
        X_test_cold_metadata
    ],
    batch_size=BATCH_SIZE
).flatten()

cold_rmse = np.sqrt(mean_squared_error(test_cold_df['rating'], cold_predictions))
cold_mae = mean_absolute_error(test_cold_df['rating'], cold_predictions)

print(f"Cold RMSE: {cold_rmse:.4f}")
print(f"Cold MAE: {cold_mae:.4f}")

# ======================================
# SAVE MODEL
# ======================================
model.save(
    os.path.join(MODEL_PATH,
    "metadata_concat_model.keras")
)

# ======================================
# SAVE RESULTS
# ======================================
results = pd.DataFrame([
    {
        'warm_rmse': warm_rmse,
        'warm_mae': warm_mae,
        'cold_rmse': cold_rmse,
        'cold_mae': cold_mae
    }
])

results.to_csv(
    os.path.join(MODEL_PATH, "attention_results.csv"),
    index=False
)

# ======================================
# EXTRACT ATTENTION WEIGHTS MODEL
# ======================================
for layer in model.layers:
    print(layer.name)

# ======================================
# SAVE TOP FEATURES FOR SAMPLE MOVIES
# ======================================
feature_names = feature_columns

summary_rows = []

for i in range(min(10, len(sample_movies))):
    movie_id = sample_movies.iloc[i]['movieId']
    item_id = sample_movies.iloc[i]['item']

    attention_scores_movie = attention_values[i]

    top_indices = np.argsort(attention_scores_movie)[-5:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]

    summary_rows.append({
        'movieId': movie_id,
        'item': item_id,
        'top_feature_1': top_features[0],
        'top_feature_2': top_features[1],
        'top_feature_3': top_features[2],
        'top_feature_4': top_features[3],
        'top_feature_5': top_features[4]
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(ATTENTION_PATH, "top_attention_features.csv"),
    index=False
)

print("DONE")
print(results)
