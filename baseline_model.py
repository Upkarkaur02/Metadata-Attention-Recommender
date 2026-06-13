# baseline_model.py

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ======================================
# CONFIG
# ======================================
DATA_PATH = "processed_data"
MODEL_PATH = "outputs/models"

EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs(MODEL_PATH, exist_ok=True)

# ======================================
# LOAD DATA
# ======================================
print("Loading processed data...")

train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_warm_df = pd.read_csv(os.path.join(DATA_PATH, "test_warm.csv"))
test_cold_df = pd.read_csv(os.path.join(DATA_PATH, "test_cold.csv"))

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

print("Train shape:", train_df.shape)
print("Warm test shape:", test_warm_df.shape)
print("Cold test shape:", test_cold_df.shape)
print("Users:", num_users)
print("Items:", num_items)

# ======================================
# MODEL INPUTS
# ======================================
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# ======================================
# EMBEDDINGS
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
# CONCATENATE
# ======================================
concat = Concatenate()([user_vec, item_vec])

x = Dense(128, activation='relu')(concat)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='linear')(x)

# ======================================
# BUILD MODEL
# ======================================
model = Model(inputs=[user_input, item_input], outputs=output)

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
    x=[train_df['user'], train_df['item']],
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
train_losses = history.history['loss']
val_losses = history.history['val_loss']

# Convert MSE losses to RMSE (RMSE = sqrt(MSE))
train_rmse_list = [np.sqrt(loss) for loss in train_losses]
val_rmse_list = [np.sqrt(loss) for loss in val_losses]

# ======================================
# SAVE TRAINING HISTORY
# ======================================
training_history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_rmse": train_rmse_list,
    "val_rmse": val_rmse_list
}

with open(os.path.join(MODEL_PATH, "baseline_history.pkl"), "wb") as f:
    pickle.dump(training_history, f)

print("Training history saved to baseline_history.pkl")

# ======================================
# EVALUATE ON WARM TEST
# ======================================
print("Evaluating on warm test...")

warm_predictions = model.predict(
    [test_warm_df['user'], test_warm_df['item']],
    batch_size=BATCH_SIZE
).flatten()

warm_rmse = np.sqrt(mean_squared_error(test_warm_df['rating'], warm_predictions))
warm_mae = mean_absolute_error(test_warm_df['rating'], warm_predictions)

print(f"Warm RMSE: {warm_rmse:.4f}")
print(f"Warm MAE: {warm_mae:.4f}")

# ======================================
# EVALUATE ON COLD TEST
# ======================================
if len(test_cold_df) > 0:
    print("Evaluating on cold test...")

    cold_predictions = model.predict(
        [test_cold_df['user'], test_cold_df['item']],
        batch_size=BATCH_SIZE
    ).flatten()

    cold_rmse = np.sqrt(mean_squared_error(test_cold_df['rating'], cold_predictions))
    cold_mae = mean_absolute_error(test_cold_df['rating'], cold_predictions)

    print(f"Cold RMSE: {cold_rmse:.4f}")
    print(f"Cold MAE: {cold_mae:.4f}")
else:
    print("No cold test items found.")

# ======================================
# SAVE MODEL
# ======================================
model.save(os.path.join(MODEL_PATH, "baseline_model.h5"))

# ======================================
# SAVE RESULTS
# ======================================
results = {
    'warm_rmse': warm_rmse,
    'warm_mae': warm_mae
}

if len(test_cold_df) > 0:
    results['cold_rmse'] = cold_rmse
    results['cold_mae'] = cold_mae

results_df = pd.DataFrame([results])
results_df.to_csv(os.path.join(MODEL_PATH, "baseline_results.csv"), index=False)

print("DONE")
print(results_df)
