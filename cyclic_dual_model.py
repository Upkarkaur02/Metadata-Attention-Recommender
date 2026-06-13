# cyclic_dual_model.py

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
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# ======================================
# CONFIG
# ======================================
DATA_PATH = "processed_data"
MODEL_PATH = "outputs/models"
ATTENTION_PATH = "outputs/attention_results"

EMBEDDING_DIM = 128  # Increased for better capacity
BATCH_SIZE = 512     # Reduced for better generalization
EPOCHS = 20          # Increased for more training
LEARNING_RATE = 0.0005  # Lower for stable convergence
CYCLES = 10          # Increased for more cyclic refinement

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
# DEFINE DUAL MODELS FOR CYCLIC TRAINING
# ======================================
# Dense layers for latent prediction
dense_ui = Dense(EMBEDDING_DIM, activation='linear', kernel_regularizer=l2(0.01))
dense_iu = Dense(EMBEDDING_DIM, activation='linear', kernel_regularizer=l2(0.01))

user_latent = tf.Variable(tf.random.normal([num_users, EMBEDDING_DIM]), name='user_latent')
item_latent = tf.Variable(tf.random.normal([num_items, EMBEDDING_DIM]), name='item_latent')

optimizer = Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step_user_to_item(users, items):
    with tf.GradientTape() as tape:
        # Predict item latent from user latent
        user_vec = tf.gather(user_latent, users)
        item_pred = dense_ui(user_vec)
        item_true = tf.gather(item_latent, items)
        loss = tf.reduce_mean(tf.square(item_pred - item_true))
    grads = tape.gradient(loss, [user_latent, item_latent])
    optimizer.apply_gradients(zip(grads, [user_latent, item_latent]))
    return loss

@tf.function
def train_step_item_to_user(users, items):
    with tf.GradientTape() as tape:
        # Predict user latent from item latent
        item_vec = tf.gather(item_latent, items)
        user_pred = dense_iu(item_vec)
        user_true = tf.gather(user_latent, users)
        loss = tf.reduce_mean(tf.square(user_pred - user_true))
    grads = tape.gradient(loss, [user_latent, item_latent])
    optimizer.apply_gradients(zip(grads, [user_latent, item_latent]))
    return loss

# Cyclic training
for cycle in range(CYCLES):
    print(f"Cycle {cycle + 1}")
    # Alternate training
    for epoch in range(EPOCHS // CYCLES):
        # Train user to item
        indices = np.random.permutation(len(train_df))
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            users = train_df.iloc[batch_idx]['user'].values
            items = train_df.iloc[batch_idx]['item'].values
            loss_ui = train_step_user_to_item(users, items)
        print(f"User to Item Loss: {loss_ui.numpy():.4f}")

        # Train item to user
        loss_iu = train_step_item_to_user(users, items)
        print(f"Item to User Loss: {loss_iu.numpy():.4f}")

from tensorflow.keras.layers import Layer


# ======================================
# BUILD RATING PREDICTION MODEL USING LEARNED LATENTS WITH ATTENTION
# ======================================
# Now use the learned latents for rating prediction
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')
metadata_input = Input(shape=(num_features,), name='metadata_input')

# Use Embedding layers and set weights to learned latents
user_embedding = Embedding(
    input_dim=num_users,
    output_dim=EMBEDDING_DIM,
    name='user_embedding'
)(user_input)
user_vec = Flatten()(user_embedding)

item_embedding = Embedding(
    input_dim=num_items,
    output_dim=EMBEDDING_DIM,
    name='item_embedding'
)(item_input)
item_vec = Flatten()(item_embedding)

# Attention mechanism: compute attention weights for metadata
attention_scores = Dense(num_features, activation='tanh')(metadata_input)
attention_scores = Softmax(name='attention_weights')(attention_scores)

weighted_metadata = Multiply()([
    metadata_input,
    attention_scores
])

metadata_representation = Dense(64, activation='relu')(weighted_metadata)

# Concatenate user, item, and metadata representations
concat = Concatenate()([user_vec, item_vec, metadata_representation])

x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concat)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='linear')(x)

model = Model(inputs=[user_input, item_input, metadata_input], outputs=output)

# Set the learned weights
model.get_layer('user_embedding').set_weights([user_latent.numpy()])
model.get_layer('item_embedding').set_weights([item_latent.numpy()])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

# ======================================
# INITIALIZE TRAINING HISTORY LISTS
# ======================================
train_losses = []
val_losses = []
train_rmse_list = []
val_rmse_list = []

# Fine-tune the model
history = model.fit(
    x=[train_df['user'], train_df['item'], X_train_metadata],
    y=train_df['rating'],
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
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

with open(os.path.join(MODEL_PATH, "cyclic_history.pkl"), "wb") as f:
    pickle.dump(training_history, f)

print("Training history saved to cyclic_history.pkl")

# ======================================
# EVALUATE
# ======================================
print("Evaluating on warm test...")

warm_predictions = model.predict(
    [test_warm_df['user'], test_warm_df['item'], X_test_warm_metadata],
    batch_size=BATCH_SIZE
).flatten()

warm_rmse = np.sqrt(mean_squared_error(test_warm_df['rating'], warm_predictions))
warm_mae = mean_absolute_error(test_warm_df['rating'], warm_predictions)

print(f"Warm RMSE: {warm_rmse:.4f}")
print(f"Warm MAE: {warm_mae:.4f}")

if len(test_cold_df) > 0:
    print("Evaluating on cold test...")

    cold_predictions = model.predict(
        [test_cold_df['user'], test_cold_df['item'], X_test_cold_metadata],
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
model.save(os.path.join(MODEL_PATH, "cyclic_dual_model.h5"))

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
results_df.to_csv(os.path.join(MODEL_PATH, "cyclic_dual_results.csv"), index=False)

print("DONE")
print(results_df)