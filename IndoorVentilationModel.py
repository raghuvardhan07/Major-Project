import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib # For saving the scaler
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
DATASET_PATH = './harmonized_dataset.csv' 
MODEL_WEIGHTS_PATH = 'pretrained_lstm.weights.h5'
SCALER_PATH = 'pretrained_scaler.joblib'
SEQUENCE_LENGTH = 10 # Number of timesteps in each input sequence
NUM_FEATURES = 5 # co2, temp, hum, occ, hour
BATCH_SIZE = 64
EPOCHS = 5 # Adjust as needed


def load_and_preprocess_data(filepath):
    """
    Loads the merged dataset, selects relevant columns, handles missing values,
    and engineers the 'hour' feature.

    Args:
        filepath (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Preprocessed data with columns ['co2', 'temperature', 'humidity', 'occupancy', 'hour'].
                      Returns None if loading fails.
    """
    logging.info(f"Loading data from: {filepath}")
    try:
        # Load dataset - Ensure 'timestamp' column is parsed as datetime
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        logging.info(f"Original dataset shape: {df.shape}")
        
        # Select relevant columns (adjust names if needed)
        # Assuming column names are: 'timestamp', 'co2', 'temperature', 'humidity', 'occupancy'
        required_cols = ['timestamp', 'co2', 'temperature', 'humidity', 'occupancy']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logging.error(f"Missing required columns: {missing}")
            return None
            
        df_selected = df[required_cols].copy()

        initial_missing = df_selected.isnull().sum().sum()
        if initial_missing > 0:
             logging.warning(f"Found {initial_missing} missing values. Applying forward fill.")
             df_selected.fillna(method='ffill', inplace=True)
             # Check if any NaNs remain (e.g., at the beginning)
             remaining_missing = df_selected.isnull().sum().sum()
             if remaining_missing > 0:
                  logging.warning(f"{remaining_missing} missing values remain after ffill. Applying backfill.")
                  df_selected.fillna(method='bfill', inplace=True) # Backfill for NaNs at the start
                  if df_selected.isnull().sum().sum() > 0:
                       logging.error("Could not fill all missing values. Exiting.")
                       return None
                       
        # --- Feature Engineering: Hour ---
        df_selected['hour'] = df_selected['timestamp'].dt.hour / 24.0
        
        # Return only the feature columns in the desired order
        feature_cols = ['co2', 'temperature', 'humidity', 'occupancy', 'hour']
        df_final = df_selected[feature_cols]
        
        logging.info(f"Preprocessing complete. Final data shape: {df_final.shape}")
        return df_final

    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error during data loading/preprocessing: {e}")
        return None

def create_sequences(data, sequence_length):
    """
    Generates sequences for LSTM training.

    Args:
        data (np.ndarray): Scaled data array (samples, features).
        sequence_length (int): Number of timesteps per sequence.

    Returns:
        tuple: (X, y) where X is (num_sequences, sequence_length, num_features)
               and y is (num_sequences, num_features) representing the step
               immediately following each sequence. Returns (None, None) if not enough data.
    """
    X, y = [], []
    if len(data) <= sequence_length:
        logging.error(f"Not enough data ({len(data)} points) to create sequences of length {sequence_length}.")
        return None, None
        
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length]) # Predict the next step's features
        
    logging.info(f"Generated {len(X)} sequences.")
    return np.array(X), np.array(y)

# --- Model Definition ---

def build_pretrain_model(input_shape, num_output_features):
    """
    Builds the LSTM model architecture for pre-training.

    Args:
        input_shape (tuple): Shape of input sequences (sequence_length, num_features).
        num_output_features (int): Number of features to predict.

    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    logging.info(f"Building pre-training model with input shape {input_shape} and {num_output_features} outputs.")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_shared_1'),
        Dropout(0.2, name='dropout_shared_1'),
        LSTM(32, name='lstm_shared_2'),
        Dropout(0.2, name='dropout_shared_2'),
        Dense(16, activation='relu', name='dense_shared_1'),
        Dense(num_output_features, name='dense_pretrain_output') # Output layer
    ], name="PretrainIndoorLSTM")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logging.info("Model compiled.")
    model.summary(print_fn=logging.info) # Log model summary
    return model

# --- Main Pre-training Function ---

def run_pretraining():
    """
    Orchestrates the pre-training process: load data, scale, create sequences,
    build model, train, and save artifacts.
    """
    # 1. Load and Preprocess Data
    data_df = load_and_preprocess_data(DATASET_PATH)
    if data_df is None:
        logging.error("Failed to load or preprocess data. Exiting pre-training.")
        return

    # 2. Scale Data
    logging.info("Scaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_df)
    logging.info("Data scaled.")
    
    # Save the scaler for potential future use or consistency checks
    try:
        joblib.dump(scaler, SCALER_PATH)
        logging.info(f"Scaler saved to {SCALER_PATH}")
    except Exception as e:
        logging.error(f"Error saving scaler: {e}")
        # Continue training even if scaler saving fails, but log the error

    # 3. Create Sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    if X is None or y is None:
        logging.error("Failed to create sequences. Exiting pre-training.")
        return
        
    # Ensure shapes are correct: X=(samples, seq_len, features), y=(samples, features)
    if X.shape[1] != SEQUENCE_LENGTH or X.shape[2] != NUM_FEATURES:
         logging.error(f"Unexpected shape for X: {X.shape}. Expected: (samples, {SEQUENCE_LENGTH}, {NUM_FEATURES})")
         return
    if y.shape[1] != NUM_FEATURES:
         logging.error(f"Unexpected shape for y: {y.shape}. Expected: (samples, {NUM_FEATURES})")
         return


    # 4. Build Model
    model_input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)
    model = build_pretrain_model(model_input_shape, NUM_FEATURES)

    # 5. Train Model
    logging.info("Starting model training...")
    try:
        history = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2, # Use part of the data for validation
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            ],
            verbose=1 # Set to 1 or 2 for progress, 0 for silent
        )
        logging.info("Model training finished.")
        
        # Log final validation loss/mae
        val_loss, val_mae = model.evaluate(X, y, verbose=0) # Evaluate on the full set (since EarlyStopping restored best)
        logging.info(f"Final Model Evaluation - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return # Exit if training fails

    # 6. Save Model Weights
    try:
        model.save_weights(MODEL_WEIGHTS_PATH)
        logging.info(f"Pre-trained model weights saved to {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        logging.error(f"Error saving model weights: {e}")

if __name__ == "__main__":
    logging.info("--- Starting Phase 1: LSTM Pre-training ---")
    run_pretraining()
    logging.info("--- Pre-training Script Finished ---") 