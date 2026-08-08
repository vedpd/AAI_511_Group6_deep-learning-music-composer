"""
Configuration file for the Composer Classification project.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Composer labels
COMPOSERS = ["Bach", "Beethoven", "Chopin", "Mozart"]
NUM_CLASSES = len(COMPOSERS)
COMPOSER_TO_LABEL = {composer: idx for idx, composer in enumerate(COMPOSERS)}
LABEL_TO_COMPOSER = {idx: composer for idx, composer in enumerate(COMPOSERS)}

# Data processing
SEQUENCE_LENGTH = 500  # Maximum sequence length for LSTM
MAX_TIME_STEPS = 1000  # Maximum time steps for CNN piano roll
SAMPLE_RATE = 44100
MIN_NOTES = 10  # Minimum number of notes to keep a file

# Feature extraction
PITCH_RANGE = 128  # MIDI pitch range
NUM_VELOCITY_BINS = 32

# Data augmentation
AUGMENTATION_ENABLED = True
PITCH_SHIFT_RANGE = (-2, 2)  # Semitones
TEMPO_SCALE_RANGE = (0.8, 1.2)
VELOCITY_VARIATION_RANGE = (0.8, 1.2)
TIME_SHIFT_MAX = 0.1  # 10% of sequence length

# Dataset split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# LSTM Model Architecture
LSTM_HIDDEN_UNITS = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_EMBEDDING_DIM = 128
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_EARLY_STOPPING_PATIENCE = 10

# CNN Model Architecture
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.3
CNN_DENSE_UNITS = 512
CNN_LEARNING_RATE = 0.001
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 50
CNN_EARLY_STOPPING_PATIENCE = 10

# Training settings
USE_AUGMENTATION = True
NUM_WORKERS = 4
PIN_MEMORY = True

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1_score"]

# Visualization
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# File naming
METADATA_FILE = INTERIM_DATA_DIR / "metadata.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.npz"
LSTM_FEATURES_FILE = FEATURES_DIR / "lstm_features.npz"
CNN_FEATURES_FILE = FEATURES_DIR / "cnn_features.npz"
LSTM_MODEL_NAME = "lstm_composer_classifier"
CNN_MODEL_NAME = "cnn_composer_classifier"
BEST_LSTM_MODEL = SAVED_MODELS_DIR / f"{LSTM_MODEL_NAME}_best.h5"
BEST_CNN_MODEL = SAVED_MODELS_DIR / f"{CNN_MODEL_NAME}_best.h5"
FINAL_LSTM_MODEL = SAVED_MODELS_DIR / f"{LSTM_MODEL_NAME}_final.h5"
FINAL_CNN_MODEL = SAVED_MODELS_DIR / f"{CNN_MODEL_NAME}_final.h5"

# Hyperparameter tuning
HP_TUNING_ENABLED = True
HP_TUNING_TRIALS = 20
HP_TUNING_EPOCHS = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, 
                 FEATURES_DIR, MODELS_DIR, CHECKPOINTS_DIR, SAVED_MODELS_DIR,
                 REPORTS_DIR, FIGURES_DIR, TABLES_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
