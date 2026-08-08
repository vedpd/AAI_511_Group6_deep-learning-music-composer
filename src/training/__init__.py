"""
Training modules for the Composer Classification project.
"""

from .train_lstm import LSTMTrainer, train_lstm_model
from .train_cnn import CNNTrainer, train_cnn_model

__all__ = [
    'LSTMTrainer',
    'train_lstm_model',
    'CNNTrainer',
    'train_cnn_model'
]
