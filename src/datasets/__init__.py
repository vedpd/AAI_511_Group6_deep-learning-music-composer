"""
Dataset modules for the Composer Classification project.
"""

from .lstm_dataset import LSTMDataset, prepare_lstm_dataset
from .cnn_dataset import CNNDataset, prepare_cnn_dataset

__all__ = [
    'LSTMDataset',
    'prepare_lstm_dataset',
    'CNNDataset',
    'prepare_cnn_dataset'
]
