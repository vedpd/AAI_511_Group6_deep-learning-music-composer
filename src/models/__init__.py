"""
Model architectures for the Composer Classification project.
"""

from .lstm_model import LSTMComposerClassifier, create_lstm_model
from .cnn_model import CNNComposerClassifier, create_cnn_model

__all__ = [
    'LSTMComposerClassifier',
    'create_lstm_model',
    'CNNComposerClassifier',
    'create_cnn_model'
]
