"""
LSTM model architecture for composer classification.
Implements LSTM-based deep learning model for sequence classification.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.utils.config import (NUM_CLASSES, LSTM_HIDDEN_UNITS, LSTM_NUM_LAYERS, 
                           LSTM_DROPOUT, LSTM_EMBEDDING_DIM, LSTM_LEARNING_RATE,
                           RANDOM_SEED)
from src.utils.helpers import set_random_seed


class LSTMComposerClassifier:
    """
    LSTM-based composer classification model.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 num_classes: int = NUM_CLASSES,
                 hidden_units: int = LSTM_HIDDEN_UNITS,
                 num_layers: int = LSTM_NUM_LAYERS,
                 dropout: float = LSTM_DROPOUT,
                 embedding_dim: int = LSTM_EMBEDDING_DIM,
                 learning_rate: float = LSTM_LEARNING_RATE,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize LSTM classifier.
        
        Args:
            input_shape: Input shape (sequence_length, features)
            num_classes: Number of composer classes
            hidden_units: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            embedding_dim: Embedding dimension for discrete features
            learning_rate: Learning rate for optimizer
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # If input is 1D (just pitch sequence), add embedding
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
            x = layers.Embedding(input_dim=128, output_dim=self.embedding_dim)(x)
            x = layers.Reshape((self.input_shape[0], self.embedding_dim))(x)
        else:
            x = inputs
        
        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            x = layers.LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                kernel_initializer='glorot_uniform'
            )(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Dense layers
        x = layers.Dense(self.hidden_units // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Composer_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_simple_model(self) -> keras.Model:
        """
        Build simpler LSTM model for faster training.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # If input is 1D, reshape
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # Single LSTM layer
        x = layers.LSTM(self.hidden_units)(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='Simple_LSTM_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_bidirectional_model(self) -> keras.Model:
        """
        Build bidirectional LSTM model.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # If input is 1D, reshape
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # Bidirectional LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            x = layers.Bidirectional(
                layers.LSTM(self.hidden_units, return_sequences=return_sequences)
            )(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Dense layers
        x = layers.Dense(self.hidden_units, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='Bidirectional_LSTM_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def get_callbacks(self, 
                     checkpoint_path: str,
                     early_stopping_patience: int = 10,
                     reduce_lr_patience: int = 5) -> list:
        """
        Get training callbacks.
        
        Args:
            checkpoint_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            List of callbacks
        """
        callbacks_list = [
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_patience,
                mode='max',
                verbose=1,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)


def create_lstm_model(input_shape: Tuple[int, int],
                    num_classes: int = NUM_CLASSES,
                    model_type: str = 'standard',
                    **kwargs) -> LSTMComposerClassifier:
    """
    Convenience function to create LSTM model.
    
    Args:
        input_shape: Input shape
        num_classes: Number of classes
        model_type: Type of model ('standard', 'simple', 'bidirectional')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        LSTM model instance
    """
    model = LSTMComposerClassifier(input_shape, num_classes, **kwargs)
    
    if model_type == 'standard':
        model.build_model()
    elif model_type == 'simple':
        model.build_simple_model()
    elif model_type == 'bidirectional':
        model.build_bidirectional_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
