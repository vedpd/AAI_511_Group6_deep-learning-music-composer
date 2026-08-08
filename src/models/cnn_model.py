"""
CNN model architecture for composer classification.
Implements CNN-based deep learning model for piano roll classification.
"""

import numpy as np
from typing import Tuple, Optional, List

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.utils.config import (NUM_CLASSES, CNN_FILTERS, CNN_KERNEL_SIZE, 
                           CNN_POOL_SIZE, CNN_DROPOUT, CNN_DENSE_UNITS,
                           CNN_LEARNING_RATE, RANDOM_SEED)
from src.utils.helpers import set_random_seed


class CNNComposerClassifier:
    """
    CNN-based composer classification model.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int = NUM_CLASSES,
                 filters: List[int] = None,
                 kernel_size: Tuple[int, int] = CNN_KERNEL_SIZE,
                 pool_size: Tuple[int, int] = CNN_POOL_SIZE,
                 dropout: float = CNN_DROPOUT,
                 dense_units: int = CNN_DENSE_UNITS,
                 learning_rate: float = CNN_LEARNING_RATE,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize CNN classifier.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of composer classes
            filters: List of filter counts for each conv layer
            kernel_size: Kernel size for convolutional layers
            pool_size: Pool size for max pooling
            dropout: Dropout rate
            dense_units: Number of units in dense layers
            learning_rate: Learning rate for optimizer
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model")
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters if filters is not None else CNN_FILTERS
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional blocks
        for i, num_filters in enumerate(self.filters):
            x = layers.Conv2D(
                num_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Flatten and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_Composer_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_simple_model(self) -> keras.Model:
        """
        Build simpler CNN model for faster training.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Single conv block
        x = layers.Conv2D(
            64,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu'
        )(inputs)
        x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Flatten and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='Simple_CNN_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_deep_model(self) -> keras.Model:
        """
        Build deeper CNN model with more layers.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # More convolutional blocks
        filters = [32, 64, 128, 256]
        for i, num_filters in enumerate(filters):
            x = layers.Conv2D(
                num_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation='relu'
            )(x)
            x = layers.Conv2D(
                num_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation='relu'
            )(x)
            x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Flatten and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dense(self.dense_units // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='Deep_CNN_Classifier')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_residual_model(self) -> keras.Model:
        """
        Build CNN model with residual connections.
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv
        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual blocks
        for filters in [64, 128, 256]:
            # Shortcut
            shortcut = x
            
            # First conv
            x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Second conv
            x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Match dimensions for shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(shortcut)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='Residual_CNN_Classifier')
        
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


def create_cnn_model(input_shape: Tuple[int, int, int],
                    num_classes: int = NUM_CLASSES,
                    model_type: str = 'standard',
                    **kwargs) -> CNNComposerClassifier:
    """
    Convenience function to create CNN model.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of classes
        model_type: Type of model ('standard', 'simple', 'deep', 'residual')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        CNN model instance
    """
    model = CNNComposerClassifier(input_shape, num_classes, **kwargs)
    
    if model_type == 'standard':
        model.build_model()
    elif model_type == 'simple':
        model.build_simple_model()
    elif model_type == 'deep':
        model.build_deep_model()
    elif model_type == 'residual':
        model.build_residual_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
