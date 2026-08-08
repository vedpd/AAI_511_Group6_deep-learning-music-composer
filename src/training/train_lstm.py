"""
Training script for LSTM composer classification model.
"""

import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.utils.config import (LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_EARLY_STOPPING_PATIENCE,
                           BEST_LSTM_MODEL, FINAL_LSTM_MODEL, CHECKPOINTS_DIR, 
                           COMPOSERS, RANDOM_SEED)
from src.utils.helpers import set_random_seed, format_time, count_parameters
from src.models.lstm_model import LSTMComposerClassifier


class LSTMTrainer:
    """
    Trainer for LSTM composer classification model.
    """
    
    def __init__(self,
                 model: LSTMComposerClassifier,
                 batch_size: int = LSTM_BATCH_SIZE,
                 epochs: int = LSTM_EPOCHS,
                 early_stopping_patience: int = LSTM_EARLY_STOPPING_PATIENCE,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize LSTM trainer.
        
        Args:
            model: LSTM model instance
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.history = None
        self.training_time = None
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              class_weights: Optional[Dict] = None,
              checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Optional class weights for imbalanced data
            checkpoint_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = BEST_LSTM_MODEL
        
        # Ensure checkpoint directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get callbacks
        callbacks = self.model.get_callbacks(
            str(checkpoint_path),
            self.early_stopping_patience
        )
        
        print(f"Training LSTM model for up to {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Measure training time
        start_time = time.time()
        
        # Train model
        self.history = self.model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {format_time(self.training_time)}")
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model on test set...")
        
        # Load best model
        if BEST_LSTM_MODEL.exists():
            print(f"Loading best model from {BEST_LSTM_MODEL}")
            self.model.model = keras.models.load_model(BEST_LSTM_MODEL)
        
        # Evaluate
        test_loss, test_accuracy = self.model.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return metrics
    
    def save_final_model(self, output_path: Optional[Path] = None) -> None:
        """
        Save the final trained model.
        
        Args:
            output_path: Path to save model
        """
        if output_path is None:
            output_path = FINAL_LSTM_MODEL
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if BEST_LSTM_MODEL.exists():
            # Save the best model
            import shutil
            shutil.copy(BEST_LSTM_MODEL, output_path)
            print(f"Saved final model to {output_path}")
        else:
            # Save current model
            self.model.model.save(output_path)
            print(f"Saved current model to {output_path}")
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training process.
        
        Returns:
            Dictionary with training summary
        """
        if self.history is None:
            return {}
        
        history_dict = self.history.history
        
        summary = {
            'epochs_trained': len(history_dict['loss']),
            'final_train_loss': history_dict['loss'][-1],
            'final_train_accuracy': history_dict['accuracy'][-1],
            'final_val_loss': history_dict['val_loss'][-1],
            'final_val_accuracy': history_dict['val_accuracy'][-1],
            'best_val_accuracy': max(history_dict['val_accuracy']),
            'best_val_loss': min(history_dict['val_loss']),
            'training_time': self.training_time,
            'parameters': count_parameters(self.model.model)
        }
        
        return summary


def train_lstm_model(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    input_shape: Tuple[int, int],
                    num_classes: int = len(COMPOSERS),
                    model_type: str = 'standard',
                    **kwargs) -> Tuple[Dict, Dict]:
    """
    Convenience function to train LSTM model with standard parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        input_shape: Input shape for model
        num_classes: Number of classes
        model_type: Type of LSTM model
        **kwargs: Additional arguments for trainer
        
    Returns:
        Tuple of (training summary, evaluation metrics)
    """
    set_random_seed()
    
    # Create model
    from src.models import create_lstm_model
    model = create_lstm_model(input_shape, num_classes, model_type)
    model.summary()
    
    # Create trainer
    trainer = LSTMTrainer(model, **kwargs)
    
    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save final model
    trainer.save_final_model()
    
    # Get training summary
    summary = trainer.get_training_summary()
    
    return summary, metrics
