"""
CNN dataset preparation for the Composer Classification project.
Handles loading and preprocessing of MIDI data for CNN models using piano rolls.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import (MAX_TIME_STEPS, PITCH_RANGE, CNN_FEATURES_FILE,
                           COMPOSERS, COMPOSER_TO_LABEL, RANDOM_SEED)
from src.utils.helpers import set_random_seed, ensure_dir
from src.preprocessing.midi_loader import MIDILoader
from src.features.piano_roll import extract_piano_roll


class CNNDataset:
    """
    Dataset class for CNN model training.
    """
    
    def __init__(self, 
                 roll_type: str = 'velocity',
                 time_steps: int = MAX_TIME_STEPS,
                 pitch_range: int = PITCH_RANGE,
                 fps: float = 20.0,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize CNN dataset.
        
        Args:
            roll_type: Type of piano roll ('velocity', 'binary', 'onset', 'combined', 'multi')
            time_steps: Number of time steps
            pitch_range: Number of pitches
            fps: Frames per second for time discretization
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        self.roll_type = roll_type
        self.time_steps = time_steps
        self.pitch_range = pitch_range
        self.fps = fps
        self.features = None
        self.labels = None
        self.filepaths = None
        self.composers = None
    
    def load_from_metadata(self, metadata_df: pd.DataFrame) -> None:
        """
        Load dataset from metadata DataFrame.
        
        Args:
            metadata_df: DataFrame with metadata containing filepath and label columns
        """
        features_list = []
        labels_list = []
        filepaths_list = []
        composers_list = []
        
        print(f"Loading {len(metadata_df)} MIDI files for CNN dataset...")
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            filepath = Path(row['filepath'])
            
            if not filepath.exists():
                warnings.warn(f"File not found: {filepath}")
                continue
            
            try:
                midi = pretty_midi.PrettyMIDI(str(filepath))
                features = extract_piano_roll(
                    midi,
                    roll_type=self.roll_type,
                    time_steps=self.time_steps,
                    pitch_range=self.pitch_range,
                    fps=self.fps
                )
                
                features_list.append(features)
                labels_list.append(row['label'])
                filepaths_list.append(str(filepath))
                composers_list.append(row['composer'])
                
            except Exception as e:
                warnings.warn(f"Failed to process {filepath}: {e}")
                continue
        
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        self.filepaths = filepaths_list
        self.composers = composers_list
        
        print(f"Loaded {len(self.features)} samples successfully")
        print(f"Feature shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def load_from_splits(self, train_df: pd.DataFrame, 
                        val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                         np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset from train/val/test splits.
        
        Args:
            train_df: Training metadata
            val_df: Validation metadata
            test_df: Test metadata
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Loading training set...")
        self.load_from_metadata(train_df)
        X_train, y_train = self.features, self.labels
        
        print("Loading validation set...")
        self.load_from_metadata(val_df)
        X_val, y_val = self.features, self.labels
        
        print("Loading test set...")
        self.load_from_metadata(test_df)
        X_test, y_test = self.features, self.labels
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_dataset(self, output_path: Path = CNN_FEATURES_FILE) -> None:
        """
        Save dataset to numpy file.
        
        Args:
            output_path: Path to save dataset
        """
        ensure_dir(output_path.parent)
        
        np.savez_compressed(
            output_path,
            features=self.features,
            labels=self.labels,
            filepaths=np.array(self.filepaths),
            composers=np.array(self.composers),
            roll_type=self.roll_type,
            time_steps=self.time_steps,
            pitch_range=self.pitch_range
        )
        
        print(f"Saved dataset to {output_path}")
    
    def load_dataset(self, input_path: Path = CNN_FEATURES_FILE) -> None:
        """
        Load dataset from numpy file.
        
        Args:
            input_path: Path to load dataset from
        """
        data = np.load(input_path, allow_pickle=True)
        
        self.features = data['features']
        self.labels = data['labels']
        self.filepaths = data['filepaths'].tolist()
        self.composers = data['composers'].tolist()
        self.roll_type = str(data['roll_type'])
        self.time_steps = int(data['time_steps'])
        self.pitch_range = int(data['pitch_range'])
        
        print(f"Loaded dataset from {input_path}")
        print(f"Feature shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def get_train_test_split(self, test_size: float = 0.2, 
                           random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, 
                                                                       np.ndarray, np.ndarray]:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=random_state
        )
        
        return X_train, y_train, X_test, y_test
    
    def get_class_weights(self) -> dict:
        """
        Calculate class weights for imbalanced dataset.
        
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        
        return {i: weight for i, weight in enumerate(class_weights)}
    
    def normalize_features(self) -> None:
        """Normalize features to [0, 1] range if not already normalized."""
        if self.roll_type == 'velocity':
            # Velocity features are already normalized to [0, 1]
            pass
        elif self.roll_type == 'binary':
            # Binary features are already 0 or 1
            pass
        elif self.roll_type == 'onset':
            # Onset features are already 0 or 1
            pass
        elif self.roll_type == 'combined':
            # Combined features are already normalized
            pass
        elif self.roll_type == 'multi':
            # Multi-channel features are already normalized
            pass
    
    def get_data_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(self.features),
            'feature_shape': self.features.shape,
            'num_classes': len(np.unique(self.labels)),
            'class_distribution': {}
        }
        
        for composer in COMPOSERS:
            count = sum(1 for c in self.composers if c == composer)
            stats['class_distribution'][composer] = count
        
        return stats


def prepare_cnn_dataset(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      roll_type: str = 'velocity',
                      time_steps: int = MAX_TIME_STEPS,
                      pitch_range: int = PITCH_RANGE,
                      fps: float = 20.0,
                      save_path: Path = CNN_FEATURES_FILE) -> Tuple[np.ndarray, np.ndarray, 
                                                                     np.ndarray, np.ndarray, 
                                                                     np.ndarray, np.ndarray]:
    """
    Convenience function to prepare CNN dataset from metadata splits.
    
    Args:
        train_df: Training metadata
        val_df: Validation metadata
        test_df: Test metadata
        roll_type: Type of piano roll
        time_steps: Number of time steps
        pitch_range: Number of pitches
        fps: Frames per second
        save_path: Path to save prepared dataset
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    set_random_seed()
    
    dataset = CNNDataset(roll_type, time_steps, pitch_range, fps)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.load_from_splits(
        train_df, val_df, test_df
    )
    
    if save_path:
        # Save training data as the main dataset
        dataset.features = X_train
        dataset.labels = y_train
        dataset.save_dataset(save_path)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
