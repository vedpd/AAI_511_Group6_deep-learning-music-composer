#!/usr/bin/env python
"""
Smoke test to verify all imports work correctly.
Run this script to ensure the project is properly set up.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all critical modules can be imported."""
    
    print("Testing imports...")
    
    try:
        print("  ✓ src.utils.config")
        from src.utils.config import (
            COMPOSERS, NUM_CLASSES, SEQUENCE_LENGTH, MAX_TIME_STEPS,
            PITCH_RANGE, ensure_directories
        )
        
        print("  ✓ src.utils.helpers")
        from src.utils.helpers import set_random_seed, ensure_dir
        
        print("  ✓ src.preprocessing.midi_loader")
        from src.preprocessing.midi_loader import MIDILoader, load_midi_dataset
        
        print("  ✓ src.preprocessing.clean_dataset")
        from src.preprocessing.clean_dataset import DatasetCleaner, clean_dataset
        
        print("  ✓ src.preprocessing.split_dataset")
        from src.preprocessing.split_dataset import DatasetSplitter, split_dataset
        
        print("  ✓ src.preprocessing.augmentation")
        from src.preprocessing.augmentation import MIDIAugmentation, augment_midi
        
        print("  ✓ src.features.note_features")
        from src.features.note_features import NoteFeatureExtractor, extract_note_features
        
        print("  ✓ src.features.piano_roll")
        from src.features.piano_roll import PianoRollExtractor, extract_piano_roll
        
        print("  ✓ src.features.chord_features")
        from src.features.chord_features import ChordFeatureExtractor, extract_chord_features
        
        print("  ✓ src.features.tempo_features")
        from src.features.tempo_features import TempoFeatureExtractor, extract_tempo_features
        
        print("  ✓ src.datasets.lstm_dataset")
        from src.datasets.lstm_dataset import LSTMDataset, prepare_lstm_dataset
        
        print("  ✓ src.datasets.cnn_dataset")
        from src.datasets.cnn_dataset import CNNDataset, prepare_cnn_dataset
        
        print("  ✓ src.models.lstm_model")
        from src.models.lstm_model import LSTMComposerClassifier, create_lstm_model
        
        print("  ✓ src.models.cnn_model")
        from src.models.cnn_model import CNNComposerClassifier, create_cnn_model
        
        print("  ✓ src.training.train_lstm")
        from src.training.train_lstm import LSTMTrainer, train_lstm_model
        
        print("  ✓ src.training.train_cnn")
        from src.training.train_cnn import CNNTrainer, train_cnn_model
        
        print("  ✓ src.evaluation.metrics")
        from src.evaluation.metrics import ModelEvaluator, evaluate_model
        
        print("  ✓ src.evaluation.visualize")
        from src.evaluation.visualize import ModelVisualizer, visualize_training_history
        
        print("\n✅ All imports successful!")
        print(f"\nProject Configuration:")
        print(f"  Composers: {COMPOSERS}")
        print(f"  Num Classes: {NUM_CLASSES}")
        print(f"  Sequence Length (LSTM): {SEQUENCE_LENGTH}")
        print(f"  Max Time Steps (CNN): {MAX_TIME_STEPS}")
        print(f"  Pitch Range: {PITCH_RANGE}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
