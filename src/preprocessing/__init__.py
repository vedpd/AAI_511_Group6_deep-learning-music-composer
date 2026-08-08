"""
Preprocessing modules for the Composer Classification project.
"""

from .midi_loader import MIDILoader, load_midi_dataset
from .clean_dataset import DatasetCleaner, clean_dataset
from .augmentation import MIDIAugmentation, augment_midi
from .split_dataset import DatasetSplitter, split_dataset

__all__ = [
    'MIDILoader',
    'load_midi_dataset',
    'DatasetCleaner',
    'clean_dataset',
    'MIDIAugmentation',
    'augment_midi',
    'DatasetSplitter',
    'split_dataset'
]
