"""
Feature extraction modules for the Composer Classification project.
"""

from .note_features import NoteFeatureExtractor, extract_note_features
from .piano_roll import PianoRollExtractor, extract_piano_roll
from .chord_features import ChordFeatureExtractor, extract_chord_features
from .tempo_features import TempoFeatureExtractor, extract_tempo_features

__all__ = [
    'NoteFeatureExtractor',
    'extract_note_features',
    'PianoRollExtractor',
    'extract_piano_roll',
    'ChordFeatureExtractor',
    'extract_chord_features',
    'TempoFeatureExtractor',
    'extract_tempo_features'
]
