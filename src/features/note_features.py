"""
Note feature extraction for LSTM models.
Extracts note sequences, pitch, duration, and velocity features from MIDI files.
"""

import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import SEQUENCE_LENGTH, PITCH_RANGE, RANDOM_SEED
from src.utils.helpers import set_random_seed


class NoteFeatureExtractor:
    """
    Extract note-level features for LSTM models.
    """
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, random_seed: int = RANDOM_SEED):
        """
        Initialize note feature extractor.
        
        Args:
            sequence_length: Maximum sequence length
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        self.sequence_length = sequence_length
    
    def extract_note_sequence(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract note sequence (pitch values ordered by time).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of note pitches
        """
        notes = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        
        # Extract pitches
        pitches = np.array([note['pitch'] for note in notes])
        
        return pitches
    
    def extract_pitch_sequence(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract pitch sequence with padding/truncation.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of pitches with fixed length
        """
        pitches = self.extract_note_sequence(midi)
        
        # Pad or truncate to sequence length
        if len(pitches) >= self.sequence_length:
            return pitches[:self.sequence_length]
        else:
            # Pad with zeros
            padded = np.zeros(self.sequence_length, dtype=np.int32)
            padded[:len(pitches)] = pitches
            return padded
    
    def extract_duration_sequence(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract note duration sequence.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of note durations with fixed length
        """
        notes = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        
        # Extract durations
        durations = np.array([note['end'] - note['start'] for note in notes])
        
        # Pad or truncate to sequence length
        if len(durations) >= self.sequence_length:
            return durations[:self.sequence_length]
        else:
            # Pad with zeros
            padded = np.zeros(self.sequence_length, dtype=np.float32)
            padded[:len(durations)] = durations
            return padded
    
    def extract_velocity_sequence(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract note velocity sequence.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of note velocities with fixed length
        """
        notes = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        
        # Extract velocities
        velocities = np.array([note['velocity'] for note in notes])
        
        # Pad or truncate to sequence length
        if len(velocities) >= self.sequence_length:
            return velocities[:self.sequence_length]
        else:
            # Pad with zeros
            padded = np.zeros(self.sequence_length, dtype=np.int32)
            padded[:len(velocities)] = velocities
            return padded
    
    def extract_combined_features(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract combined note features (pitch, duration, velocity).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of shape (sequence_length, 3) with [pitch, duration, velocity]
        """
        notes = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        
        # Extract features
        features = []
        for note in notes:
            features.append([
                note['pitch'],
                note['end'] - note['start'],
                note['velocity']
            ])
        
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to sequence length
        if len(features) >= self.sequence_length:
            return features[:self.sequence_length]
        else:
            # Pad with zeros
            padded = np.zeros((self.sequence_length, 3), dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def extract_interval_sequence(self, midi: 'pretty_midi.PrettyMIDI') -> np.ndarray:
        """
        Extract pitch interval sequence (differences between consecutive notes).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of pitch intervals with fixed length
        """
        pitches = self.extract_note_sequence(midi)
        
        if len(pitches) < 2:
            return np.zeros(self.sequence_length, dtype=np.int32)
        
        # Calculate intervals
        intervals = np.diff(pitches)
        
        # Pad or truncate to sequence length
        if len(intervals) >= self.sequence_length:
            return intervals[:self.sequence_length]
        else:
            # Pad with zeros
            padded = np.zeros(self.sequence_length, dtype=np.int32)
            padded[:len(intervals)] = intervals
            return padded
    
    def normalize_features(self, features: np.ndarray, feature_type: str = 'pitch') -> np.ndarray:
        """
        Normalize features to [0, 1] range.
        
        Args:
            features: Feature array
            feature_type: Type of feature ('pitch', 'duration', 'velocity', 'combined')
            
        Returns:
            Normalized feature array
        """
        if feature_type == 'pitch':
            return features / PITCH_RANGE
        elif feature_type == 'duration':
            # Normalize by max duration (assume 10 seconds as max)
            return np.clip(features / 10.0, 0, 1)
        elif feature_type == 'velocity':
            return features / 127.0
        elif feature_type == 'combined':
            # Normalize each column separately
            normalized = features.copy()
            normalized[:, 0] /= PITCH_RANGE  # pitch
            normalized[:, 1] /= 10.0  # duration
            normalized[:, 2] /= 127.0  # velocity
            return normalized
        else:
            return features


def extract_note_features(midi: 'pretty_midi.PrettyMIDI',
                         feature_type: str = 'combined',
                         sequence_length: int = SEQUENCE_LENGTH,
                         normalize: bool = True) -> np.ndarray:
    """
    Convenience function to extract note features from MIDI.
    
    Args:
        midi: PrettyMIDI object
        feature_type: Type of features ('pitch', 'duration', 'velocity', 'combined', 'interval')
        sequence_length: Maximum sequence length
        normalize: Whether to normalize features
        
    Returns:
        Feature array
    """
    extractor = NoteFeatureExtractor(sequence_length)
    
    if feature_type == 'pitch':
        features = extractor.extract_pitch_sequence(midi)
        if normalize:
            features = extractor.normalize_features(features, 'pitch')
    elif feature_type == 'duration':
        features = extractor.extract_duration_sequence(midi)
        if normalize:
            features = extractor.normalize_features(features, 'duration')
    elif feature_type == 'velocity':
        features = extractor.extract_velocity_sequence(midi)
        if normalize:
            features = extractor.normalize_features(features, 'velocity')
    elif feature_type == 'combined':
        features = extractor.extract_combined_features(midi)
        if normalize:
            features = extractor.normalize_features(features, 'combined')
    elif feature_type == 'interval':
        features = extractor.extract_interval_sequence(midi)
        # Normalize intervals to [-1, 1] range
        if normalize:
            features = np.clip(features / 12.0, -1, 1)  # 12 semitones = 1 octave
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features
