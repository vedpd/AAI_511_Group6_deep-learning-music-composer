"""
Tempo and timing feature extraction for music analysis.
Extracts tempo, timing, and rhythmic features from MIDI files.
"""

import numpy as np
from typing import Dict, List
from collections import Counter

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import RANDOM_SEED
from src.utils.helpers import set_random_seed


class TempoFeatureExtractor:
    """
    Extract tempo and timing features from MIDI files.
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize tempo feature extractor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
    
    def extract_tempo_changes(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract tempo changes from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Array of tempo values at each change point
        """
        try:
            tempo_changes = midi.get_tempo_changes()
            return tempo_changes[1]  # Return tempo values (not times)
        except:
            # If no tempo changes, return default
            return np.array([120.0])
    
    def get_average_tempo(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Get average tempo from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Average tempo in BPM
        """
        tempo_changes = self.extract_tempo_changes(midi)
        return float(np.mean(tempo_changes))
    
    def get_tempo_variance(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Get tempo variance from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Tempo variance
        """
        tempo_changes = self.extract_tempo_changes(midi)
        return float(np.var(tempo_changes))
    
    def extract_note_durations(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """
        Extract all note durations from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of note durations in seconds
        """
        durations = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                durations.append(note.end - note.start)
        
        return durations
    
    def get_duration_statistics(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Get statistics of note durations.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Dictionary with duration statistics
        """
        durations = self.extract_note_durations(midi)
        
        if not durations:
            return {
                'mean_duration': 0.0,
                'std_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'median_duration': 0.0
            }
        
        return {
            'mean_duration': float(np.mean(durations)),
            'std_duration': float(np.std(durations)),
            'min_duration': float(np.min(durations)),
            'max_duration': float(np.max(durations)),
            'median_duration': float(np.median(durations))
        }
    
    def extract_inter_onset_intervals(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """
        Extract inter-onset intervals (time between consecutive note starts).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of inter-onset intervals in seconds
        """
        # Collect all note onset times
        onsets = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                onsets.append(note.start)
        
        # Sort onsets
        onsets.sort()
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(onsets)):
            intervals.append(onsets[i] - onsets[i-1])
        
        return intervals
    
    def get_ioi_statistics(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Get statistics of inter-onset intervals.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Dictionary with IOI statistics
        """
        iois = self.extract_inter_onset_intervals(midi)
        
        if not iois:
            return {
                'mean_ioi': 0.0,
                'std_ioi': 0.0,
                'min_ioi': 0.0,
                'max_ioi': 0.0,
                'median_ioi': 0.0
            }
        
        return {
            'mean_ioi': float(np.mean(iois)),
            'std_ioi': float(np.std(iois)),
            'min_ioi': float(np.min(iois)),
            'max_ioi': float(np.max(iois)),
            'median_ioi': float(np.median(iois))
        }
    
    def extract_note_density(self, midi: pretty_midi.PrettyMIDI, window_size: float = 1.0) -> List[float]:
        """
        Extract note density over time (notes per second).
        
        Args:
            midi: PrettyMIDI object
            window_size: Window size in seconds
            
        Returns:
            List of note densities
        """
        duration = midi.get_end_time()
        if duration == 0:
            return [0.0]
        
        # Collect all note onset times
        onsets = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                onsets.append(note.start)
        
        onsets.sort()
        
        # Calculate density in windows
        densities = []
        num_windows = int(duration / window_size) + 1
        
        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            
            # Count notes in window
            count = sum(1 for onset in onsets if window_start <= onset < window_end)
            density = count / window_size
            densities.append(density)
        
        return densities
    
    def get_average_note_density(self, midi: pretty_midi.PrettyMIDI, window_size: float = 1.0) -> float:
        """
        Get average note density.
        
        Args:
            midi: PrettyMIDI object
            window_size: Window size in seconds
            
        Returns:
            Average note density (notes per second)
        """
        densities = self.extract_note_density(midi, window_size)
        return float(np.mean(densities)) if densities else 0.0
    
    def extract_velocity_statistics(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Extract velocity statistics from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Dictionary with velocity statistics
        """
        velocities = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                velocities.append(note.velocity)
        
        if not velocities:
            return {
                'mean_velocity': 0.0,
                'std_velocity': 0.0,
                'min_velocity': 0.0,
                'max_velocity': 0.0,
                'median_velocity': 0.0
            }
        
        return {
            'mean_velocity': float(np.mean(velocities)),
            'std_velocity': float(np.std(velocities)),
            'min_velocity': float(np.min(velocities)),
            'max_velocity': float(np.max(velocities)),
            'median_velocity': float(np.median(velocities))
        }
    
    def extract_comprehensive_tempo_features(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, any]:
        """
        Extract comprehensive tempo and timing features.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Dictionary with all tempo features
        """
        features = {}
        
        # Tempo features
        features['average_tempo'] = self.get_average_tempo(midi)
        features['tempo_variance'] = self.get_tempo_variance(midi)
        
        # Duration features
        duration_stats = self.get_duration_statistics(midi)
        features.update({f'duration_{k}': v for k, v in duration_stats.items()})
        
        # IOI features
        ioi_stats = self.get_ioi_statistics(midi)
        features.update({f'ioi_{k}': v for k, v in ioi_stats.items()})
        
        # Density features
        features['average_note_density'] = self.get_average_note_density(midi)
        
        # Velocity features
        velocity_stats = self.extract_velocity_statistics(midi)
        features.update({f'velocity_{k}': v for k, v in velocity_stats.items()})
        
        return features


def extract_tempo_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, any]:
    """
    Convenience function to extract tempo features from MIDI.
    
    Args:
        midi: PrettyMIDI object
        
    Returns:
        Dictionary with tempo features
    """
    extractor = TempoFeatureExtractor()
    return extractor.extract_comprehensive_tempo_features(midi)
