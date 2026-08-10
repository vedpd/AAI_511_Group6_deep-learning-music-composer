"""
Chord feature extraction for music analysis.
Extracts chord statistics and harmonic information from MIDI files.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import RANDOM_SEED
from src.utils.helpers import set_random_seed


class ChordFeatureExtractor:
    """
    Extract chord-level features from MIDI files.
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize chord feature extractor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        # Define chord templates (major and minor triads)
        self.chord_templates = {
            'C': [0, 4, 7],
            'C#': [1, 5, 8],
            'Db': [1, 5, 8],
            'D': [2, 6, 9],
            'D#': [3, 7, 10],
            'Eb': [3, 7, 10],
            'E': [4, 8, 11],
            'F': [5, 9, 0],
            'F#': [6, 10, 1],
            'Gb': [6, 10, 1],
            'G': [7, 11, 2],
            'G#': [8, 0, 3],
            'Ab': [8, 0, 3],
            'A': [9, 1, 4],
            'A#': [10, 2, 5],
            'Bb': [10, 2, 5],
            'B': [11, 3, 6]
        }
        
        self.minor_chord_templates = {
            'Cm': [0, 3, 7],
            'C#m': [1, 4, 8],
            'Dm': [2, 5, 9],
            'D#m': [3, 6, 10],
            'Ebm': [3, 6, 10],
            'Em': [4, 7, 11],
            'Fm': [5, 8, 0],
            'F#m': [6, 9, 1],
            'Gm': [7, 10, 2],
            'G#m': [8, 11, 3],
            'Abm': [8, 11, 3],
            'Am': [9, 0, 4],
            'A#m': [10, 1, 5],
            'Bbm': [10, 1, 5],
            'Bm': [11, 2, 6]
        }
    
    def get_active_pitches(self, midi: 'pretty_midi.PrettyMIDI', time: float, window: float = 0.1) -> List[int]:
        """
        Get list of active pitches at a given time.
        
        Args:
            midi: PrettyMIDI object
            time: Time in seconds
            window: Time window to consider active notes
            
        Returns:
            List of active pitch values
        """
        active_pitches = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.start <= time + window and note.end >= time - window:
                    active_pitches.append(note.pitch)
        
        return sorted(list(set(active_pitches)))
    
    def detect_chord(self, active_pitches: List[int]) -> str:
        """
        Detect chord from active pitches using template matching.
        
        Args:
            active_pitches: List of active pitch values
            
        Returns:
            Chord name or 'N.C.' (no chord)
        """
        if len(active_pitches) < 3:
            return 'N.C.'
        
        # Normalize pitches to 0-11 range
        normalized_pitches = [p % 12 for p in active_pitches]
        normalized_set = set(normalized_pitches)
        
        # Try major chords
        for chord_name, template in self.chord_templates.items():
            template_set = set(template)
            if template_set.issubset(normalized_set):
                return chord_name
        
        # Try minor chords
        for chord_name, template in self.minor_chord_templates.items():
            template_set = set(template)
            if template_set.issubset(normalized_set):
                return chord_name
        
        return 'N.C.'
    
    def extract_chord_progression(self, midi: 'pretty_midi.PrettyMIDI', time_step: float = 0.5) -> List[str]:
        """
        Extract chord progression over time.
        
        Args:
            midi: PrettyMIDI object
            time_step: Time step in seconds for chord detection
            
        Returns:
            List of chord names
        """
        duration = midi.get_end_time()
        chord_progression = []
        
        for time in np.arange(0, duration, time_step):
            active_pitches = self.get_active_pitches(midi, time)
            chord = self.detect_chord(active_pitches)
            chord_progression.append(chord)
        
        return chord_progression
    
    def get_chord_statistics(self, chord_progression: List[str]) -> Dict[str, any]:
        """
        Get statistics from chord progression.
        
        Args:
            chord_progression: List of chord names
            
        Returns:
            Dictionary with chord statistics
        """
        # Count chords
        chord_counts = Counter(chord_progression)
        
        # Calculate statistics
        total_chords = len(chord_progression)
        unique_chords = len([c for c in chord_counts.keys() if c != 'N.C.'])
        no_chord_ratio = chord_counts.get('N.C.', 0) / total_chords if total_chords > 0 else 0
        
        # Get most common chords
        most_common = chord_counts.most_common(5)
        
        stats = {
            'total_chords': total_chords,
            'unique_chords': unique_chords,
            'no_chord_ratio': no_chord_ratio,
            'most_common_chords': most_common,
            'chord_counts': dict(chord_counts)
        }
        
        return stats
    
    def extract_chord_features(self, midi: 'pretty_midi.PrettyMIDI', time_step: float = 0.5) -> Dict[str, any]:
        """
        Extract comprehensive chord features from MIDI.
        
        Args:
            midi: PrettyMIDI object
            time_step: Time step for chord detection
            
        Returns:
            Dictionary with chord features
        """
        chord_progression = self.extract_chord_progression(midi, time_step)
        chord_stats = self.get_chord_statistics(chord_progression)
        
        return chord_stats
    
    def get_key_signature_estimation(self, midi: 'pretty_midi.PrettyMIDI') -> str:
        """
        Estimate key signature from pitch distribution.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Estimated key signature
        """
        # Collect all pitches
        all_pitches = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                all_pitches.append(note.pitch)
        
        if not all_pitches:
            return 'C'
        
        # Normalize to 0-11 range
        normalized_pitches = [p % 12 for p in all_pitches]
        pitch_counts = Counter(normalized_pitches)
        
        # Simple estimation based on most common pitch
        most_common_pitch = pitch_counts.most_common(1)[0][0]
        
        # Map to likely keys (simplified)
        key_map = {
            0: 'C', 1: 'Db', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'
        }
        
        return key_map.get(most_common_pitch, 'C')


def extract_chord_features(midi: 'pretty_midi.PrettyMIDI', time_step: float = 0.5) -> Dict[str, any]:
    """
    Convenience function to extract chord features from MIDI.
    
    Args:
        midi: PrettyMIDI object
        time_step: Time step for chord detection
        
    Returns:
        Dictionary with chord features
    """
    extractor = ChordFeatureExtractor()
    return extractor.extract_chord_features(midi, time_step)
