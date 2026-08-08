"""
Piano roll feature extraction for CNN models.
Converts MIDI files to piano roll matrices.
"""

import numpy as np
from typing import Tuple
from copy import deepcopy

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import MAX_TIME_STEPS, PITCH_RANGE, RANDOM_SEED
from src.utils.helpers import set_random_seed


class PianoRollExtractor:
    """
    Extract piano roll representations for CNN models.
    """
    
    def __init__(self, 
                 time_steps: int = MAX_TIME_STEPS,
                 pitch_range: int = PITCH_RANGE,
                 fps: float = 20.0,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize piano roll extractor.
        
        Args:
            time_steps: Number of time steps
            pitch_range: Number of pitches (MIDI has 128)
            fps: Frames per second for time discretization
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        self.time_steps = time_steps
        self.pitch_range = pitch_range
        self.fps = fps
    
    def extract_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract piano roll from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Piano roll matrix of shape (pitch_range, time_steps)
        """
        # Get duration in seconds
        duration = midi.get_end_time()
        
        # Calculate number of time steps
        num_time_steps = int(duration * self.fps)
        
        # Initialize piano roll
        piano_roll = np.zeros((self.pitch_range, num_time_steps), dtype=np.float32)
        
        # Fill piano roll
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Convert time to frame indices
                start_frame = int(note.start * self.fps)
                end_frame = int(note.end * self.fps)
                
                # Ensure within bounds
                start_frame = max(0, min(start_frame, num_time_steps - 1))
                end_frame = max(0, min(end_frame, num_time_steps - 1))
                
                # Set velocity in piano roll
                if note.pitch < self.pitch_range:
                    piano_roll[note.pitch, start_frame:end_frame] = note.velocity / 127.0
        
        return piano_roll
    
    def resize_piano_roll(self, piano_roll: np.ndarray) -> np.ndarray:
        """
        Resize piano roll to fixed time steps using interpolation.
        
        Args:
            piano_roll: Piano roll matrix of shape (pitch_range, original_time_steps)
            
        Returns:
            Resized piano roll of shape (pitch_range, time_steps)
        """
        original_time_steps = piano_roll.shape[1]
        
        if original_time_steps == self.time_steps:
            return piano_roll
        elif original_time_steps < self.time_steps:
            # Upsample by repeating
            repeat_factor = self.time_steps // original_time_steps
            remainder = self.time_steps % original_time_steps
            
            if repeat_factor > 0:
                resized = np.repeat(piano_roll, repeat_factor, axis=1)
            else:
                resized = piano_roll.copy()
            
            if remainder > 0:
                # Add remainder by padding with last column
                padding = np.tile(piano_roll[:, -1:], (1, remainder))
                resized = np.hstack([resized, padding])
            
            return resized[:, :self.time_steps]
        else:
            # Downsample by averaging
            from scipy import signal
            downsample_factor = original_time_steps // self.time_steps
            
            # Create averaging kernel
            kernel = np.ones(downsample_factor) / downsample_factor
            
            # Apply to each pitch
            resized = np.zeros((self.pitch_range, self.time_steps), dtype=np.float32)
            for pitch in range(self.pitch_range):
                resized[pitch] = signal.resample(piano_roll[pitch], self.time_steps)
            
            return resized
    
    def extract_binary_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract binary piano roll (1 if note is active, 0 otherwise).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Binary piano roll matrix
        """
        piano_roll = self.extract_piano_roll(midi)
        binary_roll = (piano_roll > 0).astype(np.float32)
        return binary_roll
    
    def extract_velocity_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract velocity-weighted piano roll.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Velocity-weighted piano roll matrix
        """
        return self.extract_piano_roll(midi)
    
    def extract_onset_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract onset-only piano roll (1 at note onset, 0 elsewhere).
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Onset piano roll matrix
        """
        duration = midi.get_end_time()
        num_time_steps = int(duration * self.fps)
        
        # Initialize onset piano roll
        onset_roll = np.zeros((self.pitch_range, num_time_steps), dtype=np.float32)
        
        # Fill onset piano roll
        for instrument in midi.instruments:
            for note in instrument.notes:
                start_frame = int(note.start * self.fps)
                start_frame = max(0, min(start_frame, num_time_steps - 1))
                
                if note.pitch < self.pitch_range:
                    onset_roll[note.pitch, start_frame] = 1.0
        
        return onset_roll
    
    def extract_combined_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract combined piano roll with velocity and onset information.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Combined piano roll of shape (pitch_range, time_steps, 2)
        """
        velocity_roll = self.extract_piano_roll(midi)
        onset_roll = self.extract_onset_piano_roll(midi)
        
        # Resize both to same dimensions
        velocity_roll = self.resize_piano_roll(velocity_roll)
        onset_roll = self.resize_piano_roll(onset_roll)
        
        # Combine
        combined = np.stack([velocity_roll, onset_roll], axis=-1)
        
        return combined
    
    def extract_multi_channel_piano_roll(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract multi-channel piano roll for different instruments.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Multi-channel piano roll of shape (pitch_range, time_steps, num_instruments)
        """
        duration = midi.get_end_time()
        num_time_steps = int(duration * self.fps)
        num_instruments = len(midi.instruments)
        
        # Initialize multi-channel piano roll
        multi_roll = np.zeros((self.pitch_range, num_time_steps, num_instruments), dtype=np.float32)
        
        # Fill piano roll for each instrument
        for inst_idx, instrument in enumerate(midi.instruments):
            for note in instrument.notes:
                start_frame = int(note.start * self.fps)
                end_frame = int(note.end * self.fps)
                
                start_frame = max(0, min(start_frame, num_time_steps - 1))
                end_frame = max(0, min(end_frame, num_time_steps - 1))
                
                if note.pitch < self.pitch_range:
                    multi_roll[note.pitch, start_frame:end_frame, inst_idx] = note.velocity / 127.0
        
        # Resize time dimension
        resized = np.zeros((self.pitch_range, self.time_steps, num_instruments), dtype=np.float32)
        for inst_idx in range(num_instruments):
            resized[:, :, inst_idx] = self.resize_piano_roll(multi_roll[:, :, inst_idx])
        
        return resized


def extract_piano_roll(midi: pretty_midi.PrettyMIDI,
                      roll_type: str = 'velocity',
                      time_steps: int = MAX_TIME_STEPS,
                      pitch_range: int = PITCH_RANGE,
                      fps: float = 20.0) -> np.ndarray:
    """
    Convenience function to extract piano roll from MIDI.
    
    Args:
        midi: PrettyMIDI object
        roll_type: Type of piano roll ('velocity', 'binary', 'onset', 'combined', 'multi')
        time_steps: Number of time steps
        pitch_range: Number of pitches
        fps: Frames per second
        
    Returns:
        Piano roll matrix
    """
    extractor = PianoRollExtractor(time_steps, pitch_range, fps)
    
    if roll_type == 'velocity':
        piano_roll = extractor.extract_velocity_piano_roll(midi)
        piano_roll = extractor.resize_piano_roll(piano_roll)
    elif roll_type == 'binary':
        piano_roll = extractor.extract_binary_piano_roll(midi)
        piano_roll = extractor.resize_piano_roll(piano_roll)
    elif roll_type == 'onset':
        piano_roll = extractor.extract_onset_piano_roll(midi)
        piano_roll = extractor.resize_piano_roll(piano_roll)
    elif roll_type == 'combined':
        piano_roll = extractor.extract_combined_piano_roll(midi)
    elif roll_type == 'multi':
        piano_roll = extractor.extract_multi_channel_piano_roll(midi)
    else:
        raise ValueError(f"Unknown piano roll type: {roll_type}")
    
    return piano_roll
