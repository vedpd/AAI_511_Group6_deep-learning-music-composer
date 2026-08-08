"""
Data augmentation utilities for MIDI files.
Implements pitch shifting, tempo scaling, velocity variation, and time shifting.
"""

import numpy as np
import random
from typing import List, Tuple
from copy import deepcopy

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from src.utils.config import (AUGMENTATION_ENABLED, PITCH_SHIFT_RANGE, 
                           TEMPO_SCALE_RANGE, VELOCITY_VARIATION_RANGE, 
                           TIME_SHIFT_MAX, RANDOM_SEED)
from src.utils.helpers import set_random_seed


class MIDIAugmentation:
    """
    MIDI data augmentation class.
    """
    
    def __init__(self, 
                 pitch_shift_range: Tuple[int, int] = PITCH_SHIFT_RANGE,
                 tempo_scale_range: Tuple[float, float] = TEMPO_SCALE_RANGE,
                 velocity_variation_range: Tuple[float, float] = VELOCITY_VARIATION_RANGE,
                 time_shift_max: float = TIME_SHIFT_MAX,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize MIDI augmentation.
        
        Args:
            pitch_shift_range: Range of pitch shift in semitones (min, max)
            tempo_scale_range: Range of tempo scaling (min, max)
            velocity_variation_range: Range of velocity variation (min, max)
            time_shift_max: Maximum time shift as fraction of sequence length
            random_seed: Random seed for reproducibility
        """
        set_random_seed(random_seed)
        
        self.pitch_shift_range = pitch_shift_range
        self.tempo_scale_range = tempo_scale_range
        self.velocity_variation_range = velocity_variation_range
        self.time_shift_max = time_shift_max
        self.enabled = AUGMENTATION_ENABLED
    
    def pitch_shift(self, midi: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
        """
        Shift pitch of all notes by semitones.
        
        Args:
            midi: PrettyMIDI object
            semitones: Number of semitones to shift (positive = up, negative = down)
            
        Returns:
            Augmented PrettyMIDI object
        """
        augmented = deepcopy(midi)
        
        for instrument in augmented.instruments:
            for note in instrument.notes:
                note.pitch += semitones
                # Ensure pitch stays within valid MIDI range
                note.pitch = max(0, min(127, note.pitch))
        
        return augmented
    
    def random_pitch_shift(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply random pitch shift within configured range.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Augmented PrettyMIDI object
        """
        semitones = random.randint(self.pitch_shift_range[0], self.pitch_shift_range[1])
        return self.pitch_shift(midi, semitones)
    
    def tempo_scale(self, midi: pretty_midi.PrettyMIDI, scale_factor: float) -> pretty_midi.PrettyMIDI:
        """
        Scale tempo (note timing) by factor.
        
        Args:
            midi: PrettyMIDI object
            scale_factor: Tempo scaling factor (1.0 = no change, >1.0 = faster, <1.0 = slower)
            
        Returns:
            Augmented PrettyMIDI object
        """
        augmented = deepcopy(midi)
        
        for instrument in augmented.instruments:
            for note in instrument.notes:
                note.start *= scale_factor
                note.end *= scale_factor
        
        return augmented
    
    def random_tempo_scale(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply random tempo scaling within configured range.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Augmented PrettyMIDI object
        """
        scale_factor = random.uniform(self.tempo_scale_range[0], self.tempo_scale_range[1])
        return self.tempo_scale(midi, scale_factor)
    
    def velocity_variation(self, midi: pretty_midi.PrettyMIDI, variation_factor: float) -> pretty_midi.PrettyMIDI:
        """
        Vary note velocities by factor.
        
        Args:
            midi: PrettyMIDI object
            variation_factor: Velocity variation factor
            
        Returns:
            Augmented PrettyMIDI object
        """
        augmented = deepcopy(midi)
        
        for instrument in augmented.instruments:
            for note in instrument.notes:
                note.velocity = int(note.velocity * variation_factor)
                # Ensure velocity stays within valid MIDI range
                note.velocity = max(0, min(127, note.velocity))
        
        return augmented
    
    def random_velocity_variation(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply random velocity variation within configured range.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Augmented PrettyMIDI object
        """
        variation_factor = random.uniform(self.velocity_variation_range[0], self.velocity_variation_range[1])
        return self.velocity_variation(midi, variation_factor)
    
    def time_shift(self, midi: pretty_midi.PrettyMIDI, shift_amount: float) -> pretty_midi.PrettyMIDI:
        """
        Shift all notes in time.
        
        Args:
            midi: PrettyMIDI object
            shift_amount: Time shift in seconds (positive = later, negative = earlier)
            
        Returns:
            Augmented PrettyMIDI object
        """
        augmented = deepcopy(midi)
        
        for instrument in augmented.instruments:
            for note in instrument.notes:
                note.start += shift_amount
                note.end += shift_amount
                # Ensure timing doesn't go negative
                if note.start < 0:
                    note.end -= note.start
                    note.start = 0
        
        return augmented
    
    def random_time_shift(self, midi: pretty_midi.PrettyMIDI, max_duration: float = None) -> pretty_midi.PrettyMIDI:
        """
        Apply random time shift within configured range.
        
        Args:
            midi: PrettyMIDI object
            max_duration: Maximum duration of the MIDI for calculating shift
            
        Returns:
            Augmented PrettyMIDI object
        """
        if max_duration is None:
            max_duration = midi.get_end_time()
        
        max_shift = max_duration * self.time_shift_max
        shift_amount = random.uniform(-max_shift, max_shift)
        return self.time_shift(midi, shift_amount)
    
    def apply_random_augmentation(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply a random combination of augmentations.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Augmented PrettyMIDI object
        """
        if not self.enabled:
            return midi
        
        augmented = deepcopy(midi)
        
        # Randomly choose which augmentations to apply
        if random.random() < 0.5:  # 50% chance of pitch shift
            augmented = self.random_pitch_shift(augmented)
        
        if random.random() < 0.5:  # 50% chance of tempo scaling
            augmented = self.random_tempo_scale(augmented)
        
        if random.random() < 0.5:  # 50% chance of velocity variation
            augmented = self.random_velocity_variation(augmented)
        
        if random.random() < 0.3:  # 30% chance of time shift
            augmented = self.random_time_shift(augmented)
        
        return augmented
    
    def apply_all_augmentations(self, midi: pretty_midi.PrettyMIDI) -> List[pretty_midi.PrettyMIDI]:
        """
        Apply all augmentation techniques to create multiple augmented versions.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of augmented PrettyMIDI objects
        """
        if not self.enabled:
            return [midi]
        
        augmented_versions = []
        
        # Original
        augmented_versions.append(deepcopy(midi))
        
        # Pitch shift variations
        for semitones in range(self.pitch_shift_range[0], self.pitch_shift_range[1] + 1):
            if semitones != 0:
                augmented_versions.append(self.pitch_shift(midi, semitones))
        
        # Tempo scaling variations
        for scale in [self.tempo_scale_range[0], self.tempo_scale_range[1]]:
            if scale != 1.0:
                augmented_versions.append(self.tempo_scale(midi, scale))
        
        # Velocity variation
        for var in [self.velocity_variation_range[0], self.velocity_variation_range[1]]:
            if var != 1.0:
                augmented_versions.append(self.velocity_variation(midi, var))
        
        return augmented_versions
    
    def augment_dataset(self, midi_files: List[pretty_midi.PrettyMIDI], 
                       augment_factor: int = 2) -> List[pretty_midi.PrettyMIDI]:
        """
        Augment a dataset of MIDI files.
        
        Args:
            midi_files: List of PrettyMIDI objects
            augment_factor: Number of augmented versions to create per file
            
        Returns:
            List of augmented PrettyMIDI objects
        """
        if not self.enabled:
            return midi_files
        
        augmented_dataset = []
        
        for midi in midi_files:
            # Add original
            augmented_dataset.append(deepcopy(midi))
            
            # Add augmented versions
            for _ in range(augment_factor):
                augmented = self.apply_random_augmentation(midi)
                augmented_dataset.append(augmented)
        
        return augmented_dataset


def augment_midi(midi: pretty_midi.PrettyMIDI,
                 augmentation_type: str = 'random',
                 **kwargs) -> pretty_midi.PrettyMIDI:
    """
    Convenience function to augment a single MIDI file.
    
    Args:
        midi: PrettyMIDI object
        augmentation_type: Type of augmentation ('random', 'pitch', 'tempo', 'velocity', 'time')
        **kwargs: Additional parameters for specific augmentation types
        
    Returns:
        Augmented PrettyMIDI object
    """
    augmenter = MIDIAugmentation()
    
    if augmentation_type == 'random':
        return augmenter.apply_random_augmentation(midi)
    elif augmentation_type == 'pitch':
        semitones = kwargs.get('semitones', None)
        if semitones is None:
            return augmenter.random_pitch_shift(midi)
        else:
            return augmenter.pitch_shift(midi, semitones)
    elif augmentation_type == 'tempo':
        scale = kwargs.get('scale', None)
        if scale is None:
            return augmenter.random_tempo_scale(midi)
        else:
            return augmenter.tempo_scale(midi, scale)
    elif augmentation_type == 'velocity':
        variation = kwargs.get('variation', None)
        if variation is None:
            return augmenter.random_velocity_variation(midi)
        else:
            return augmenter.velocity_variation(midi, variation)
    elif augmentation_type == 'time':
        shift = kwargs.get('shift', None)
        if shift is None:
            return augmenter.random_time_shift(midi)
        else:
            return augmenter.time_shift(midi, shift)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
