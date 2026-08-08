"""
MIDI file loader for the Composer Classification project.
Handles loading and basic validation of MIDI files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    warnings.warn("pretty_midi not available. MIDI loading will be limited.")

try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    warnings.warn("music21 not available. Alternative MIDI loading will be limited.")

from src.utils.config import RAW_DATA_DIR, COMPOSERS, COMPOSER_TO_LABEL, METADATA_FILE, MIN_NOTES
from src.utils.helpers import set_random_seed, ensure_dir, get_file_list


class MIDILoader:
    """
    MIDI file loader with metadata extraction.
    """
    
    def __init__(self, data_dir: Path = RAW_DATA_DIR, composers: List[str] = COMPOSERS):
        """
        Initialize MIDI loader.
        
        Args:
            data_dir: Base directory containing composer folders
            composers: List of composer names to process
        """
        self.data_dir = data_dir
        self.composers = composers
        self.metadata = []
        
    def load_midi_file(self, filepath: Path) -> Optional[pretty_midi.PrettyMIDI]:
        """
        Load a single MIDI file using pretty_midi.
        
        Args:
            filepath: Path to MIDI file
            
        Returns:
            PrettyMIDI object or None if loading fails
        """
        if not PRETTY_MIDI_AVAILABLE:
            raise ImportError("pretty_midi is required for MIDI loading")
        
        try:
            midi = pretty_midi.PrettyMIDI(str(filepath))
            return midi
        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}")
            return None
    
    def extract_metadata(self, midi: pretty_midi.PrettyMIDI, filepath: Path, composer: str) -> Dict[str, any]:
        """
        Extract metadata from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            filepath: Path to MIDI file
            composer: Composer name
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Get basic information
            duration = midi.get_end_time()
            num_notes = sum(len(instrument.notes) for instrument in midi.instruments)
            
            # Estimate tempo (if available)
            try:
                tempo_changes = midi.get_tempo_changes()
                if len(tempo_changes[1]) > 0:
                    tempo = tempo_changes[1][0]
                else:
                    tempo = 120.0  # Default tempo
            except:
                tempo = 120.0
            
            # Count instruments
            num_instruments = len(midi.instruments)
            
            metadata = {
                'filename': filepath.name,
                'filepath': str(filepath),
                'composer': composer,
                'label': COMPOSER_TO_LABEL[composer],
                'duration': duration,
                'num_notes': num_notes,
                'tempo': tempo,
                'num_instruments': num_instruments,
                'valid': num_notes >= MIN_NOTES
            }
            
            return metadata
            
        except Exception as e:
            warnings.warn(f"Failed to extract metadata from {filepath}: {e}")
            return {
                'filename': filepath.name,
                'filepath': str(filepath),
                'composer': composer,
                'label': COMPOSER_TO_LABEL[composer],
                'duration': 0,
                'num_notes': 0,
                'tempo': 0,
                'num_instruments': 0,
                'valid': False
            }
    
    def load_dataset(self, extract_metadata: bool = True) -> List[Dict[str, any]]:
        """
        Load all MIDI files from the dataset.
        
        Args:
            extract_metadata: Whether to extract metadata from each file
            
        Returns:
            List of metadata dictionaries
        """
        self.metadata = []
        
        for composer in self.composers:
            composer_dir = self.data_dir / composer
            if not composer_dir.exists():
                warnings.warn(f"Directory not found: {composer_dir}")
                continue
            
            midi_files = get_file_list(composer_dir, ".mid")
            print(f"Found {len(midi_files)} MIDI files for {composer}")
            
            for filepath in tqdm(midi_files, desc=f"Loading {composer}"):
                if extract_metadata:
                    midi = self.load_midi_file(filepath)
                    if midi is not None:
                        metadata = self.extract_metadata(midi, filepath, composer)
                        self.metadata.append(metadata)
                else:
                    self.metadata.append({
                        'filename': filepath.name,
                        'filepath': str(filepath),
                        'composer': composer,
                        'label': COMPOSER_TO_LABEL[composer],
                        'valid': True
                    })
        
        return self.metadata
    
    def save_metadata(self, output_path: Path = METADATA_FILE) -> None:
        """
        Save metadata to CSV file.
        
        Args:
            output_path: Path to save metadata CSV
        """
        ensure_dir(output_path.parent)
        df = pd.DataFrame(self.metadata)
        df.to_csv(output_path, index=False)
        print(f"Saved metadata to {output_path}")
    
    def load_metadata(self, input_path: Path = METADATA_FILE) -> pd.DataFrame:
        """
        Load metadata from CSV file.
        
        Args:
            input_path: Path to metadata CSV
            
        Returns:
            DataFrame with metadata
        """
        df = pd.read_csv(input_path)
        self.metadata = df.to_dict('records')
        return df
    
    def get_valid_files(self) -> List[Path]:
        """
        Get list of valid MIDI file paths.
        
        Returns:
            List of valid file paths
        """
        return [Path(m['filepath']) for m in self.metadata if m['valid']]
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        df = pd.DataFrame(self.metadata)
        
        stats = {
            'total_files': len(df),
            'valid_files': len(df[df['valid']]),
            'invalid_files': len(df[~df['valid']]),
            'composers': {}
        }
        
        for composer in self.composers:
            composer_df = df[df['composer'] == composer]
            stats['composers'][composer] = {
                'total': len(composer_df),
                'valid': len(composer_df[composer_df['valid']]),
                'invalid': len(composer_df[~composer_df['valid']]),
                'avg_duration': composer_df['duration'].mean() if len(composer_df) > 0 else 0,
                'avg_notes': composer_df['num_notes'].mean() if len(composer_df) > 0 else 0,
                'avg_tempo': composer_df['tempo'].mean() if len(composer_df) > 0 else 0
            }
        
        return stats


def load_midi_dataset(data_dir: Path = RAW_DATA_DIR, 
                     composers: List[str] = COMPOSERS,
                     save_metadata: bool = True) -> Tuple[List[Dict[str, any]], Dict[str, any]]:
    """
    Convenience function to load MIDI dataset and extract metadata.
    
    Args:
        data_dir: Base directory containing composer folders
        composers: List of composer names to process
        save_metadata: Whether to save metadata to CSV
        
    Returns:
        Tuple of (metadata list, statistics dictionary)
    """
    set_random_seed()
    
    loader = MIDILoader(data_dir, composers)
    metadata = loader.load_dataset(extract_metadata=True)
    stats = loader.get_statistics()
    
    if save_metadata:
        loader.save_metadata()
    
    return metadata, stats
