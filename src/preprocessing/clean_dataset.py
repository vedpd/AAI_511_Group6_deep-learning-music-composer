"""
Dataset cleaning utilities for the Composer Classification project.
Handles filtering of invalid MIDI files and dataset validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

from src.utils.config import METADATA_FILE, MIN_NOTES, COMPOSERS
from src.utils.helpers import set_random_seed, ensure_dir


class DatasetCleaner:
    """
    Dataset cleaner for filtering and validating MIDI files.
    """
    
    def __init__(self, metadata_path: Path = METADATA_FILE):
        """
        Initialize dataset cleaner.
        
        Args:
            metadata_path: Path to metadata CSV file
        """
        self.metadata_path = metadata_path
        self.metadata_df = None
        self.valid_metadata = None
        self.invalid_metadata = None
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from CSV file.
        
        Returns:
            DataFrame with metadata
        """
        self.metadata_df = pd.read_csv(self.metadata_path)
        return self.metadata_df
    
    def filter_by_note_count(self, min_notes: int = MIN_NOTES) -> pd.DataFrame:
        """
        Filter files by minimum note count.
        
        Args:
            min_notes: Minimum number of notes required
            
        Returns:
            DataFrame with filtered metadata
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        self.valid_metadata = self.metadata_df[self.metadata_df['num_notes'] >= min_notes].copy()
        self.invalid_metadata = self.metadata_df[self.metadata_df['num_notes'] < min_notes].copy()
        
        print(f"Filtered {len(self.invalid_metadata)} files with < {min_notes} notes")
        print(f"Retained {len(self.valid_metadata)} valid files")
        
        return self.valid_metadata
    
    def filter_by_duration(self, min_duration: float = 1.0, max_duration: float = 600.0) -> pd.DataFrame:
        """
        Filter files by duration range.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            DataFrame with filtered metadata
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        if self.valid_metadata is not None:
            df = self.valid_metadata
        else:
            df = self.metadata_df
        
        filtered = df[(df['duration'] >= min_duration) & (df['duration'] <= max_duration)].copy()
        excluded = df[(df['duration'] < min_duration) | (df['duration'] > max_duration)].copy()
        
        print(f"Filtered {len(excluded)} files with duration outside [{min_duration}, {max_duration}] seconds")
        print(f"Retained {len(filtered)} valid files")
        
        self.valid_metadata = filtered
        if self.invalid_metadata is not None:
            self.invalid_metadata = pd.concat([self.invalid_metadata, excluded])
        else:
            self.invalid_metadata = excluded
        
        return self.valid_metadata
    
    def filter_by_composer(self, composers: List[str] = COMPOSERS) -> pd.DataFrame:
        """
        Filter files by composer list.
        
        Args:
            composers: List of composer names to keep
            
        Returns:
            DataFrame with filtered metadata
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        if self.valid_metadata is not None:
            df = self.valid_metadata
        else:
            df = self.metadata_df
        
        filtered = df[df['composer'].isin(composers)].copy()
        excluded = df[~df['composer'].isin(composers)].copy()
        
        print(f"Filtered {len(excluded)} files from non-specified composers")
        print(f"Retained {len(filtered)} files from specified composers")
        
        self.valid_metadata = filtered
        if self.invalid_metadata is not None:
            self.invalid_metadata = pd.concat([self.invalid_metadata, excluded])
        else:
            self.invalid_metadata = excluded
        
        return self.valid_metadata
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate files based on filename.
        
        Returns:
            DataFrame with unique files
        """
        if self.valid_metadata is None:
            if self.metadata_df is None:
                self.load_metadata()
            self.valid_metadata = self.metadata_df.copy()
        
        before_count = len(self.valid_metadata)
        self.valid_metadata = self.valid_metadata.drop_duplicates(subset=['filename'], keep='first')
        after_count = len(self.valid_metadata)
        
        print(f"Removed {before_count - after_count} duplicate files")
        print(f"Retained {after_count} unique files")
        
        return self.valid_metadata
    
    def validate_files(self) -> pd.DataFrame:
        """
        Validate that all marked files exist and can be loaded.
        
        Returns:
            DataFrame with validated metadata
        """
        if self.valid_metadata is None:
            if self.metadata_df is None:
                self.load_metadata()
            self.valid_metadata = self.metadata_df.copy()
        
        valid_files = []
        invalid_files = []
        
        for idx, row in tqdm(self.valid_metadata.iterrows(), total=len(self.valid_metadata), desc="Validating files"):
            filepath = Path(row['filepath'])
            if filepath.exists():
                valid_files.append(row)
            else:
                invalid_files.append(row)
        
        self.valid_metadata = pd.DataFrame(valid_files)
        self.invalid_metadata = pd.concat([self.invalid_metadata, pd.DataFrame(invalid_files)]) if self.invalid_metadata is not None else pd.DataFrame(invalid_files)
        
        print(f"Found {len(invalid_files)} missing files")
        print(f"Retained {len(valid_files)} existing files")
        
        return self.valid_metadata
    
    def balance_dataset(self, method: str = 'undersample', max_samples_per_class: int = None) -> pd.DataFrame:
        """
        Balance dataset by composer.
        
        Args:
            method: 'undersample' or 'oversample'
            max_samples_per_class: Maximum samples per class (for undersample)
            
        Returns:
            DataFrame with balanced metadata
        """
        if self.valid_metadata is None:
            if self.metadata_df is None:
                self.load_metadata()
            self.valid_metadata = self.metadata_df.copy()
        
        # Get samples per composer
        composer_counts = self.valid_metadata['composer'].value_counts()
        min_count = composer_counts.min()
        
        if method == 'undersample':
            target_count = max_samples_per_class if max_samples_per_class else min_count
            balanced_dfs = []
            
            for composer in COMPOSERS:
                composer_df = self.valid_metadata[self.valid_metadata['composer'] == composer]
                if len(composer_df) > target_count:
                    composer_df = composer_df.sample(n=target_count, random_state=42)
                balanced_dfs.append(composer_df)
            
            self.valid_metadata = pd.concat(balanced_dfs)
            
        elif method == 'oversample':
            max_count = composer_counts.max()
            balanced_dfs = []
            
            for composer in COMPOSERS:
                composer_df = self.valid_metadata[self.valid_metadata['composer'] == composer]
                while len(composer_df) < max_count:
                    composer_df = pd.concat([composer_df, composer_df.sample(max_count - len(composer_df), replace=True)])
                if len(composer_df) > max_count:
                    composer_df = composer_df.sample(n=max_count, random_state=42)
                balanced_dfs.append(composer_df)
            
            self.valid_metadata = pd.concat(balanced_dfs)
        
        print(f"Balanced dataset: {len(self.valid_metadata)} total files")
        print("Samples per composer:")
        print(self.valid_metadata['composer'].value_counts())
        
        return self.valid_metadata
    
    def get_cleaning_report(self) -> Dict[str, any]:
        """
        Generate a cleaning report.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if self.metadata_df is None:
            self.load_metadata()
        
        report = {
            'original_count': len(self.metadata_df),
            'valid_count': len(self.valid_metadata) if self.valid_metadata is not None else 0,
            'invalid_count': len(self.invalid_metadata) if self.invalid_metadata is not None else 0,
            'composers': {}
        }
        
        if self.valid_metadata is not None:
            for composer in COMPOSERS:
                composer_df = self.valid_metadata[self.valid_metadata['composer'] == composer]
                report['composers'][composer] = len(composer_df)
        
        return report
    
    def save_cleaned_metadata(self, output_path: Path) -> None:
        """
        Save cleaned metadata to CSV.
        
        Args:
            output_path: Path to save cleaned metadata
        """
        ensure_dir(output_path.parent)
        if self.valid_metadata is not None:
            self.valid_metadata.to_csv(output_path, index=False)
            print(f"Saved cleaned metadata to {output_path}")
        else:
            print("No valid metadata to save")


def clean_dataset(metadata_path: Path = METADATA_FILE,
                 min_notes: int = MIN_NOTES,
                 min_duration: float = 1.0,
                 max_duration: float = 600.0,
                 balance: bool = True,
                 balance_method: str = 'undersample',
                 output_path: Path = None) -> pd.DataFrame:
    """
    Convenience function to clean dataset with standard parameters.
    
    Args:
        metadata_path: Path to metadata CSV
        min_notes: Minimum number of notes
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        balance: Whether to balance the dataset
        balance_method: Method for balancing ('undersample' or 'oversample')
        output_path: Path to save cleaned metadata
        
    Returns:
        DataFrame with cleaned metadata
    """
    set_random_seed()
    
    cleaner = DatasetCleaner(metadata_path)
    cleaner.load_metadata()
    
    print("Starting dataset cleaning...")
    print(f"Original dataset: {len(cleaner.metadata_df)} files")
    
    # Apply filters
    cleaner.filter_by_note_count(min_notes)
    cleaner.filter_by_duration(min_duration, max_duration)
    cleaner.filter_by_composer(COMPOSERS)
    cleaner.remove_duplicates()
    cleaner.validate_files()
    
    if balance:
        cleaner.balance_dataset(method=balance_method)
    
    # Generate report
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    print(f"Original files: {report['original_count']}")
    print(f"Valid files: {report['valid_count']}")
    print(f"Invalid files: {report['invalid_count']}")
    print("Files per composer:")
    for composer, count in report['composers'].items():
        print(f"  {composer}: {count}")
    
    # Save cleaned metadata
    if output_path:
        cleaner.save_cleaned_metadata(output_path)
    
    return cleaner.valid_metadata
