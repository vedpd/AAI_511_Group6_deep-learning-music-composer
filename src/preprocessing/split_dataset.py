"""
Dataset splitting utilities for the Composer Classification project.
Handles train/validation/test splitting with stratification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from src.utils.config import (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, 
                           RANDOM_STATE, COMPOSERS, COMPOSER_TO_LABEL)
from src.utils.helpers import set_random_seed, ensure_dir


class DatasetSplitter:
    """
    Dataset splitter for train/validation/test splits.
    """
    
    def __init__(self, metadata_df: pd.DataFrame, 
                 train_split: float = TRAIN_SPLIT,
                 val_split: float = VAL_SPLIT,
                 test_split: float = TEST_SPLIT,
                 random_state: int = RANDOM_STATE):
        """
        Initialize dataset splitter.
        
        Args:
            metadata_df: DataFrame with metadata
            train_split: Proportion of training data
            val_split: Proportion of validation data
            test_split: Proportion of test data
            random_state: Random seed for reproducibility
        """
        set_random_seed(random_state)
        
        self.metadata_df = metadata_df.copy()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Validate splits sum to 1
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def stratified_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split by composer.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            self.metadata_df,
            test_size=self.test_split,
            stratify=self.metadata_df['label'],
            random_state=self.random_state
        )
        
        # Calculate adjusted train split for the remaining data
        adjusted_train_split = self.train_split / (self.train_split + self.val_split)
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=1 - adjusted_train_split,
            stratify=train_val_df['label'],
            random_state=self.random_state
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return train_df, val_df, test_df
    
    def random_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform random split (non-stratified).
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            self.metadata_df,
            test_size=self.test_split,
            random_state=self.random_state
        )
        
        # Calculate adjusted train split for the remaining data
        adjusted_train_split = self.train_split / (self.train_split + self.val_split)
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=1 - adjusted_train_split,
            random_state=self.random_state
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return train_df, val_df, test_df
    
    def get_split_statistics(self) -> dict:
        """
        Get statistics for each split.
        
        Returns:
            Dictionary with split statistics
        """
        stats = {
            'train': self._get_split_stats(self.train_df, 'Training'),
            'val': self._get_split_stats(self.val_df, 'Validation'),
            'test': self._get_split_stats(self.test_df, 'Test')
        }
        
        return stats
    
    def _get_split_stats(self, df: pd.DataFrame, split_name: str) -> dict:
        """
        Get statistics for a single split.
        
        Args:
            df: DataFrame for the split
            split_name: Name of the split
            
        Returns:
            Dictionary with statistics
        """
        if df is None:
            return {'name': split_name, 'total': 0, 'composers': {}}
        
        stats = {
            'name': split_name,
            'total': len(df),
            'composers': {}
        }
        
        for composer in COMPOSERS:
            composer_df = df[df['composer'] == composer]
            stats['composers'][composer] = len(composer_df)
        
        return stats
    
    def print_split_statistics(self) -> None:
        """Print split statistics to console."""
        stats = self.get_split_statistics()
        
        print("\nDataset Split Statistics:")
        print("=" * 50)
        
        for split_name, split_stats in stats.items():
            print(f"\n{split_stats['name']} Set:")
            print(f"  Total: {split_stats['total']} files")
            print(f"  Percentage: {split_stats['total'] / len(self.metadata_df) * 100:.1f}%")
            print("  Composers:")
            for composer, count in split_stats['composers'].items():
                percentage = count / split_stats['total'] * 100 if split_stats['total'] > 0 else 0
                print(f"    {composer}: {count} ({percentage:.1f}%)")
    
    def save_splits(self, output_dir: Path) -> None:
        """
        Save split metadata to CSV files.
        
        Args:
            output_dir: Directory to save split files
        """
        ensure_dir(output_dir)
        
        if self.train_df is not None:
            self.train_df.to_csv(output_dir / 'train_metadata.csv', index=False)
            print(f"Saved training metadata to {output_dir / 'train_metadata.csv'}")
        
        if self.val_df is not None:
            self.val_df.to_csv(output_dir / 'val_metadata.csv', index=False)
            print(f"Saved validation metadata to {output_dir / 'val_metadata.csv'}")
        
        if self.test_df is not None:
            self.test_df.to_csv(output_dir / 'test_metadata.csv', index=False)
            print(f"Saved test metadata to {output_dir / 'test_metadata.csv'}")
    
    def load_splits(self, input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load split metadata from CSV files.
        
        Args:
            input_dir: Directory containing split files
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_path = input_dir / 'train_metadata.csv'
        val_path = input_dir / 'val_metadata.csv'
        test_path = input_dir / 'test_metadata.csv'
        
        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
        else:
            self.train_df = None
        
        if val_path.exists():
            self.val_df = pd.read_csv(val_path)
        else:
            self.val_df = None
        
        if test_path.exists():
            self.test_df = pd.read_csv(test_path)
        else:
            self.test_df = None
        
        return self.train_df, self.val_df, self.test_df
    
    def get_file_paths(self, split: str) -> List[Path]:
        """
        Get file paths for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            List of file paths
        """
        if split == 'train' and self.train_df is not None:
            return [Path(fp) for fp in self.train_df['filepath'].tolist()]
        elif split == 'val' and self.val_df is not None:
            return [Path(fp) for fp in self.val_df['filepath'].tolist()]
        elif split == 'test' and self.test_df is not None:
            return [Path(fp) for fp in self.test_df['filepath'].tolist()]
        else:
            return []
    
    def get_labels(self, split: str) -> np.ndarray:
        """
        Get labels for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            Array of labels
        """
        if split == 'train' and self.train_df is not None:
            return self.train_df['label'].values
        elif split == 'val' and self.val_df is not None:
            return self.val_df['label'].values
        elif split == 'test' and self.test_df is not None:
            return self.test_df['label'].values
        else:
            return np.array([])


def split_dataset(metadata_df: pd.DataFrame,
                 train_split: float = TRAIN_SPLIT,
                 val_split: float = VAL_SPLIT,
                 test_split: float = TEST_SPLIT,
                 stratified: bool = True,
                 random_state: int = RANDOM_STATE,
                 output_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to split dataset with standard parameters.
    
    Args:
        metadata_df: DataFrame with metadata
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        stratified: Whether to use stratified splitting
        random_state: Random seed for reproducibility
        output_dir: Directory to save split files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    set_random_seed(random_state)
    
    splitter = DatasetSplitter(metadata_df, train_split, val_split, test_split, random_state)
    
    if stratified:
        train_df, val_df, test_df = splitter.stratified_split()
    else:
        train_df, val_df, test_df = splitter.random_split()
    
    splitter.print_split_statistics()
    
    if output_dir:
        splitter.save_splits(output_dir)
    
    return train_df, val_df, test_df
