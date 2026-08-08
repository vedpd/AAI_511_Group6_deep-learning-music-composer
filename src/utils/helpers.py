"""
Helper functions for the Composer Classification project.
Contains utility functions for data processing, visualization, and general operations.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

from .config import RANDOM_SEED, FIGURES_DIR, TABLES_DIR, FIGURE_DPI, FIGURE_FORMAT


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(directory: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_training_history(history: Dict[str, List[float]], 
                         model_name: str,
                         save_fig: bool = True) -> None:
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Training history dictionary
        model_name: Name of the model for title
        save_fig: Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_fig:
        filepath = FIGURES_DIR / f"{model_name}_training_history.{FIGURE_FORMAT}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved training history plot to {filepath}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         model_name: str,
                         save_fig: bool = True) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model for title
        save_fig: Whether to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_fig:
        filepath = FIGURES_DIR / f"{model_name}_confusion_matrix.{FIGURE_FORMAT}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved confusion matrix to {filepath}")
    
    plt.show()


def plot_class_distribution(labels: np.ndarray, 
                          class_names: List[str],
                          title: str = "Class Distribution",
                          save_fig: bool = True) -> None:
    """
    Plot class distribution.
    
    Args:
        labels: Array of class labels
        class_names: List of class names
        title: Plot title
        save_fig: Whether to save the figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(unique)), counts)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique])
    plt.title(title)
    plt.xlabel('Composer')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        safe_title = title.replace(' ', '_').replace('/', '_')
        filepath = FIGURES_DIR / f"{safe_title}.{FIGURE_FORMAT}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved class distribution plot to {filepath}")
    
    plt.show()


def plot_metric_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          title: str = "Model Comparison",
                          save_fig: bool = True) -> None:
    """
    Plot metric comparison between models.
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        title: Plot title
        save_fig: Whether to save the figure
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        offset = width * (i - (len(models) - 1) / 2)
        ax.bar(x + offset, values, width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        safe_title = title.replace(' ', '_')
        filepath = FIGURES_DIR / f"{safe_title}.{FIGURE_FORMAT}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved metric comparison plot to {filepath}")
    
    plt.show()


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "") -> None:
    """
    Print a progress bar to console.
    
    Args:
        current: Current progress
        total: Total items
        prefix: Prefix string
        suffix: Suffix string
    """
    percent = 100 * (current / float(total))
    filled_length = int(50 * current // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def get_file_list(directory: Path, extension: str = ".mid") -> List[Path]:
    """
    Get list of files with specific extension from directory.
    
    Args:
        directory: Directory to search
        extension: File extension to match
        
    Returns:
        List of file paths
    """
    return list(directory.glob(f"*{extension}")) + list(directory.glob(f"*{extension.upper()}"))


def count_parameters(model) -> int:
    """
    Count total parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Total number of parameters
    """
    return sum([np.prod(p.shape) for p in model.trainable_weights])
