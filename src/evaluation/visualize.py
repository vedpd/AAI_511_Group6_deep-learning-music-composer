"""
Visualization utilities for model evaluation and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd

from src.utils.config import COMPOSERS, FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, TABLES_DIR
from src.utils.helpers import ensure_dir


class ModelVisualizer:
    """
    Visualization utilities for model analysis.
    """
    
    def __init__(self, class_names: List[str] = COMPOSERS):
        """
        Initialize model visualizer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            model_name: str = "Model",
                            save_fig: bool = True) -> plt.Figure:
        """
        Plot training history (accuracy and loss).
        
        Args:
            history: Training history dictionary
            model_name: Name of the model for title
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            filepath = FIGURES_DIR / f"{model_name}_training_history.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved training history plot to {filepath}")
        
        return fig
    
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                           title: str = "Model Comparison",
                           save_fig: bool = True) -> plt.Figure:
        """
        Plot model comparison metrics.
        
        Args:
            metrics_dict: Dictionary of model names to their metrics
            title: Plot title
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            values = [metrics_dict[model].get(metric, 0) for metric in metrics]
            offset = width * (i - (len(models) - 1) / 2)
            ax.bar(x + offset, values, width, label=model)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            safe_title = title.replace(' ', '_')
            filepath = FIGURES_DIR / f"{safe_title}.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved model comparison plot to {filepath}")
        
        return fig
    
    def plot_per_class_performance(self, metrics_dict: Dict[str, Dict[str, float]],
                                 title: str = "Per-Class Performance",
                                 save_fig: bool = True) -> plt.Figure:
        """
        Plot per-class performance comparison.
        
        Args:
            metrics_dict: Dictionary containing per-class metrics
            title: Plot title
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        models = list(metrics_dict.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        metric_titles = ['Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[idx]
            
            x = np.arange(len(self.class_names))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                if metric in metrics_dict[model]:
                    values = [metrics_dict[model][metric].get(class_name, 0) 
                             for class_name in self.class_names]
                    offset = width * (i - (len(models) - 1) / 2)
                    ax.bar(x + offset, values, width, label=model)
            
            ax.set_ylabel('Score')
            ax.set_title(metric_title)
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            safe_title = title.replace(' ', '_')
            filepath = FIGURES_DIR / f"{safe_title}.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved per-class performance plot to {filepath}")
        
        return fig
    
    def plot_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model",
                         save_fig: bool = True) -> plt.Figure:
        """
        Plot error analysis (most common misclassifications).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Find misclassifications
        misclassified = y_true != y_pred
        misclassified_true = y_true[misclassified]
        misclassified_pred = y_pred[misclassified]
        
        # Count misclassification pairs
        error_pairs = []
        for true_label, pred_label in zip(misclassified_true, misclassified_pred):
            error_pairs.append((self.class_names[true_label], self.class_names[pred_label]))
        
        from collections import Counter
        error_counts = Counter(error_pairs)
        
        # Get top errors
        top_errors = error_counts.most_common(10)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if top_errors:
            error_labels = [f"{true} → {pred}" for (true, pred), count in top_errors]
            error_values = [count for (true, pred), count in top_errors]
            
            ax.barh(range(len(error_labels)), error_values)
            ax.set_yticks(range(len(error_labels)))
            ax.set_yticklabels(error_labels)
            ax.set_xlabel('Count')
            ax.set_title(f'{model_name} - Top Misclassifications')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No misclassifications!', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_name} - Error Analysis')
        
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            filepath = FIGURES_DIR / f"{model_name}_error_analysis.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved error analysis plot to {filepath}")
        
        return fig
    
    def create_comparison_table(self, metrics_dict: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create comparison table for model metrics.
        
        Args:
            metrics_dict: Dictionary of model names to their metrics
            save_path: Optional path to save table
            
        Returns:
            DataFrame with comparison table
        """
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in metrics_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}",
                'F1 Score': f"{metrics.get('f1_score', 0):.4f}"
            }
            if 'roc_auc' in metrics:
                row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_path:
            from src.utils.config import TABLES_DIR
            ensure_dir(TABLES_DIR)
            filepath = TABLES_DIR / save_path
            comparison_df.to_csv(filepath, index=False)
            print(f"Saved comparison table to {filepath}")
        
        return comparison_df


def visualize_training_history(history: Dict[str, List[float]], 
                             model_name: str = "Model",
                             save_fig: bool = True) -> plt.Figure:
    """
    Convenience function to plot training history.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_fig: Whether to save the figure
        
    Returns:
        Matplotlib figure
    """
    visualizer = ModelVisualizer()
    return visualizer.plot_training_history(history, model_name, save_fig)


def visualize_model_comparison(metrics_dict: Dict[str, Dict[str, float]],
                              title: str = "Model Comparison",
                              save_fig: bool = True) -> plt.Figure:
    """
    Convenience function to plot model comparison.
    
    Args:
        metrics_dict: Dictionary of model names to their metrics
        title: Plot title
        save_fig: Whether to save the figure
        
    Returns:
        Matplotlib figure
    """
    visualizer = ModelVisualizer()
    return visualizer.plot_model_comparison(metrics_dict, title, save_fig)
