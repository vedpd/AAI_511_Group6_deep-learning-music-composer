"""
Evaluation metrics for composer classification models.
Implements comprehensive evaluation metrics and analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import COMPOSERS, FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT
from src.utils.helpers import ensure_dir


class ModelEvaluator:
    """
    Comprehensive model evaluation for composer classification.
    """
    
    def __init__(self, class_names: List[str] = COMPOSERS):
        """
        Initialize model evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'macro_f1': macro_f1,
            'precision_per_class': dict(zip(self.class_names, precision_per_class)),
            'recall_per_class': dict(zip(self.class_names, recall_per_class)),
            'f1_per_class': dict(zip(self.class_names, f1_per_class))
        }
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                # For multi-class, use one-vs-rest approach
                y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, label] = 1
                
                roc_auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc'] = roc_auc
            except:
                pass
        
        self.metrics = metrics
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model", normalize: bool = False,
                            save_fig: bool = True) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for title
            normalize: Whether to normalize the confusion matrix
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            filepath = FIGURES_DIR / f"{model_name}_confusion_matrix.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved confusion matrix to {filepath}")
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model", save_fig: bool = True) -> plt.Figure:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for title
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Binarize labels
        y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            ensure_dir(FIGURES_DIR)
            filepath = FIGURES_DIR / f"{model_name}_roc_curves.{FIGURE_FORMAT}"
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved ROC curves to {filepath}")
        
        return fig
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        return report
    
    def print_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                     y_pred_proba: np.ndarray = None) -> None:
        """
        Print all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_score']:.4f}")
        print(f"F1 Score (macro):    {metrics['macro_f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 50)
        for class_name in self.class_names:
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision_per_class'][class_name]:.4f}")
            print(f"  Recall:    {metrics['recall_per_class'][class_name]:.4f}")
            print(f"  F1 Score:  {metrics['f1_per_class'][class_name]:.4f}")
        
        print("\nClassification Report:")
        print("-" * 50)
        print(self.get_classification_report(y_true, y_pred))
        print("="*50)
    
    def compare_models(self, metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary of model names to their metrics
            
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in metrics_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1_score', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_pred_proba: np.ndarray = None,
                  class_names: List[str] = COMPOSERS,
                  model_name: str = "Model",
                  save_plots: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: List of class names
        model_name: Name of the model
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = ModelEvaluator(class_names)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print metrics
    evaluator.print_metrics(y_true, y_pred, y_pred_proba)
    
    # Plot confusion matrix
    if save_plots:
        evaluator.plot_confusion_matrix(y_true, y_pred, model_name)
        
        # Plot ROC curves if probabilities provided
        if y_pred_proba is not None:
            evaluator.plot_roc_curves(y_true, y_pred_proba, model_name)
    
    return metrics
