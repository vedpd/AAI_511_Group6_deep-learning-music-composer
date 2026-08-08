"""
Evaluation modules for the Composer Classification project.
"""

from .metrics import ModelEvaluator, evaluate_model
from .visualize import ModelVisualizer, visualize_training_history, visualize_model_comparison

__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'ModelVisualizer',
    'visualize_training_history',
    'visualize_model_comparison'
]
