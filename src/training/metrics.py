"""
Evaluation Metrics
===================
Confusion matrix, per-class accuracy, and classification reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, num_classes=43, model_name='Model',
                          save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.title(f'{model_name} — Confusion Matrix (Test Set)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return cm


def print_worst_classes(y_true, y_pred, n=5):
    """Print the n worst-performing classes by accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    worst = np.argsort(class_acc)[:n]

    print(f"\nTop {n} worst-performing classes:")
    for c in worst:
        print(f"  Class {c}: {class_acc[c]:.1f}% accuracy "
              f"({cm.sum(axis=1)[c]} test samples)")

    return class_acc


def print_classification_report(y_true, y_pred, class_names=None):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))
