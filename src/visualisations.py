"""Visualize model evaluation metrics."""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'


def plot_roc_curve(y_test, y_scores, model_name='Model', roc_auc=None, save_path=None):
    """Plot ROC curve for binary classification model.
    
    Args:
        y_test: True labels
        y_scores: Predicted probabilities for positive class
        model_name: Name of model for legend
        roc_auc: Pre-calculated ROC-AUC score (optional)
        save_path: Path to save figure (optional)
    """
    if roc_auc is None:
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, _ = roc_curve(y_test, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(f'ROC Curve - {model_name}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_roc_curve.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'ROC curve saved: {save_path}')
    plt.close()
    
    return save_path


def plot_confusion_matrix(cm, model_name='Model', save_path=None):
    """Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (2x2 array)
        model_name: Name of model for title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(7, 6))
    
    # Plot heatmap
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, cm[i, j], ha='center', va='center',
                           color='white' if cm[i, j] > cm.max() / 2 else 'black',
                           fontsize=14, fontweight='bold')
    
    plt.colorbar(im)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    plt.xticks([0, 1], ['No Difficulty (0)', 'Difficulty (1)'])
    plt.yticks([0, 1], ['No Difficulty (0)', 'Difficulty (1)'])
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved: {save_path}')
    plt.close()
    
    return save_path


def plot_metrics_table(results_dict, model_name='Model', save_path=None):
    """Plot metrics as a formatted table.
    
    Args:
        results_dict: Dictionary with metric results
        model_name: Name of model for title
        save_path: Path to save figure (optional)
    """
    # Create metrics dataframe
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Class 1)', 'Recall (Class 1)', 'F1 Score (Class 1)', 'ROC-AUC'],
        'Score': [
            f"{results_dict['accuracy']:.3f}",
            f"{results_dict['precision_1']:.3f}",
            f"{results_dict['recall_1']:.3f}",
            f"{results_dict['f1_1']:.3f}",
            f"{results_dict['roc_auc']:.3f}"
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F0F5')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title(f'Model Performance Metrics - {model_name}', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_metrics_table.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Metrics table saved: {save_path}')
    plt.close()
    
    return save_path


def plot_all_visualizations(results_dict, cm, y_test, y_scores, model_name='Model'):
    """Generate all visualizations in one call.
    
    Args:
        results_dict: Dictionary with all metric results
        cm: Confusion matrix
        y_test: True labels
        y_scores: Predicted probabilities
        model_name: Name of model for titles
    """
    print(f'\nGenerating visualizations for {model_name}...')
    plot_roc_curve(y_test, y_scores, model_name, results_dict['roc_auc'])
    plot_confusion_matrix(cm, model_name)
    plot_metrics_table(results_dict, model_name)
    print(f'All visualizations saved to: {OUTPUT_DIR}\n')
