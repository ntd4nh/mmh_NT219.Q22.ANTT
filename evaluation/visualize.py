"""
Visualization Module for AES Cryptanalysis
=============================================
Plot functions for analyzing training results and attack performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os


def set_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None,
                          save_path=None, title="Training Curves"):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_losses: list — training loss per epoch
        val_losses: list — validation loss per epoch
        train_accs: list — training accuracy per epoch (optional)
        val_accs: list — validation accuracy per epoch (optional)
        save_path: str — path to save plot
        title: str — plot title
    """
    set_style()
    
    num_plots = 1 if train_accs is None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        if val_accs:
            axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved training curves to {save_path}")
    plt.close()


def plot_ge_vs_traces(num_traces_list, ge_list, save_path=None,
                       title="Guessing Entropy vs Number of Traces",
                       labels=None, multiple_runs=None):
    """
    Plot Guessing Entropy as a function of number of traces.
    This is the KEY plot for SCA papers.
    
    Args:
        num_traces_list: list — number of traces
        ge_list: list — GE values (or list of lists for multiple models)
        save_path: str
        title: str
        labels: list — legend labels for multiple models
        multiple_runs: dict — {model_name: (traces, ge)} for comparison
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if multiple_runs:
        colors = plt.cm.Set1(np.linspace(0, 1, len(multiple_runs)))
        for i, (name, (traces, ge)) in enumerate(multiple_runs.items()):
            ax.plot(traces, ge, label=name, linewidth=2, color=colors[i])
    else:
        ax.plot(num_traces_list, ge_list, 'b-', linewidth=2, label='DL Attack')
    
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect recovery (GE=0)')
    ax.set_xlabel('Number of Traces', fontsize=13)
    ax.set_ylabel('Guessing Entropy (log₂)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved GE plot to {save_path}")
    plt.close()


def plot_confusion_matrix(predictions, labels, num_classes=256, 
                           save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix for key byte prediction.
    For 256 classes, shows a heatmap overview.
    """
    set_style()
    
    pred_classes = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(labels, pred_classes):
        cm[true, pred] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Key Byte', fontsize=13)
    ax.set_ylabel('True Key Byte', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved confusion matrix to {save_path}")
    plt.close()


def plot_key_rank_distribution(predictions, true_key_byte, save_path=None,
                                 title="Key Rank Distribution"):
    """
    Plot histogram of key ranks across test samples.
    """
    set_style()
    from evaluation.metrics import key_rank
    
    ranks = [key_rank(predictions[i], true_key_byte) for i in range(len(predictions))]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ranks, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(ranks), color='red', linestyle='--', linewidth=2,
               label=f'Mean Rank = {np.mean(ranks):.1f}')
    ax.set_xlabel('Key Rank', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved rank distribution to {save_path}")
    plt.close()


def plot_accuracy_vs_rounds(results_dict, save_path=None,
                              title="Attack Accuracy vs Number of AES Rounds"):
    """
    Plot accuracy degradation as AES rounds increase.
    
    Args:
        results_dict: dict — {model_name: {num_rounds: accuracy}}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (model_name, round_acc) in enumerate(results_dict.items()):
        rounds = sorted(round_acc.keys())
        accs = [round_acc[r] * 100 for r in rounds]
        ax.plot(rounds, accs, marker=markers[i % len(markers)], 
                label=model_name, linewidth=2, markersize=8, color=colors[i])
    
    ax.axhline(y=100/256, color='grey', linestyle=':', alpha=0.5, label='Random (0.39%)')
    ax.set_xlabel('Number of AES Rounds', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved accuracy vs rounds plot to {save_path}")
    plt.close()


def plot_model_comparison(results, metric_name='accuracy', save_path=None,
                            title="Model Comparison"):
    """
    Bar chart comparing different models.
    
    Args:
        results: dict — {model_name: metric_value}
        metric_name: str — name of the metric
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results.keys())
    values = list(results.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel(metric_name.capitalize(), fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Saved model comparison to {save_path}")
    plt.close()
