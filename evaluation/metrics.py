"""
Evaluation Metrics for AES Cryptanalysis
==========================================
Standard metrics used in side-channel analysis and DL-based cryptanalysis:
- Guessing Entropy (GE)
- Success Rate (SR@N)
- Key Rank
- Accuracy, F1
"""

import numpy as np
from collections import Counter


def key_rank(scores, true_key_byte):
    """
    Compute the rank of the true key byte among all candidates.
    
    Args:
        scores: np.array (256,) — score/probability for each key byte candidate
        true_key_byte: int — the actual key byte value [0-255]
    
    Returns:
        rank: int — position of true key (0 = best, 255 = worst)
    """
    # Sort candidates by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    rank = np.where(sorted_indices == true_key_byte)[0][0]
    return int(rank)


def guessing_entropy(all_scores, true_key_byte, num_experiments=100):
    """
    Compute Guessing Entropy: average rank of the true key across experiments.
    
    The GE measures how many guesses an attacker needs on average to find
    the correct key. Lower GE = better attack.
    
    Args:
        all_scores: np.array (num_experiments, 256) — accumulated scores
        true_key_byte: int — true key byte value
        num_experiments: int — number of random orderings to average over
    
    Returns:
        ge: float — average guessing entropy
    """
    ranks = []
    for i in range(min(num_experiments, len(all_scores))):
        rank = key_rank(all_scores[i], true_key_byte)
        ranks.append(rank)
    
    return float(np.mean(ranks))


def guessing_entropy_vs_traces(model, traces, plaintexts, true_key_byte,
                                target_byte=0, max_traces=None, step=10,
                                device='cpu'):
    """
    Compute GE as a function of number of traces used.
    This is the key plot in SCA papers.
    
    Args:
        model: trained PyTorch model
        traces: np.array (N, trace_length) — test traces
        plaintexts: np.array (N, 16) — corresponding plaintexts
        true_key_byte: int — true key byte value
        target_byte: int — which byte position is targeted
        max_traces: int — max traces to use (None = use all)
        step: int — compute GE every `step` traces
        device: torch device
    
    Returns:
        num_traces_list: list — number of traces used
        ge_list: list — corresponding GE values
    """
    import torch
    
    model.eval()
    n = len(traces) if max_traces is None else min(max_traces, len(traces))
    
    num_traces_list = []
    ge_list = []
    
    # Accumulated log-probabilities for each key candidate
    log_probs = np.zeros(256, dtype=np.float64)
    
    with torch.no_grad():
        for i in range(n):
            # Get model prediction
            x = torch.tensor(traces[i:i+1], dtype=torch.float32).to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # (256,)
            
            # For each key candidate k, compute P(k | trace, plaintext)
            # Using the Hamming Weight model:
            # The model predicts SBox output class
            # For each key guess k: sbox_out = SBox(pt[target] XOR k)
            from utils.aes_ops import SBOX
            pt_byte = plaintexts[i, target_byte]
            
            for k in range(256):
                sbox_out = SBOX[pt_byte ^ k]
                # Accumulate log-probability
                if probs[sbox_out] > 0:
                    log_probs[k] += np.log(probs[sbox_out] + 1e-36)
            
            # Compute GE at intervals
            if (i + 1) % step == 0 or i == n - 1:
                rank = key_rank(log_probs, true_key_byte)
                num_traces_list.append(i + 1)
                ge_list.append(rank)
    
    return num_traces_list, ge_list


def success_rate(predictions, true_label, top_n_list=[1, 5, 10, 50]):
    """
    Compute Success Rate @ N: fraction of samples where true key
    is in top-N predictions.
    
    Args:
        predictions: np.array (num_samples, 256) — predicted scores/probabilities
        true_label: int or np.array — true key byte value(s)
        top_n_list: list of N values
    
    Returns:
        sr_dict: dict — {N: success_rate_value}
    """
    if isinstance(true_label, int):
        true_label = np.full(len(predictions), true_label)
    
    sr_dict = {}
    for n in top_n_list:
        top_n_preds = np.argsort(predictions, axis=1)[:, -n:]  # top N
        correct = np.array([true_label[i] in top_n_preds[i] for i in range(len(predictions))])
        sr_dict[n] = float(np.mean(correct))
    
    return sr_dict


def traces_to_recovery(ge_values, num_traces_values, threshold_rank=0):
    """
    Compute Traces to Recovery (TTR): minimum number of traces
    needed for GE to reach the threshold rank.
    
    Args:
        ge_values: list — guessing entropy values
        num_traces_values: list — corresponding number of traces
        threshold_rank: int — target rank (0 = first guess correct)
    
    Returns:
        ttr: int — number of traces needed (or -1 if not achieved)
    """
    for i, ge in enumerate(ge_values):
        if ge <= threshold_rank:
            return num_traces_values[i]
    return -1  # Not achieved


def classification_accuracy(predictions, labels):
    """
    Simple classification accuracy.
    
    Args:
        predictions: np.array (N, num_classes) — logits or probabilities
        labels: np.array (N,) — true labels
    
    Returns:
        accuracy: float
    """
    pred_classes = np.argmax(predictions, axis=1)
    return float(np.mean(pred_classes == labels))


def per_class_accuracy(predictions, labels, num_classes=256):
    """
    Per-class accuracy breakdown.
    
    Args:
        predictions: np.array (N, num_classes) — logits/probabilities
        labels: np.array (N,) — true labels
        num_classes: int
    
    Returns:
        class_accuracies: dict — {class: accuracy}
    """
    pred_classes = np.argmax(predictions, axis=1)
    class_acc = {}
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_acc[c] = float(np.mean(pred_classes[mask] == c))
    
    return class_acc


def print_metrics_summary(predictions, labels, true_key_byte=None, experiment_name=""):
    """Print a formatted summary of all metrics."""
    print(f"\n{'='*60}")
    print(f"📊 Metrics Summary: {experiment_name}")
    print(f"{'='*60}")
    
    # Classification accuracy
    acc = classification_accuracy(predictions, labels)
    print(f"  Accuracy:     {acc*100:.2f}%")
    
    # Top-N accuracy
    for n in [1, 5, 10, 50]:
        top_n = np.argsort(predictions, axis=1)[:, -n:]
        correct = np.mean([labels[i] in top_n[i] for i in range(len(labels))])
        print(f"  Top-{n:2d} Acc:   {correct*100:.2f}%")
    
    # If we know the true key byte, compute attack-specific metrics
    if true_key_byte is not None:
        # Average key rank
        ranks = [key_rank(predictions[i], true_key_byte) for i in range(len(predictions))]
        avg_rank = np.mean(ranks)
        print(f"\n  Avg Key Rank: {avg_rank:.1f} / 255")
        print(f"  Median Rank:  {np.median(ranks):.1f}")
        print(f"  Best Rank:    {min(ranks)}")
        print(f"  Worst Rank:   {max(ranks)}")
    
    # Random baseline
    print(f"\n  Random baseline accuracy: {(1/256)*100:.2f}%")
    print(f"{'='*60}\n")
