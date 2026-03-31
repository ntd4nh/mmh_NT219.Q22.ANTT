"""
Preprocessing Utilities
========================
Data preprocessing functions for traces and ciphertext:
- Normalization (z-score, min-max)
- Signal processing (filtering, alignment)
- Data augmentation (noise injection, jitter)
"""

import numpy as np


def z_score_normalize(data, mean=None, std=None):
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        data: np.array (N, D)
        mean: precomputed mean (for test data)
        std: precomputed std
    
    Returns:
        normalized data, mean, std
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    return (data - mean) / std, mean, std


def min_max_normalize(data, min_val=None, max_val=None):
    """Min-max normalization to [0, 1]."""
    if min_val is None:
        min_val = data.min(axis=0)
    if max_val is None:
        max_val = data.max(axis=0)
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    return (data - min_val) / range_val, min_val, max_val


def add_gaussian_noise(data, noise_std=0.1, seed=None):
    """Add Gaussian noise for data augmentation."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, data.shape).astype(data.dtype)
    return data + noise


def random_shift(traces, max_shift=5, seed=None):
    """
    Random circular shift for trace augmentation.
    Simulates clock jitter / desynchronization.
    """
    rng = np.random.RandomState(seed)
    shifted = np.zeros_like(traces)
    for i in range(len(traces)):
        shift = rng.randint(-max_shift, max_shift + 1)
        shifted[i] = np.roll(traces[i], shift)
    return shifted


def amplitude_scaling(traces, scale_range=(0.9, 1.1), seed=None):
    """Random amplitude scaling for augmentation."""
    rng = np.random.RandomState(seed)
    scales = rng.uniform(scale_range[0], scale_range[1], size=(len(traces), 1))
    return traces * scales


def crop_traces(traces, start, end):
    """Crop traces to a specific window (Point of Interest)."""
    return traces[:, start:end]


def bytes_to_float(data):
    """Convert byte array [0-255] to float [0, 1]."""
    return data.astype(np.float32) / 255.0


def float_to_bytes(data):
    """Convert float [0, 1] back to bytes [0-255]."""
    return (data * 255).clip(0, 255).astype(np.uint8)
