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


class TraceAugmentor:
    """
    Composable data augmentation pipeline for SCA traces.
    
    Combines multiple augmentation techniques with configurable
    probabilities. Each augmentation is applied independently
    with its own probability per sample.
    
    Usage:
        augmentor = TraceAugmentor(noise_std=0.1, max_shift=3)
        augmented_batch = augmentor(batch_tensor)  # works with PyTorch tensors
    """
    
    def __init__(self, noise_std=0.05, max_shift=3, scale_range=(0.95, 1.05),
                 noise_prob=0.5, shift_prob=0.3, scale_prob=0.3, seed=None):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
            max_shift: Maximum random circular shift (samples)
            scale_range: (min, max) amplitude scaling range
            noise_prob: Probability of applying noise to each sample
            shift_prob: Probability of applying shift to each sample
            scale_prob: Probability of applying scaling to each sample
            seed: Random seed (None = non-deterministic)
        """
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, data):
        """
        Apply augmentation pipeline to a batch.
        
        Args:
            data: np.array (N, D) or torch.Tensor (N, D)
        
        Returns:
            Augmented data (same type as input)
        """
        import torch
        is_tensor = isinstance(data, torch.Tensor)
        
        if is_tensor:
            device = data.device
            arr = data.cpu().numpy().copy()
        else:
            arr = data.copy()
        
        n = len(arr)
        
        # Apply noise
        noise_mask = self.rng.random(n) < self.noise_prob
        if noise_mask.any():
            noise = self.rng.normal(0, self.noise_std, arr[noise_mask].shape)
            arr[noise_mask] += noise.astype(arr.dtype)
        
        # Apply random shift
        shift_mask = self.rng.random(n) < self.shift_prob
        if shift_mask.any():
            for i in np.where(shift_mask)[0]:
                shift = self.rng.randint(-self.max_shift, self.max_shift + 1)
                arr[i] = np.roll(arr[i], shift)
        
        # Apply amplitude scaling
        scale_mask = self.rng.random(n) < self.scale_prob
        if scale_mask.any():
            scales = self.rng.uniform(
                self.scale_range[0], self.scale_range[1], 
                size=(scale_mask.sum(), 1)
            )
            arr[scale_mask] *= scales.astype(arr.dtype)
        
        if is_tensor:
            return torch.tensor(arr, dtype=data.dtype, device=device)
        return arr
    
    def __repr__(self):
        return (f"TraceAugmentor(noise_std={self.noise_std}, max_shift={self.max_shift}, "
                f"scale_range={self.scale_range}, "
                f"noise_prob={self.noise_prob}, shift_prob={self.shift_prob}, "
                f"scale_prob={self.scale_prob})")
