"""
Synthetic Dataset Generator
============================
Generates datasets for different AES cryptanalysis attack modes:
1. Ciphertext-only: fixed key, random plaintexts → ciphertext, label = key byte
2. Known-plaintext: random (plaintext, ciphertext) pairs, label = key byte
3. Chosen-plaintext: structured plaintext selection for differential analysis
4. Simulated SCA traces: Hamming weight power model + Gaussian noise
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.aes_ops import (
    aes_encrypt, SBOX, hamming_weight_batch, get_sbox_output
)


class CiphertextOnlyDataset:
    """
    Ciphertext-Only Attack Dataset
    ================================
    Scenario: Attacker only observes ciphertext. 
    Goal: Predict the target key byte from ciphertext alone.
    
    For toy-AES (reduced rounds), the relationship between
    ciphertext and key is simpler and DL can learn patterns.
    """
    
    def __init__(self, num_samples=100000, num_rounds=2, target_byte=0, seed=42):
        """
        Args:
            num_samples: Number of samples to generate
            num_rounds: Number of AES rounds (1-10, use 1-3 for toy)
            target_byte: Which key byte to predict (0-15)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_rounds = num_rounds
        self.target_byte = target_byte
        self.seed = seed
        
    def generate(self):
        """
        Generate dataset with a FIXED random key.
        
        Returns:
            ciphertexts: np.array (N, 16) — input features
            labels: np.array (N,) — target key byte value [0-255]
            key: np.array (16,) — the secret key used
            plaintexts: np.array (N, 16) — for debugging/analysis
        """
        rng = np.random.RandomState(self.seed)
        
        # Generate a fixed random key
        key = rng.randint(0, 256, size=16).astype(np.uint8)
        
        # Generate random plaintexts
        plaintexts = rng.randint(0, 256, size=(self.num_samples, 16)).astype(np.uint8)
        
        # Encrypt each plaintext
        ciphertexts = np.zeros((self.num_samples, 16), dtype=np.uint8)
        for i in range(self.num_samples):
            ciphertexts[i] = aes_encrypt(plaintexts[i], key, self.num_rounds)
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i+1}/{self.num_samples} samples...")
        
        # Label = the target key byte
        labels = np.full(self.num_samples, key[self.target_byte], dtype=np.uint8)
        
        print(f"✅ Ciphertext-only dataset generated:")
        print(f"   Samples: {self.num_samples}, Rounds: {self.num_rounds}")
        print(f"   Target key byte [{self.target_byte}] = {key[self.target_byte]}")
        
        return ciphertexts, labels, key, plaintexts


class KnownPlaintextDataset:
    """
    Known-Plaintext Attack Dataset
    ================================
    Scenario: Attacker knows both plaintext and ciphertext pairs.
    Goal: Predict the target key byte from (plaintext, ciphertext) pairs.
    
    Input: concatenation of plaintext + ciphertext (32 bytes)
    """
    
    def __init__(self, num_samples=100000, num_rounds=4, target_byte=0, 
                 num_keys=256, seed=42):
        """
        Args:
            num_samples: Total samples across all keys
            num_rounds: Number of AES rounds
            target_byte: Which key byte to predict (0-15)
            num_keys: Number of different keys to use (up to 256)
            seed: Random seed
        """
        self.num_samples = num_samples
        self.num_rounds = num_rounds
        self.target_byte = target_byte
        self.num_keys = min(num_keys, 256)
        self.seed = seed
    
    def generate(self):
        """
        Generate dataset with MULTIPLE random keys.
        Each key's target byte covers all 256 possible values.
        
        Returns:
            inputs: np.array (N, 32) — concatenated (plaintext, ciphertext)
            labels: np.array (N,) — target key byte value [0-255]
            keys: np.array (K, 16) — all keys used
        """
        rng = np.random.RandomState(self.seed)
        
        samples_per_key = self.num_samples // self.num_keys
        total = samples_per_key * self.num_keys
        
        inputs = np.zeros((total, 32), dtype=np.uint8)
        labels = np.zeros(total, dtype=np.uint8)
        keys = np.zeros((self.num_keys, 16), dtype=np.uint8)
        
        idx = 0
        for k in range(self.num_keys):
            # Generate key with specific target byte value
            key = rng.randint(0, 256, size=16).astype(np.uint8)
            # Ensure target byte covers the class k % 256
            key[self.target_byte] = k % 256
            keys[k] = key
            
            # Generate plaintexts for this key
            plaintexts = rng.randint(0, 256, size=(samples_per_key, 16)).astype(np.uint8)
            
            for i in range(samples_per_key):
                ct = aes_encrypt(plaintexts[i], key, self.num_rounds)
                inputs[idx] = np.concatenate([plaintexts[i], ct])
                labels[idx] = key[self.target_byte]
                idx += 1
            
            if (k + 1) % 50 == 0:
                print(f"  Generated keys {k+1}/{self.num_keys}...")
        
        # Shuffle
        perm = rng.permutation(total)
        inputs = inputs[perm]
        labels = labels[perm]
        
        print(f"✅ Known-plaintext dataset generated:")
        print(f"   Samples: {total}, Rounds: {self.num_rounds}")
        print(f"   Keys: {self.num_keys}, Samples/key: {samples_per_key}")
        print(f"   Label distribution: {len(np.unique(labels))} unique classes")
        
        return inputs, labels, keys


class ChosenPlaintextDataset:
    """
    Chosen-Plaintext Attack Dataset
    ================================
    Scenario: Attacker can choose specific plaintexts to encrypt.
    Strategy: Use differential-style analysis — encrypt pairs of plaintexts
    with known input differences and observe ciphertext differences.
    
    The DL model learns from (plaintext_pair, ciphertext_pair) → key byte.
    """
    
    def __init__(self, num_samples=100000, num_rounds=3, target_byte=0,
                 num_keys=256, seed=42):
        self.num_samples = num_samples
        self.num_rounds = num_rounds
        self.target_byte = target_byte
        self.num_keys = min(num_keys, 256)
        self.seed = seed
    
    def generate(self):
        """
        Generate differential pairs.
        
        Returns:
            inputs: np.array (N, 48) — (plaintext1, ciphertext1, ciphertext_diff)
            labels: np.array (N,) — target key byte
            keys: np.array (K, 16) — all keys used
        """
        rng = np.random.RandomState(self.seed)
        
        samples_per_key = self.num_samples // self.num_keys
        total = samples_per_key * self.num_keys
        
        # Input: plaintext1 (16) + ciphertext1 (16) + ciphertext_diff (16) = 48
        inputs = np.zeros((total, 48), dtype=np.uint8)
        labels = np.zeros(total, dtype=np.uint8)
        keys = np.zeros((self.num_keys, 16), dtype=np.uint8)
        
        idx = 0
        for k in range(self.num_keys):
            key = rng.randint(0, 256, size=16).astype(np.uint8)
            key[self.target_byte] = k % 256
            keys[k] = key
            
            for i in range(samples_per_key):
                # Choose plaintext1 randomly
                pt1 = rng.randint(0, 256, size=16).astype(np.uint8)
                
                # Create pt2 by flipping target byte (chosen difference)
                pt2 = pt1.copy()
                pt2[self.target_byte] ^= rng.randint(1, 256)  # Non-zero difference
                
                ct1 = aes_encrypt(pt1, key, self.num_rounds)
                ct2 = aes_encrypt(pt2, key, self.num_rounds)
                
                # Ciphertext difference
                ct_diff = (ct1 ^ ct2).astype(np.uint8)
                
                inputs[idx] = np.concatenate([pt1, ct1, ct_diff])
                labels[idx] = key[self.target_byte]
                idx += 1
            
            if (k + 1) % 50 == 0:
                print(f"  Generated keys {k+1}/{self.num_keys}...")
        
        # Shuffle
        perm = rng.permutation(total)
        inputs = inputs[perm]
        labels = labels[perm]
        
        print(f"✅ Chosen-plaintext dataset generated:")
        print(f"   Samples: {total}, Rounds: {self.num_rounds}")
        
        return inputs, labels, keys


class SimulatedSCADataset:
    """
    Simulated Side-Channel Analysis Dataset
    ==========================================
    Simulates power consumption traces using Hamming Weight model.
    
    Power model: P(t) = HW(SBox(pt[i] ⊕ k[i])) + noise
    
    Each trace represents simulated power measurements during
    AES SubBytes operation.
    """
    
    def __init__(self, num_traces=50000, trace_length=700, snr=5.0,
                 target_byte=0, num_keys=1, seed=42):
        """
        Args:
            num_traces: Number of power traces
            trace_length: Length of each trace (time points)
            snr: Signal-to-noise ratio
            target_byte: Which key byte is leaking
            num_keys: Number of different keys (1 for single-device profiling)
            seed: Random seed
        """
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.snr = snr
        self.target_byte = target_byte
        self.num_keys = num_keys
        self.seed = seed
    
    def generate(self):
        """
        Generate simulated power traces.
        
        Returns:
            traces: np.array (N, trace_length) — simulated power traces (float32)
            labels: np.array (N,) — S-Box output value [0-255] (for profiling attack)
            plaintexts: np.array (N, 16)
            key: np.array (16,)
        """
        rng = np.random.RandomState(self.seed)
        
        # Fixed key for profiling
        key = rng.randint(0, 256, size=16).astype(np.uint8)
        
        # Random plaintexts
        plaintexts = rng.randint(0, 256, size=(self.num_traces, 16)).astype(np.uint8)
        
        # Compute S-Box outputs (intermediate value = target)
        sbox_outputs = np.array([
            get_sbox_output(plaintexts[i, self.target_byte], key[self.target_byte])
            for i in range(self.num_traces)
        ], dtype=np.uint8)
        
        # Compute Hamming Weights of S-Box outputs
        hw_values = hamming_weight_batch(sbox_outputs).astype(np.float32)
        
        # Generate traces
        noise_std = np.sqrt(np.var(hw_values) / self.snr)
        traces = np.zeros((self.num_traces, self.trace_length), dtype=np.float32)
        
        # Create a template trace shape (simulated AES operation pattern)
        template = np.zeros(self.trace_length, dtype=np.float32)
        
        # Simulated peaks at specific time points (each byte processed sequentially)
        for byte_idx in range(16):
            center = 50 + byte_idx * 35  # Each byte ~35 time points apart
            if center < self.trace_length:
                # Gaussian-shaped peak at processing point
                t = np.arange(self.trace_length, dtype=np.float32)
                template += np.exp(-0.5 * ((t - center) / 5) ** 2)
        
        for i in range(self.num_traces):
            # Base trace with template shape
            traces[i] = template.copy()
            
            # Add signal at target byte's processing point
            target_center = 50 + self.target_byte * 35
            if target_center < self.trace_length:
                t = np.arange(self.trace_length, dtype=np.float32)
                signal = hw_values[i] * np.exp(-0.5 * ((t - target_center) / 5) ** 2)
                traces[i] += signal
            
            # Add Gaussian noise
            traces[i] += rng.normal(0, noise_std, self.trace_length).astype(np.float32)
        
        # Labels = S-Box output (256 classes for profiling attack)
        labels = sbox_outputs
        
        print(f"✅ Simulated SCA dataset generated:")
        print(f"   Traces: {self.num_traces}, Length: {self.trace_length}")
        print(f"   SNR: {self.snr}, Target byte: {self.target_byte}")
        print(f"   Key byte [{self.target_byte}] = {key[self.target_byte]}")
        
        return traces, labels, plaintexts, key


def save_dataset(data, labels, filepath, **extra):
    """Save dataset to .npz file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, data=data, labels=labels, **extra)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 Saved to {filepath} ({size_mb:.1f} MB)")


def load_dataset(filepath):
    """Load dataset from .npz file."""
    loaded = np.load(filepath)
    return dict(loaded)


# ============================================================
# Main: Generate all datasets
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AES Cryptanalysis Datasets")
    parser.add_argument("--mode", choices=["ciphertext", "known", "chosen", "sca", "all"],
                        default="all", help="Which dataset to generate")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--rounds", type=int, default=2, help="Number of AES rounds")
    parser.add_argument("--output-dir", type=str, default="./data/generated",
                        help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["ciphertext", "all"]:
        print("\n" + "="*60)
        print("Generating Ciphertext-Only Dataset...")
        print("="*60)
        gen = CiphertextOnlyDataset(
            num_samples=args.samples,
            num_rounds=args.rounds,
            target_byte=0
        )
        ct, labels, key, pt = gen.generate()
        save_dataset(ct, labels, 
                     os.path.join(args.output_dir, "ciphertext_only.npz"),
                     key=key, plaintexts=pt)
    
    if args.mode in ["known", "all"]:
        print("\n" + "="*60)
        print("Generating Known-Plaintext Dataset...")
        print("="*60)
        gen = KnownPlaintextDataset(
            num_samples=args.samples,
            num_rounds=args.rounds,
            target_byte=0,
            num_keys=256
        )
        inputs, labels, keys = gen.generate()
        save_dataset(inputs, labels,
                     os.path.join(args.output_dir, "known_plaintext.npz"),
                     keys=keys)
    
    if args.mode in ["chosen", "all"]:
        print("\n" + "="*60)
        print("Generating Chosen-Plaintext Dataset...")
        print("="*60)
        gen = ChosenPlaintextDataset(
            num_samples=args.samples,
            num_rounds=args.rounds,
            target_byte=0,
            num_keys=256
        )
        inputs, labels, keys = gen.generate()
        save_dataset(inputs, labels,
                     os.path.join(args.output_dir, "chosen_plaintext.npz"),
                     keys=keys)
    
    if args.mode in ["sca", "all"]:
        print("\n" + "="*60)
        print("Generating Simulated SCA Dataset...")
        print("="*60)
        gen = SimulatedSCADataset(
            num_traces=args.samples,
            trace_length=700,
            snr=5.0,
            target_byte=0
        )
        traces, labels, pt, key = gen.generate()
        save_dataset(traces, labels,
                     os.path.join(args.output_dir, "simulated_sca.npz"),
                     key=key, plaintexts=pt)
    
    print("\n✅ All datasets generated successfully!")
