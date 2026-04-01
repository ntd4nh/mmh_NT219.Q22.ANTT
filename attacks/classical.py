"""
Classical Side-Channel Attacks
================================
Implements classical (non-DL) attack methods for comparison:
- CPA (Correlation Power Analysis)
- DPA (Differential Power Analysis)

Used as baselines to compare against DL-based attacks.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.aes_ops import SBOX, hamming_weight_batch


class CPA:
    """
    Correlation Power Analysis (CPA) Attack.
    
    Classical profiling attack that correlates power traces with
    hypothetical intermediate values for each key candidate.
    
    The Hamming Weight (HW) model is used:
        hypothesis[k] = HW(SBox(plaintext_byte XOR k))
    
    For each key guess k, compute Pearson correlation between
    HW predictions and actual trace values at each time point.
    The correct key produces the highest correlation.
    
    Reference: Brier et al., "Correlation Power Analysis with a
    Leakage Model", CHES 2004.
    """
    
    def __init__(self, target_byte=0):
        """
        Args:
            target_byte: Which key byte to attack (0-15)
        """
        self.target_byte = target_byte
        self.correlations = None  # (256, trace_length)
        self.key_scores = None   # (256,)
    
    def attack(self, traces, plaintexts, num_traces=None):
        """
        Run CPA attack on a set of traces.
        
        Args:
            traces: np.array (N, trace_length) — power traces
            plaintexts: np.array (N, 16) — corresponding plaintexts
            num_traces: int — use only first N traces (None = all)
        
        Returns:
            key_ranking: np.array (256,) — key candidates sorted by score (best first)
            correlations: np.array (256, trace_length) — correlation for each key guess
        """
        if num_traces is not None:
            traces = traces[:num_traces]
            plaintexts = plaintexts[:num_traces]
        
        n, trace_len = traces.shape
        
        # For each key candidate, compute hypothetical intermediate values
        correlations = np.zeros((256, trace_len), dtype=np.float64)
        
        for k in range(256):
            # SBox output for this key guess
            sbox_outputs = np.array([
                SBOX[int(plaintexts[i, self.target_byte]) ^ k]
                for i in range(n)
            ], dtype=np.uint8)
            
            # Hamming weight of SBox outputs
            hw = hamming_weight_batch(sbox_outputs).astype(np.float64)
            
            # Pearson correlation between HW model and each trace point
            hw_centered = hw - hw.mean()
            hw_std = hw.std()
            
            if hw_std < 1e-10:
                continue
            
            for t in range(trace_len):
                trace_col = traces[:, t].astype(np.float64)
                trace_centered = trace_col - trace_col.mean()
                trace_std = trace_col.std()
                
                if trace_std < 1e-10:
                    continue
                
                corr = np.dot(hw_centered, trace_centered) / (n * hw_std * trace_std)
                correlations[k, t] = corr
        
        self.correlations = correlations
        
        # Key score = max absolute correlation across all time points
        self.key_scores = np.max(np.abs(correlations), axis=1)  # (256,)
        
        # Return ranking (best first)
        key_ranking = np.argsort(self.key_scores)[::-1]
        
        return key_ranking, correlations
    
    def attack_incremental(self, traces, plaintexts, step=50, max_traces=None):
        """
        Run CPA with increasing number of traces to compute GE curve.
        
        Args:
            traces: np.array (N, trace_length)
            plaintexts: np.array (N, 16)
            step: Compute rank every `step` traces
            max_traces: Maximum traces to use
        
        Returns:
            num_traces_list: list — number of traces used
            ranks: list — rank of true key at each step
        """
        n = len(traces) if max_traces is None else min(max_traces, len(traces))
        
        num_traces_list = []
        ranks = []
        
        for num_t in range(step, n + 1, step):
            key_ranking, _ = self.attack(traces, plaintexts, num_traces=num_t)
            num_traces_list.append(num_t)
            # We don't know the true key here, caller must compute rank
            ranks.append(key_ranking)
        
        return num_traces_list, ranks
    
    def get_key_rank(self, true_key_byte):
        """
        Get rank of the true key byte after running attack.
        
        Args:
            true_key_byte: int — the actual key byte value
        
        Returns:
            rank: int — position of true key (0 = best)
        """
        if self.key_scores is None:
            raise ValueError("Run attack() first")
        
        sorted_keys = np.argsort(self.key_scores)[::-1]
        rank = np.where(sorted_keys == true_key_byte)[0][0]
        return int(rank)
    
    def ge_vs_traces(self, traces, plaintexts, true_key_byte, 
                      step=50, max_traces=None):
        """
        Compute Guessing Entropy vs number of traces for CPA.
        
        Args:
            traces: np.array (N, trace_length)
            plaintexts: np.array (N, 16)
            true_key_byte: int — true key byte value
            step: Compute GE every `step` traces
            max_traces: Max traces to use
        
        Returns:
            num_traces_list: list
            ge_list: list — key rank at each step
        """
        n = len(traces) if max_traces is None else min(max_traces, len(traces))
        
        num_traces_list = []
        ge_list = []
        
        for num_t in range(step, n + 1, step):
            self.attack(traces, plaintexts, num_traces=num_t)
            rank = self.get_key_rank(true_key_byte)
            num_traces_list.append(num_t)
            ge_list.append(rank)
        
        return num_traces_list, ge_list


class DPA:
    """
    Differential Power Analysis (DPA) Attack.
    
    Partitions traces into two groups based on a selection function
    (typically one bit of an intermediate value), then computes
    the difference of means to find points of interest.
    
    Reference: Kocher et al., "Differential Power Analysis", CRYPTO 1999.
    """
    
    def __init__(self, target_byte=0, target_bit=0):
        """
        Args:
            target_byte: Which key byte to attack (0-15)
            target_bit: Which bit of SBox output to partition on (0-7)
        """
        self.target_byte = target_byte
        self.target_bit = target_bit
    
    def attack(self, traces, plaintexts, num_traces=None):
        """
        Run DPA attack.
        
        Args:
            traces: np.array (N, trace_length)
            plaintexts: np.array (N, 16)
            num_traces: int — use only first N traces
        
        Returns:
            key_ranking: np.array (256,) — best key guesses first
            diff_traces: np.array (256, trace_length) — difference of means
        """
        if num_traces is not None:
            traces = traces[:num_traces]
            plaintexts = plaintexts[:num_traces]
        
        n, trace_len = traces.shape
        diff_traces = np.zeros((256, trace_len), dtype=np.float64)
        
        for k in range(256):
            # Compute selection function: bit `target_bit` of SBox(pt XOR k)
            sbox_outputs = np.array([
                SBOX[int(plaintexts[i, self.target_byte]) ^ k]
                for i in range(n)
            ], dtype=np.uint8)
            
            selection = (sbox_outputs >> self.target_bit) & 1
            
            group0 = traces[selection == 0]
            group1 = traces[selection == 1]
            
            if len(group0) > 0 and len(group1) > 0:
                diff_traces[k] = group1.mean(axis=0) - group0.mean(axis=0)
        
        # Key score = max absolute difference
        key_scores = np.max(np.abs(diff_traces), axis=1)
        key_ranking = np.argsort(key_scores)[::-1]
        
        self.key_scores = key_scores
        self.diff_traces = diff_traces
        
        return key_ranking, diff_traces
    
    def get_key_rank(self, true_key_byte):
        """Get rank of the true key byte after running attack."""
        if self.key_scores is None:
            raise ValueError("Run attack() first")
        sorted_keys = np.argsort(self.key_scores)[::-1]
        rank = np.where(sorted_keys == true_key_byte)[0][0]
        return int(rank)


if __name__ == "__main__":
    """Quick test of CPA on simulated traces."""
    from data.synthetic.generator import SimulatedSCADataset
    
    print("Generating test traces...")
    dataset = SimulatedSCADataset(
        num_traces=5000, trace_length=700, snr=5.0, target_byte=0, seed=42
    )
    traces, labels, plaintexts, key = dataset.generate()
    true_key_byte = int(key[0])
    
    # Test CPA
    print(f"\nRunning CPA attack (true key byte = {true_key_byte})...")
    cpa = CPA(target_byte=0)
    ranking, corr = cpa.attack(traces, plaintexts)
    rank = cpa.get_key_rank(true_key_byte)
    print(f"CPA key rank: {rank} (0 = perfect)")
    print(f"Top-5 key guesses: {ranking[:5].tolist()}")
    
    # Test DPA
    print(f"\nRunning DPA attack...")
    dpa = DPA(target_byte=0, target_bit=0)
    ranking, diff = dpa.attack(traces, plaintexts)
    rank = dpa.get_key_rank(true_key_byte)
    print(f"DPA key rank: {rank} (0 = perfect)")
    print(f"Top-5 key guesses: {ranking[:5].tolist()}")
    
    print("\n✅ Classical attacks test passed!")
