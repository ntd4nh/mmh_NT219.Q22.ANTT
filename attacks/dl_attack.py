"""
DL-Based Attack Wrapper
=========================
Provides a unified interface for DL-based key recovery attacks.

Wraps the log-likelihood accumulation approach used in profiling SCA:
1. Train model to predict intermediate values (S-Box output)
2. For each trace, get model predictions
3. Accumulate log-probabilities for each key candidate
4. Rank key candidates by accumulated score
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.aes_ops import SBOX


class DLAttack:
    """
    Deep Learning-based SCA Attack.
    
    Uses a trained DL model to perform key recovery via
    log-likelihood accumulation over multiple traces.
    
    The model predicts P(sbox_output | trace) for each trace.
    For each key guess k:
        score[k] += log P(SBox(pt[target] XOR k) | trace)
    
    The key with the highest accumulated score is the best guess.
    """
    
    def __init__(self, model, device='cpu', target_byte=0):
        """
        Args:
            model: Trained PyTorch model (predicts 256 SBox output classes)
            device: torch device
            target_byte: Which key byte is targeted
        """
        self.model = model
        self.device = device
        self.target_byte = target_byte
        self.log_probs = None
    
    def attack(self, traces, plaintexts, num_traces=None, batch_size=256):
        """
        Run DL attack via log-likelihood accumulation.
        
        Args:
            traces: np.array (N, trace_length) — preprocessed traces
            plaintexts: np.array (N, 16) — corresponding plaintexts
            num_traces: int — use only first N traces
            batch_size: int — inference batch size
        
        Returns:
            key_ranking: np.array (256,) — key candidates (best first)
            log_probs: np.array (256,) — accumulated log-probabilities
        """
        if num_traces is not None:
            traces = traces[:num_traces]
            plaintexts = plaintexts[:num_traces]
        
        n = len(traces)
        log_probs = np.zeros(256, dtype=np.float64)
        
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.tensor(
                    traces[start:end], dtype=torch.float32
                ).to(self.device)
                
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                for i in range(end - start):
                    pt_byte = int(plaintexts[start + i, self.target_byte])
                    for k in range(256):
                        sbox_out = SBOX[pt_byte ^ k]
                        if probs[i, sbox_out] > 0:
                            log_probs[k] += np.log(probs[i, sbox_out] + 1e-36)
        
        self.log_probs = log_probs
        key_ranking = np.argsort(log_probs)[::-1]
        
        return key_ranking, log_probs
    
    def get_key_rank(self, true_key_byte):
        """Get rank of the true key byte after attack."""
        if self.log_probs is None:
            raise ValueError("Run attack() first")
        sorted_keys = np.argsort(self.log_probs)[::-1]
        rank = np.where(sorted_keys == true_key_byte)[0][0]
        return int(rank)
    
    def ge_vs_traces(self, traces, plaintexts, true_key_byte,
                      step=50, max_traces=None, batch_size=256):
        """
        Compute GE vs number of traces for DL attack.
        
        Incrementally accumulates log-probabilities and computes
        key rank at regular intervals.
        
        Args:
            traces: np.array (N, trace_length)
            plaintexts: np.array (N, 16)
            true_key_byte: int
            step: Compute rank every `step` traces
            max_traces: Max traces to use
            batch_size: Inference batch size
        
        Returns:
            num_traces_list: list
            ge_list: list — key rank at each step
        """
        n = len(traces) if max_traces is None else min(max_traces, len(traces))
        
        num_traces_list = []
        ge_list = []
        log_probs = np.zeros(256, dtype=np.float64)
        
        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                # Single trace inference
                x = torch.tensor(
                    traces[i:i+1], dtype=torch.float32
                ).to(self.device)
                output = self.model(x)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                pt_byte = int(plaintexts[i, self.target_byte])
                for k in range(256):
                    sbox_out = SBOX[pt_byte ^ k]
                    if probs[sbox_out] > 0:
                        log_probs[k] += np.log(probs[sbox_out] + 1e-36)
                
                if (i + 1) % step == 0:
                    sorted_keys = np.argsort(log_probs)[::-1]
                    rank = np.where(sorted_keys == true_key_byte)[0][0]
                    num_traces_list.append(i + 1)
                    ge_list.append(int(rank))
        
        self.log_probs = log_probs
        return num_traces_list, ge_list


def compare_attacks(dl_model, traces, plaintexts, true_key_byte,
                     target_byte=0, device='cpu', max_traces=2000, step=50):
    """
    Compare DL attack vs CPA attack on the same data.
    
    Args:
        dl_model: Trained PyTorch model
        traces: np.array (N, trace_length)
        plaintexts: np.array (N, 16)
        true_key_byte: int
        target_byte: int
        device: torch device
        max_traces: Max traces for comparison
        step: GE computation step
    
    Returns:
        results: dict with 'DL' and 'CPA' GE curves
    """
    from attacks.classical import CPA
    
    print("  Running DL attack...")
    dl = DLAttack(dl_model, device=device, target_byte=target_byte)
    dl_traces, dl_ge = dl.ge_vs_traces(
        traces, plaintexts, true_key_byte,
        step=step, max_traces=max_traces
    )
    
    print("  Running CPA attack...")
    cpa = CPA(target_byte=target_byte)
    cpa_traces, cpa_ge = cpa.ge_vs_traces(
        traces, plaintexts, true_key_byte,
        step=step, max_traces=max_traces
    )
    
    results = {
        'DL Attack': (dl_traces, dl_ge),
        'CPA': (cpa_traces, cpa_ge),
    }
    
    # Print summary
    print(f"\n  DL  final rank: {dl_ge[-1] if dl_ge else 'N/A'}")
    print(f"  CPA final rank: {cpa_ge[-1] if cpa_ge else 'N/A'}")
    
    return results
