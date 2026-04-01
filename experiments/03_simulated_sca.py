"""
Experiment 03: Side-Channel Analysis (Simulated)
===================================================
Train DL models on simulated power traces to recover AES key bytes.

Since we don't have ChipWhisperer hardware, we simulate power traces
using the Hamming Weight leakage model:
    Power(t) ∝ HW(SBox(plaintext[i] ⊕ key[i])) + noise

Models predict the S-Box output (256 classes), then we use
log-likelihood accumulation to rank key candidates.

Usage:
    python experiments/03_simulated_sca.py --snr 5.0 --epochs 50
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.synthetic.generator import SimulatedSCADataset
from models.cnn import DeepCNN
from models.transformer import CryptoTransformerSCA
from evaluation.metrics import (
    guessing_entropy_vs_traces, print_metrics_summary, traces_to_recovery
)
from evaluation.visualize import (
    plot_training_curves, plot_ge_vs_traces, plot_model_comparison
)
from utils.aes_ops import SBOX


def parse_args():
    parser = argparse.ArgumentParser(description="Simulated SCA Experiment")
    parser.add_argument("--num-traces", type=int, default=100000, help="Total traces")
    parser.add_argument("--trace-length", type=int, default=700, help="Trace length")
    parser.add_argument("--snr", type=float, default=5.0, help="Signal-to-noise ratio")
    parser.add_argument("--target-byte", type=int, default=0, help="Target key byte")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--model", choices=["cnn", "transformer", "both"], default="both")
    parser.add_argument("--augment", action="store_true",
                        help="Enable trace augmentation (noise, shift, scaling)")
    parser.add_argument("--use-autoencoder", action="store_true",
                        help="Pre-train denoising autoencoder for trace preprocessing")
    parser.add_argument("--compare-cpa", action="store_true",
                        help="Compare DL attack with classical CPA")
    parser.add_argument("--output-dir", type=str, default="./artifacts/results/sca_simulated")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dataloaders(args):
    """Generate simulated SCA dataset."""
    print("\n📦 Generating simulated SCA traces...")
    
    dataset = SimulatedSCADataset(
        num_traces=args.num_traces,
        trace_length=args.trace_length,
        snr=args.snr,
        target_byte=args.target_byte,
        seed=42
    )
    traces, labels, plaintexts, key = dataset.generate()
    
    # Split: 70% train, 15% val, 15% test
    n = len(traces)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    # Normalize traces (z-score)
    mean = traces[:n_train].mean()
    std = traces[:n_train].std()
    traces = (traces - mean) / (std + 1e-8)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(traces[:n_train], dtype=torch.float32),
                       torch.tensor(labels[:n_train], dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(traces[n_train:n_train+n_val], dtype=torch.float32),
                       torch.tensor(labels[n_train:n_train+n_val], dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(traces[n_train+n_val:], dtype=torch.float32),
                       torch.tensor(labels[n_train+n_val:], dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Save test data for GE computation
    test_data = {
        'traces': traces[n_train+n_val:],
        'labels': labels[n_train+n_val:],
        'plaintexts': plaintexts[n_train+n_val:],
        'key': key,
        'mean': mean,
        'std': std,
    }
    
    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n - n_train - n_val:,}")
    print(f"  Trace normalized: mean={mean:.4f}, std={std:.4f}")
    
    return train_loader, val_loader, test_loader, test_data


def train_model(model, model_name, train_loader, val_loader, args, device):
    """Train SCA model with optional augmentation."""
    print(f"\n🚀 Training {model_name}...")
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Setup augmentation
    augmentor = None
    if getattr(args, 'augment', False):
        from utils.preprocessing import TraceAugmentor
        augmentor = TraceAugmentor(
            noise_std=0.05, max_shift=3, scale_range=(0.95, 1.05),
            noise_prob=0.5, shift_prob=0.3, scale_prob=0.3
        )
        print(f"  Augmentation: {augmentor}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_state = None
    patience = 15
    no_improve = 0
    
    start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for bx, by in train_loader:
            # Apply augmentation if enabled
            if augmentor is not None:
                bx = augmentor(bx)
            
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(by)
            t_correct += out.argmax(1).eq(by).sum().item()
            t_total += len(by)
        
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * len(by)
                v_correct += out.argmax(1).eq(by).sum().item()
                v_total += len(by)
        
        scheduler.step()
        
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total
        history['train_loss'].append(t_loss / t_total)
        history['val_loss'].append(v_loss / v_total)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {train_acc*100:.2f}% | "
                  f"Val: {val_acc*100:.2f}% | Best: {best_val_acc*100:.2f}%")
        
        if no_improve >= patience:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    elapsed = time.time() - start
    print(f"  ✅ Done in {elapsed:.0f}s. Best val acc: {best_val_acc*100:.2f}%")
    
    return model, history


def train_autoencoder(traces_train, args, device):
    """
    Pre-train denoising autoencoder on training traces.
    The encoder will be used as a feature extractor for attack models.
    """
    from models.autoencoder import DenoisingAutoencoder
    import torch.nn.functional as F
    
    print("\n🛠️  Pre-training Denoising Autoencoder...")
    
    ae = DenoisingAutoencoder(
        trace_length=args.trace_length, latent_dim=64
    ).to(device)
    
    num_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print(f"  AE Parameters: {num_params:,}")
    
    # Create dataloader from training traces
    ae_dataset = torch.utils.data.TensorDataset(
        torch.tensor(traces_train, dtype=torch.float32)
    )
    ae_loader = DataLoader(ae_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    
    optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_epochs = min(20, args.epochs // 3)  # Don't spend too long on AE
    
    for epoch in range(1, ae_epochs + 1):
        ae.train()
        total_loss = 0
        for (batch,) in ae_loader:
            batch = batch.to(device)
            noisy = DenoisingAutoencoder.add_noise(batch, noise_factor=0.3)
            reconstructed = ae(noisy)
            loss = F.mse_loss(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        
        avg_loss = total_loss / len(traces_train)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  AE Epoch {epoch:3d}/{ae_epochs} | MSE Loss: {avg_loss:.6f}")
    
    print(f"  ✅ Autoencoder pre-training done.")
    return ae


def extract_ae_features(ae, traces, device, batch_size=256):
    """
    Extract features from autoencoder encoder for downstream models.
    Returns: np.array (N, 128) — encoder features
    """
    ae.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(traces), batch_size):
            end = min(start + batch_size, len(traces))
            batch = torch.tensor(traces[start:end], dtype=torch.float32).to(device)
            feat = ae.get_features(batch)  # (batch, 128)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_ge_curve(model, test_data, device, max_traces=2000, step=50):
    """Compute GE vs traces curve for SCA attack."""
    model.eval()
    
    traces = test_data['traces'][:max_traces]
    plaintexts = test_data['plaintexts'][:max_traces]
    key = test_data['key']
    target_byte = 0
    true_key_byte = int(key[target_byte])
    
    num_traces_list = []
    ge_list = []
    
    # Accumulated log-probabilities for each key candidate
    log_probs = np.zeros(256, dtype=np.float64)
    
    with torch.no_grad():
        for i in range(len(traces)):
            x = torch.tensor(traces[i:i+1], dtype=torch.float32).to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            pt_byte = plaintexts[i, target_byte]
            
            for k in range(256):
                sbox_out = SBOX[pt_byte ^ k]
                if probs[sbox_out] > 0:
                    log_probs[k] += np.log(probs[sbox_out] + 1e-36)
            
            if (i + 1) % step == 0:
                sorted_candidates = np.argsort(log_probs)[::-1]
                rank = np.where(sorted_candidates == true_key_byte)[0][0]
                num_traces_list.append(i + 1)
                ge_list.append(rank)
    
    return num_traces_list, ge_list


def run_experiment(args):
    """Main SCA experiment."""
    device = get_device(args.device)
    print(f"\n{'#'*60}")
    print(f"# Simulated SCA Experiment (SNR={args.snr})")
    print(f"# Device: {device}")
    print(f"{'#'*60}")
    
    train_loader, val_loader, test_loader, test_data = create_dataloaders(args)
    true_key_byte = int(test_data['key'][args.target_byte])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ge_results = {}
    results = {}
    
    # ---- Train DeepCNN ----
    if args.model in ["cnn", "both"]:
        cnn = DeepCNN(input_size=args.trace_length, num_classes=256, dropout=0.4)
        cnn, cnn_hist = train_model(cnn, "DeepCNN", train_loader, val_loader, args, device)
        
        # Compute GE curve
        print("  Computing GE curve for DeepCNN...")
        traces_list, ge_list = compute_ge_curve(cnn, test_data, device)
        ge_results['DeepCNN'] = (traces_list, ge_list)
        
        ttr = traces_to_recovery(ge_list, traces_list, threshold_rank=0)
        print(f"  DeepCNN TTR (rank=0): {ttr} traces")
        results['DeepCNN'] = {'ttr': ttr, 'final_ge': ge_list[-1] if ge_list else -1}
        
        plot_training_curves(
            cnn_hist['train_loss'], cnn_hist['val_loss'],
            cnn_hist['train_acc'], cnn_hist['val_acc'],
            save_path=os.path.join(args.output_dir, "deepcnn_training.png"),
            title=f"DeepCNN Training — SCA (SNR={args.snr})"
        )
    
    # ---- Train Transformer ----
    if args.model in ["transformer", "both"]:
        tf = CryptoTransformerSCA(
            trace_length=args.trace_length, patch_size=10,
            num_classes=256, embed_dim=128, num_heads=4, num_layers=4
        )
        tf, tf_hist = train_model(tf, "TransformerSCA", train_loader, val_loader, args, device)
        
        print("  Computing GE curve for TransformerSCA...")
        traces_list, ge_list = compute_ge_curve(tf, test_data, device)
        ge_results['TransformerSCA'] = (traces_list, ge_list)
        
        ttr = traces_to_recovery(ge_list, traces_list, threshold_rank=0)
        print(f"  TransformerSCA TTR (rank=0): {ttr} traces")
        results['TransformerSCA'] = {'ttr': ttr, 'final_ge': ge_list[-1] if ge_list else -1}
        
        plot_training_curves(
            tf_hist['train_loss'], tf_hist['val_loss'],
            tf_hist['train_acc'], tf_hist['val_acc'],
            save_path=os.path.join(args.output_dir, "transformer_training.png"),
            title=f"TransformerSCA Training — SCA (SNR={args.snr})"
        )
    
    # ---- Plot GE comparison ----
    if ge_results:
        plot_ge_vs_traces(
            None, None,
            save_path=os.path.join(args.output_dir, "ge_vs_traces.png"),
            title=f"Guessing Entropy vs Traces (SNR={args.snr})",
            multiple_runs=ge_results
        )
    
    # ---- CPA Comparison ----
    if getattr(args, 'compare_cpa', False) and ge_results:
        print("\n📊 Comparing DL vs CPA...")
        from attacks.classical import CPA
        
        cpa = CPA(target_byte=args.target_byte)
        cpa_traces_list, cpa_ge_list = cpa.ge_vs_traces(
            test_data['traces'][:2000], test_data['plaintexts'][:2000],
            true_key_byte, step=50, max_traces=2000
        )
        ge_results['CPA (classical)'] = (cpa_traces_list, cpa_ge_list)
        
        # Plot combined comparison
        plot_ge_vs_traces(
            None, None,
            save_path=os.path.join(args.output_dir, "dl_vs_cpa_comparison.png"),
            title=f"DL vs CPA Comparison (SNR={args.snr})",
            multiple_runs=ge_results
        )
        
        print(f"  CPA final rank: {cpa_ge_list[-1] if cpa_ge_list else 'N/A'}")
    
    # ---- Save results ----
    summary = {
        'experiment': 'simulated_sca',
        'snr': args.snr,
        'trace_length': args.trace_length,
        'num_traces': args.num_traces,
        'true_key_byte': true_key_byte,
        'augmentation': getattr(args, 'augment', False),
        'autoencoder': getattr(args, 'use_autoencoder', False),
        'results': results,
    }
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Results saved to {args.output_dir}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
