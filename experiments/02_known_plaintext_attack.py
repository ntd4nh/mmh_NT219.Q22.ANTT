"""
Experiment 02: Known-Plaintext Attack on AES
================================================
Train DL models to predict the key byte from (plaintext, ciphertext) pairs.

Input: 32 bytes (16-byte plaintext + 16-byte ciphertext)
Output: 256-class prediction of target key byte

Compare performance across different number of AES rounds (1-10).

Usage:
    python experiments/02_known_plaintext_attack.py --rounds 4 --epochs 50
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

from data.synthetic.generator import KnownPlaintextDataset
from models.cnn import SmallCNN
from models.transformer import CryptoTransformer
from evaluation.metrics import (
    classification_accuracy, print_metrics_summary
)
from evaluation.visualize import (
    plot_training_curves, plot_accuracy_vs_rounds, plot_model_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description="Known-Plaintext Attack Experiment")
    parser.add_argument("--rounds", type=int, default=4, help="Number of AES rounds")
    parser.add_argument("--train-samples", type=int, default=200000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=20000, help="Validation samples")
    parser.add_argument("--test-samples", type=int, default=20000, help="Test samples")
    parser.add_argument("--target-byte", type=int, default=0, help="Target key byte")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", choices=["cnn", "transformer", "both"], default="both")
    parser.add_argument("--sweep-rounds", action="store_true",
                        help="Run experiment for rounds 1-6 and plot accuracy vs rounds")
    parser.add_argument("--output-dir", type=str, default="./artifacts/results/known_plaintext")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dataloaders(args, num_rounds=None):
    """Generate known-plaintext dataset."""
    if num_rounds is None:
        num_rounds = args.rounds
    
    total_samples = args.train_samples + args.val_samples + args.test_samples
    
    dataset = KnownPlaintextDataset(
        num_samples=total_samples,
        num_rounds=num_rounds,
        target_byte=args.target_byte,
        num_keys=256,
        seed=42
    )
    inputs, labels, keys = dataset.generate()
    
    # Split
    train_x = inputs[:args.train_samples]
    train_y = labels[:args.train_samples]
    val_x = inputs[args.train_samples:args.train_samples + args.val_samples]
    val_y = labels[args.train_samples:args.train_samples + args.val_samples]
    test_x = inputs[args.train_samples + args.val_samples:]
    test_y = labels[args.train_samples + args.val_samples:]
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                       torch.tensor(train_y, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x, dtype=torch.float32),
                       torch.tensor(val_y, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_x, dtype=torch.float32),
                       torch.tensor(test_y, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, 
                        args, device):
    """Train model and return test accuracy."""
    model = model.to(device)
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
        # Train
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for bx, by in train_loader:
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
        
        # Val
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
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d} | "
                  f"Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | "
                  f"Best: {best_val_acc*100:.2f}%")
        
        if no_improve >= patience:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break
    
    # Load best & test
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            test_correct += out.argmax(1).eq(by).sum().item()
            test_total += len(by)
            all_preds.append(out.cpu().numpy())
            all_labels.append(by.cpu().numpy())
    
    test_acc = test_correct / test_total
    elapsed = time.time() - start
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print(f"  ✅ {model_name} Test Accuracy: {test_acc*100:.2f}% ({elapsed:.0f}s)")
    
    return test_acc, history, all_preds, all_labels


def run_single_experiment(args):
    """Run experiment for a single round count."""
    device = get_device(args.device)
    print(f"\n{'#'*60}")
    print(f"# Known-Plaintext Attack — AES ({args.rounds} rounds)")
    print(f"{'#'*60}")
    
    train_loader, val_loader, test_loader = create_dataloaders(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    if args.model in ["cnn", "both"]:
        cnn = SmallCNN(input_size=32, num_classes=256, dropout=0.3)
        acc, hist, preds, labels = train_and_evaluate(
            cnn, "SmallCNN", train_loader, val_loader, test_loader, args, device
        )
        results['SmallCNN'] = acc
        print_metrics_summary(preds, labels, experiment_name=f"SmallCNN (Known-PT, {args.rounds}R)")
        
        plot_training_curves(
            hist['train_loss'], hist['val_loss'], hist['train_acc'], hist['val_acc'],
            save_path=os.path.join(args.output_dir, f"cnn_r{args.rounds}_curves.png"),
            title=f"SmallCNN — Known-Plaintext {args.rounds}-Round AES"
        )
    
    if args.model in ["transformer", "both"]:
        tf = CryptoTransformer(input_size=32, num_classes=256,
                                embed_dim=128, num_heads=4, num_layers=4)
        acc, hist, preds, labels = train_and_evaluate(
            tf, "Transformer", train_loader, val_loader, test_loader, args, device
        )
        results['Transformer'] = acc
        print_metrics_summary(preds, labels, experiment_name=f"Transformer (Known-PT, {args.rounds}R)")
        
        plot_training_curves(
            hist['train_loss'], hist['val_loss'], hist['train_acc'], hist['val_acc'],
            save_path=os.path.join(args.output_dir, f"tf_r{args.rounds}_curves.png"),
            title=f"Transformer — Known-Plaintext {args.rounds}-Round AES"
        )
    
    return results


def run_round_sweep(args):
    """Run experiment across multiple round counts to show security degradation."""
    device = get_device(args.device)
    print(f"\n{'#'*60}")
    print(f"# Known-Plaintext Attack — Round Sweep (1-6)")
    print(f"{'#'*60}")
    
    all_results = {'SmallCNN': {}, 'Transformer': {}}
    
    for num_rounds in [1, 2, 3, 4, 5, 6]:
        print(f"\n{'='*60}")
        print(f"🔄 Testing {num_rounds} rounds...")
        print(f"{'='*60}")
        
        train_loader, val_loader, test_loader = create_dataloaders(args, num_rounds=num_rounds)
        
        # CNN
        if args.model in ["cnn", "both"]:
            cnn = SmallCNN(input_size=32, num_classes=256, dropout=0.3)
            acc, _, _, _ = train_and_evaluate(
                cnn, "SmallCNN", train_loader, val_loader, test_loader, args, device
            )
            all_results['SmallCNN'][num_rounds] = acc
        
        # Transformer
        if args.model in ["transformer", "both"]:
            tf = CryptoTransformer(input_size=32, num_classes=256)
            acc, _, _, _ = train_and_evaluate(
                tf, "Transformer", train_loader, val_loader, test_loader, args, device
            )
            all_results['Transformer'][num_rounds] = acc
    
    # Plot results
    os.makedirs(args.output_dir, exist_ok=True)
    plot_accuracy_vs_rounds(
        all_results,
        save_path=os.path.join(args.output_dir, "accuracy_vs_rounds.png"),
        title="Known-Plaintext Attack: Accuracy vs AES Rounds"
    )
    
    # Save results
    serializable = {k: {str(rk): rv for rk, rv in v.items()} for k, v in all_results.items()}
    with open(os.path.join(args.output_dir, "round_sweep_results.json"), 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"📊 Results Summary: Accuracy vs Rounds")
    print(f"{'='*60}")
    print(f"{'Rounds':<8}", end="")
    for model_name in all_results:
        if all_results[model_name]:
            print(f"{model_name:<20}", end="")
    print()
    print("-" * 50)
    
    for r in [1, 2, 3, 4, 5, 6]:
        print(f"{r:<8}", end="")
        for model_name in all_results:
            if r in all_results[model_name]:
                acc = all_results[model_name][r]
                print(f"{acc*100:>6.2f}%             ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()
    
    print(f"\nRandom baseline: {100/256:.2f}%")
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    
    if args.sweep_rounds:
        run_round_sweep(args)
    else:
        run_single_experiment(args)
