"""
Experiment 01: Ciphertext-Only Attack on Toy-AES
===================================================
Train DL models (CNN & Transformer) to predict the key byte
from ciphertext alone, using reduced-round AES.

This is the PRIMARY experiment demonstrating that DL can
learn cryptographic patterns from ciphertext.

Usage:
    python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.synthetic.generator import CiphertextOnlyDataset
from models.cnn import SmallCNN
from models.transformer import CryptoTransformer
from evaluation.metrics import (
    classification_accuracy, key_rank, success_rate, print_metrics_summary
)
from evaluation.visualize import (
    plot_training_curves, plot_confusion_matrix, 
    plot_key_rank_distribution, plot_model_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ciphertext-Only Attack Experiment")
    parser.add_argument("--rounds", type=int, default=2, help="Number of AES rounds (1-4)")
    parser.add_argument("--train-samples", type=int, default=200000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=20000, help="Validation samples")
    parser.add_argument("--test-samples", type=int, default=20000, help="Test samples")
    parser.add_argument("--target-byte", type=int, default=0, help="Target key byte (0-15)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", choices=["cnn", "transformer", "both"], default="both",
                        help="Model to train")
    parser.add_argument("--output-dir", type=str, default="./artifacts/results/ciphertext_only",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dataloaders(args):
    """Generate datasets and create DataLoaders."""
    print("\n📦 Generating datasets...")
    
    total_samples = args.train_samples + args.val_samples + args.test_samples
    
    dataset = CiphertextOnlyDataset(
        num_samples=total_samples,
        num_rounds=args.rounds,
        target_byte=args.target_byte,
        seed=42
    )
    ciphertexts, labels, key, plaintexts = dataset.generate()
    
    # Split into train/val/test
    train_ct = ciphertexts[:args.train_samples]
    train_labels = labels[:args.train_samples]
    
    val_ct = ciphertexts[args.train_samples:args.train_samples + args.val_samples]
    val_labels = labels[args.train_samples:args.train_samples + args.val_samples]
    
    test_ct = ciphertexts[args.train_samples + args.val_samples:]
    test_labels = labels[args.train_samples + args.val_samples:]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(train_ct, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_ct, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_ct, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    print(f"  Key byte [{args.target_byte}] = {key[args.target_byte]}")
    
    return train_loader, val_loader, test_loader, key


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += len(batch_y)
    
    return total_loss / total, correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * len(batch_y)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += len(batch_y)
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return total_loss / total, correct / total, all_outputs, all_labels


def train_model(model, model_name, train_loader, val_loader, args, device):
    """Full training loop with early stopping."""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    print(f"  Device: {device}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }
    
    best_val_acc = 0
    best_epoch = 0
    patience = 15
    no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), 
                       os.path.join(args.output_dir, f"{model_name}_best.pth"))
        else:
            no_improve += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
                  f"Best: {best_val_acc*100:.2f}% (ep {best_epoch}) | "
                  f"Time: {elapsed:.0f}s")
        
        if no_improve >= patience:
            print(f"  ⛔ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    total_time = time.time() - start_time
    print(f"\n  ✅ Training complete in {total_time:.0f}s")
    print(f"  Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, f"{model_name}_best.pth"),
                    weights_only=True)
    )
    
    return model, history


def run_experiment(args):
    """Main experiment runner."""
    device = get_device(args.device)
    print(f"\n{'#'*60}")
    print(f"# Ciphertext-Only Attack — Toy-AES ({args.rounds} rounds)")
    print(f"# Device: {device}")
    print(f"{'#'*60}")
    
    # Create data
    train_loader, val_loader, test_loader, key = create_dataloaders(args)
    true_key_byte = int(key[args.target_byte])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # ---- Train CNN ----
    if args.model in ["cnn", "both"]:
        cnn = SmallCNN(input_size=16, num_classes=256, dropout=0.3)
        cnn, cnn_history = train_model(cnn, "SmallCNN", train_loader, val_loader, args, device)
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, cnn_preds, test_labels = evaluate(cnn, test_loader, criterion, device)
        
        print_metrics_summary(cnn_preds, test_labels, true_key_byte, "SmallCNN (Test)")
        
        results['SmallCNN'] = {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'history': cnn_history,
        }
        
        # Save plots
        plot_training_curves(
            cnn_history['train_loss'], cnn_history['val_loss'],
            cnn_history['train_acc'], cnn_history['val_acc'],
            save_path=os.path.join(args.output_dir, "cnn_training_curves.png"),
            title=f"SmallCNN Training — {args.rounds}-Round AES"
        )
        plot_key_rank_distribution(
            cnn_preds, true_key_byte,
            save_path=os.path.join(args.output_dir, "cnn_key_rank_dist.png"),
            title=f"SmallCNN Key Rank Distribution — {args.rounds}-Round AES"
        )
    
    # ---- Train Transformer ----
    if args.model in ["transformer", "both"]:
        transformer = CryptoTransformer(
            input_size=16, num_classes=256,
            embed_dim=128, num_heads=4, num_layers=4,
            ff_dim=512, dropout=0.1
        )
        transformer, tf_history = train_model(
            transformer, "CryptoTransformer", train_loader, val_loader, args, device
        )
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, tf_preds, test_labels = evaluate(
            transformer, test_loader, criterion, device
        )
        
        print_metrics_summary(tf_preds, test_labels, true_key_byte, "CryptoTransformer (Test)")
        
        results['CryptoTransformer'] = {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'history': tf_history,
        }
        
        # Save plots
        plot_training_curves(
            tf_history['train_loss'], tf_history['val_loss'],
            tf_history['train_acc'], tf_history['val_acc'],
            save_path=os.path.join(args.output_dir, "transformer_training_curves.png"),
            title=f"CryptoTransformer Training — {args.rounds}-Round AES"
        )
        plot_key_rank_distribution(
            tf_preds, true_key_byte,
            save_path=os.path.join(args.output_dir, "transformer_key_rank_dist.png"),
            title=f"CryptoTransformer Key Rank Distribution — {args.rounds}-Round AES"
        )
    
    # ---- Model Comparison ----
    if args.model == "both" and len(results) == 2:
        comparison = {name: r['test_acc'] for name, r in results.items()}
        comparison['Random Baseline'] = 1.0 / 256
        
        plot_model_comparison(
            comparison, metric_name='Test Accuracy',
            save_path=os.path.join(args.output_dir, "model_comparison.png"),
            title=f"Model Comparison — Ciphertext-Only {args.rounds}-Round AES"
        )
    
    # ---- Save results summary ----
    summary = {
        'experiment': 'ciphertext_only',
        'num_rounds': args.rounds,
        'target_byte': args.target_byte,
        'true_key_byte': true_key_byte,
        'train_samples': args.train_samples,
        'device': str(device),
    }
    for name, r in results.items():
        summary[name] = {
            'test_accuracy': r['test_acc'],
            'test_loss': r['test_loss'],
            'random_baseline': 1.0/256,
            'improvement_over_random': r['test_acc'] / (1.0/256),
        }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📁 Results saved to {args.output_dir}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
