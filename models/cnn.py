"""
CNN Models for AES Cryptanalysis
==================================
1D-CNN architectures for:
- Ciphertext/plaintext analysis (SmallCNN)
- Side-channel trace analysis (DeepCNN)

Based on ResNet-style residual blocks with batch normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual Block with optional downsampling."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SmallCNN(nn.Module):
    """
    Small 1D-CNN for ciphertext/plaintext byte analysis.
    
    Input: (batch, 16) for ciphertext-only → reshaped to (batch, 1, 16)
           (batch, 32) for known-plaintext → reshaped to (batch, 1, 32)
           (batch, 48) for chosen-plaintext → reshaped to (batch, 1, 48)
    Output: (batch, num_classes) — probability for each key byte value
    """
    
    def __init__(self, input_size=16, num_classes=256, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(32, 64, kernel_size=3, dropout=dropout),
            ResidualBlock1D(64, 128, kernel_size=3, dropout=dropout),
            ResidualBlock1D(128, 256, kernel_size=3, dropout=dropout),
        )
        
        # Global Average Pooling + Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # x shape: (batch, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1).float()  # (batch, 1, input_size)
        
        x = self.conv_init(x)
        x = self.res_blocks(x)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Deep 1D-CNN for side-channel trace analysis.
    Designed for longer inputs (e.g., 700-point power traces).
    
    Architecture based on SCA-CNN literature (ASCAD paper style).
    
    Input: (batch, trace_length)
    Output: (batch, num_classes)
    """
    
    def __init__(self, input_size=700, num_classes=256, dropout=0.4):
        super().__init__()
        self.input_size = input_size
        
        # Initial feature extraction with larger kernels
        self.conv_init = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Deep residual blocks with progressively smaller kernels
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(32, 32, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(32, 64, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(64, 128, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(128, 256, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(256, 512, kernel_size=3, dropout=dropout),
        )
        
        # Global Average Pooling + Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, trace_length)
        
        x = self.conv_init(x)
        x = self.res_blocks(x)
        x = self.classifier(x)
        return x


# ============================================================
# Model summary helper
# ============================================================
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, input_size):
    """Print model summary."""
    total, trainable = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Input size: {input_size}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"{'='*60}")
    
    # Test forward pass
    x = torch.randn(2, input_size)
    y = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test SmallCNN
    print("Testing SmallCNN (ciphertext-only, 16 bytes):")
    model = SmallCNN(input_size=16, num_classes=256)
    model_summary(model, 16)
    
    print("Testing SmallCNN (known-plaintext, 32 bytes):")
    model = SmallCNN(input_size=32, num_classes=256)
    model_summary(model, 32)
    
    # Test DeepCNN
    print("Testing DeepCNN (SCA traces, 700 points):")
    model = DeepCNN(input_size=700, num_classes=256)
    model_summary(model, 700)
