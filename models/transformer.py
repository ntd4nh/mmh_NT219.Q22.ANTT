"""
Transformer Model for AES Cryptanalysis
==========================================
Transformer-based architecture for learning long-range dependencies
in ciphertext/plaintext byte sequences or power traces.

Uses sinusoidal positional encoding and multi-head self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
    
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CryptoTransformer(nn.Module):
    """
    Transformer for AES Cryptanalysis.
    
    Each byte of the input is treated as a token.
    For ciphertext-only: 16 tokens (ciphertext bytes)
    For known-plaintext: 32 tokens (plaintext + ciphertext bytes)
    
    Architecture:
    - Byte embedding: each byte value [0-255] → embedding vector
    - Positional encoding: sinusoidal
    - Transformer encoder: multi-head self-attention
    - Classification head: mean pooling → FC → 256 classes
    """
    
    def __init__(self, input_size=16, num_classes=256, embed_dim=128,
                 num_heads=4, num_layers=4, ff_dim=512, dropout=0.1,
                 use_byte_embedding=True):
        """
        Args:
            input_size: Number of input tokens (16 for ciphertext, 32 for known-PT)
            num_classes: Number of output classes (256 for key byte prediction)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of Transformer encoder layers
            ff_dim: Feedforward network dimension
            dropout: Dropout rate
            use_byte_embedding: If True, embed byte values; if False, project raw values
        """
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.use_byte_embedding = use_byte_embedding
        
        # Input embedding
        if use_byte_embedding:
            # Each byte value [0-255] gets its own embedding vector
            self.embedding = nn.Embedding(256, embed_dim)
        else:
            # For continuous inputs (e.g., SCA traces)
            self.embedding = nn.Linear(1, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=input_size + 1, dropout=dropout)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_size) — byte values [0-255] or continuous values
        
        Returns:
            (batch, num_classes) — class logits
        """
        batch_size = x.size(0)
        
        if self.use_byte_embedding:
            # Integer byte values → embedding (clamp to valid range)
            x = x.long().clamp(0, 255)
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
        else:
            # Continuous values → projection
            x = x.float().unsqueeze(-1)  # (batch, seq_len, 1)
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, seq_len + 1, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len + 1, embed_dim)
        
        # Use CLS token output for classification
        cls_output = x[:, 0, :]  # (batch, embed_dim)
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits


class CryptoTransformerSCA(CryptoTransformer):
    """
    Transformer variant for SCA traces (continuous input).
    
    Input: power trace (batch, trace_length)
    Output: (batch, 256)
    
    Uses patching: split trace into patches, each patch → embedding.
    """
    
    def __init__(self, trace_length=700, patch_size=10, num_classes=256,
                 embed_dim=128, num_heads=4, num_layers=4, ff_dim=512, dropout=0.1):
        # Calculate number of patches
        num_patches = trace_length // patch_size
        
        super().__init__(
            input_size=num_patches,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_byte_embedding=False
        )
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Override embedding: project patch → embedding
        self.embedding = nn.Linear(patch_size, embed_dim)
        
        # Update positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_patches + 1, dropout=dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, trace_length) — continuous power trace values
        
        Returns:
            (batch, num_classes) — class logits
        """
        batch_size = x.size(0)
        
        # Reshape into patches: (batch, num_patches, patch_size)
        x = x.float()
        x = x[:, :self.num_patches * self.patch_size]  # Trim to fit
        x = x.reshape(batch_size, self.num_patches, self.patch_size)
        
        # Embed patches
        x = self.embedding(x)  # (batch, num_patches, embed_dim)
        
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # CLS token output
        cls_output = x[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_output)
        return logits


if __name__ == "__main__":
    from cnn import model_summary
    
    # Test CryptoTransformer (ciphertext-only)
    print("Testing CryptoTransformer (16 byte input):")
    model = CryptoTransformer(input_size=16, num_classes=256)
    model_summary(model, 16)
    
    # Test CryptoTransformer (known-plaintext)
    print("Testing CryptoTransformer (32 byte input):")
    model = CryptoTransformer(input_size=32, num_classes=256)
    model_summary(model, 32)
    
    # Test CryptoTransformerSCA
    print("Testing CryptoTransformerSCA (700-point trace):")
    model = CryptoTransformerSCA(trace_length=700, patch_size=10)
    x = torch.randn(2, 700)
    y = model(x)
    print(f"Input: {tuple(x.shape)} → Output: {tuple(y.shape)}")
