"""
Denoising Autoencoder for Trace Preprocessing
================================================
Autoencoder for denoising power traces before feeding
into attack models. Improves SNR and alignment invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for denoising power traces.
    
    Architecture:
    - Encoder: Conv1d → BN → ReLU → MaxPool (×3) → bottleneck
    - Decoder: ConvTranspose1d → BN → ReLU (×3) → output
    
    Training: Add Gaussian noise to input, train to reconstruct clean trace.
    Inference: Use encoder output as feature extractor for downstream models.
    """
    
    def __init__(self, trace_length=700, latent_dim=64):
        super().__init__()
        self.trace_length = trace_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(latent_dim),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * latent_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 1, kernel_size=3, padding=1),
        )
        
        # Final projection to match original trace length
        self.output_proj = nn.Linear(latent_dim * 4, trace_length)
    
    def encode(self, x):
        """Encode trace to latent representation."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, trace_length)
        z = self.encoder(x)  # (batch, 128, latent_dim)
        return z
    
    def decode(self, z):
        """Decode latent representation back to trace."""
        batch_size = z.size(0)
        z_flat = self.bottleneck(z)
        z_reshaped = z_flat.reshape(batch_size, 128, self.latent_dim)
        out = self.decoder(z_reshaped)  # (batch, 1, latent_dim*4)
        out = out.squeeze(1)  # (batch, latent_dim*4)
        out = self.output_proj(out)  # (batch, trace_length)
        return out
    
    def forward(self, x):
        """Full autoencoder: encode then decode."""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
    
    def get_features(self, x):
        """Extract features from encoder (for downstream tasks)."""
        z = self.encode(x)
        return z.mean(dim=-1)  # (batch, 128) — global average over spatial dim
    
    @staticmethod
    def add_noise(x, noise_factor=0.3):
        """Add Gaussian noise for denoising training."""
        noise = torch.randn_like(x) * noise_factor
        return x + noise


if __name__ == "__main__":
    # Test autoencoder
    model = DenoisingAutoencoder(trace_length=700, latent_dim=64)
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"DenoisingAutoencoder — Parameters: {total:,}")
    
    # Test forward
    x = torch.randn(4, 700)
    noisy_x = DenoisingAutoencoder.add_noise(x)
    reconstructed = model(noisy_x)
    print(f"Input:         {tuple(x.shape)}")
    print(f"Noisy input:   {tuple(noisy_x.shape)}")
    print(f"Reconstructed: {tuple(reconstructed.shape)}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features:      {tuple(features.shape)}")
    
    # Reconstruction loss
    loss = F.mse_loss(reconstructed, x)
    print(f"Reconstruction loss (untrained): {loss.item():.4f}")
