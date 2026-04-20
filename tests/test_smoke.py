"""Lightweight forward-pass smoke tests for both VAE architectures.

Verifies that models instantiate, produce correct output shapes, and that the
loss function runs without error.  Uses small dimensions so the test finishes
in under a second on CPU.
"""

import torch
from models.vae import VAE
from models.transformer_vae import TransformerVAE
from utils.loss import VAE_Loss

# small dims so this runs fast on CPU
INPUT_DIM = 5          # one-hot nucleotide alphabet (A, C, G, T, N)
SEQ_LENGTH = 128       # short stand-in for the real 1759/1854 nt sequences
BATCH_SIZE = 4


def _check_vae():
    model = VAE(input_dim=INPUT_DIM, hidden_dim=32, latent_dim=16,
                seq_length=SEQ_LENGTH)
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_DIM, SEQ_LENGTH)
    x_hat, mu, logvar = model(x)

    assert x_hat.shape == x.shape, (
        f"VAE output shape {x_hat.shape} != input shape {x.shape}")
    assert mu.shape == (BATCH_SIZE, 16), (
        f"VAE mu shape {mu.shape} != expected (B, 16)")
    assert logvar.shape == mu.shape, (
        f"VAE logvar shape {logvar.shape} != mu shape {mu.shape}")

    criterion = VAE_Loss(beta=0.01)
    loss, recon, kl = criterion(x_hat, x, mu, logvar)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss is not finite"
    print("  Conv1D VAE: OK")


def _check_tvae():
    model = TransformerVAE(
        input_dim=INPUT_DIM, d_model=32, latent_dim=16,
        seq_length=SEQ_LENGTH, nhead=4, num_layers=1,
        dim_feedforward=64, dropout=0.0, num_memory_tokens=4)
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_DIM, SEQ_LENGTH)
    x_hat, mu, logvar = model(x)

    assert x_hat.shape == x.shape, (
        f"TVAE output shape {x_hat.shape} != input shape {x.shape}")
    assert mu.shape == (BATCH_SIZE, 16), (
        f"TVAE mu shape {mu.shape} != expected (B, 16)")
    assert logvar.shape == mu.shape, (
        f"TVAE logvar shape {logvar.shape} != mu shape {mu.shape}")

    criterion = VAE_Loss(beta=0.01)
    loss, recon, kl = criterion(x_hat, x, mu, logvar)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss is not finite"
    print("  Transformer VAE: OK")


def run_smoke_test():
    """Entry point called by run.py."""
    print("Smoke test — forward-pass shape checks")
    with torch.no_grad():
        _check_vae()
        _check_tvae()
    print("All smoke tests passed")
