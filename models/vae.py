import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """we kept the baseline conv VAE fairly plain so the comparison stays readable"""

    def __init__(self, input_dim, hidden_dim, latent_dim, seq_length):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        self.encoder_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.SiLU(),
        )

        # we probe the encoder once here so we don't hard-code a flatten size and
        # then forget to update it the next time we change the conv stack
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, seq_length)
            dummy_output = self.encoder_layers(dummy_input)
            self.enc_out_channels = dummy_output.shape[1]
            self.enc_out_length = dummy_output.shape[2]
            self.flat_dim = dummy_output.view(1, -1).shape[1]


        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)      # maps flattened features to latent mean
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)   # maps flattened features to latent log-variance
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)   # projects z back to conv shape for decoding

        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(self.enc_out_channels, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            # we leave this as raw logits because the loss expects class scores, not probabilities
        )

    def reparameterize(self, mu, logvar):
        """sample z from q(z|x) using the reparameterisation trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """x: (B, C, L) -> (x_hat, mu, logvar)"""
        x_enc = self.encoder_layers(x)
        x_flat = x_enc.view(x_enc.size(0), -1)

        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        z_projected = self.fc_decode(z)
        z_reshaped = z_projected.view(z.size(0), self.enc_out_channels, self.enc_out_length)
        x_hat = self.decoder_layers(z_reshaped)

        # we patch the final length here because the strided conv stack can land one
        # step short or long depending on the input length, and we don't want that
        # bookkeeping leaking into every training notebook :(
        if x_hat.shape[2] != self.seq_length:
            x_hat = F.interpolate(x_hat, size=self.seq_length)

        return x_hat, mu, logvar
