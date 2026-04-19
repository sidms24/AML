import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """we stayed with sinusoidal positions so this piece is standard and easy to justify"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, L, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerVAE(nn.Module):
    """we mirrored the Conv1D VAE interface so the training code can compare both models cleanly"""

    def __init__(self, input_dim, d_model, latent_dim, seq_length,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
                 num_memory_tokens=8):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_memory_tokens = num_memory_tokens
        self.enc_embed = nn.Linear(input_dim, d_model)
        # we use a learned CLS token instead of mean pooling because we want the model
        # to decide what summary of the full HA sequence is worth keeping
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.enc_pos = PositionalEncoding(d_model, max_len=seq_length + 2,
                                          dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # we turn z into a small memory bank so each decoded position can pull a
        # different view of the latent state instead of sharing one broadcast vector
        self.z_to_memory = nn.Linear(latent_dim, num_memory_tokens * d_model)
        self.dec_queries = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        self.dec_pos = PositionalEncoding(d_model, max_len=seq_length + 1,
                                          dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_layers)
        self.dec_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x):
        """x: (B, C, L) -> mu, logvar each (B, latent_dim)"""
        x = x.transpose(1, 2)
        x = self.enc_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.enc_pos(x)
        x = self.encoder(x)
        x = self.enc_norm(x)
        cls_out = x[:, 0, :]
        return self.fc_mu(cls_out), self.fc_logvar(cls_out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """z: (B, latent_dim) -> x_hat: (B, C, L)"""
        B = z.size(0)
        memory = self.z_to_memory(z)
        memory = memory.reshape(B, self.num_memory_tokens, self.d_model)
        queries = self.dec_queries.expand(B, -1, -1)
        queries = self.dec_pos(queries)
        h = self.decoder(queries, memory)
        h = self.dec_norm(h)
        h = self.output_proj(h)
        return h.transpose(1, 2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
