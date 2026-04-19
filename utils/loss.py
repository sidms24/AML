import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_Loss(nn.Module):
    """we use categorical reconstruction because each position should pick one base, not five independent bits"""

    def __init__(self, beta = 1):
        super(VAE_Loss, self).__init__()
        # we keep beta mutable because the training loop anneals it instead of baking it in
        self.beta = beta

    def forward(self, x_hat, x, mu, logvar):
        targets = x.argmax(dim=1)
        recon_loss = F.cross_entropy(x_hat, targets, reduction='sum')
        # we keep the KL in closed form because there's no need to estimate something
        # the Gaussian VAE already gives us analytically
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (self.beta * kl_loss)
        return total_loss, recon_loss, kl_loss
