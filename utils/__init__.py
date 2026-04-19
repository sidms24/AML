import warnings
import torch

# TF32 on Ampere+ GPUs — free ~2x matmul speedup, negligible precision loss for our use case
torch.set_float32_matmul_precision('high')

# UMAP forces n_jobs=1 when random_state is set — we want reproducibility so just silence it
warnings.filterwarnings('ignore', message='n_jobs value.*overridden to 1 by setting random_state')

from .encoders import dna_one_hot
from .loss import VAE_Loss
from .inference import extract_latents, latent_analysis, balanced_pre_post_split
from .plot_funcs import (
    plot_training_curves,
    plot_elbow_method,
    plot_elbow,
    plot_embeddings,
    plot_clusters_and_years,
    plot_umap_tsne,
    plot_pre_post_covid,
    plot_full_analysis
)
from .train import VAE_train
from .dataloader import load_data
from .drift import compute_drift_scores, assign_season, compute_blended_drift, compute_latent_diversity, validate_latent_drift
from .sirc_helpers import (
    load_vae_from_hf,
    load_tvae_from_hf,
    extract_vae_latents,
    extract_tvae_latents,
    compute_hamming_drift,
    get_raw_sequences,
    load_cdc_csv,
    bootstrap_corr,
    build_sim_results,
    eval_metrics,
    compute_metrics,
)

__all__ = [
    # encoders
    "dna_one_hot",
    # loss
    "VAE_Loss",
    # inference
    "extract_latents",
    "latent_analysis",
    "balanced_pre_post_split",
    # plot funcs
    "plot_training_curves",
    "plot_elbow_method",
    "plot_elbow",
    "plot_embeddings",
    "plot_clusters_and_years",
    "plot_umap_tsne",
    "plot_pre_post_covid",
    "plot_full_analysis",
    # training
    "VAE_train",
    # data loader
    "load_data",
    # drift scores
    "compute_drift_scores",
    "assign_season",
    "compute_blended_drift",
    "compute_latent_diversity",
    "validate_latent_drift",
    # sirc notebook helpers
    "load_vae_from_hf",
    "load_tvae_from_hf",
    "extract_vae_latents",
    "extract_tvae_latents",
    "compute_hamming_drift",
    "get_raw_sequences",
    "load_cdc_csv",
    "bootstrap_corr",
    "build_sim_results",
    "eval_metrics",
    "compute_metrics",
]
