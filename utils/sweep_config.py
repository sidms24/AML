# global constants and search spaces for Optuna hyperparameter sweeps
# imported by param_sweep.py — edit the dicts here, not the sweep logic

ABLATION_EPOCHS   = 40
ANNEAL_EPOCHS     = 10
WARMUP_EPOCHS     = 10
PATIENCE          = 10
BATCH_SIZE        = 128
SEED              = 42
SAVE_EVERY        = 1000 # effectively no mid-run checkpoints for 40-epoch ablation
N_TRIALS = 100
N_TRIALS_TVAE = 12
ABLATION_EPOCHS_TVAE = 40
PATIENCE_TVAE = 15
ANNEAL_EPOCHS_TVAE = 20
VAE_SEARCH_SPACE = {
    'latent_dim': [8, 16, 32, 64, 128],
    'beta':       (1e-2, 5.0),
     'hidden_dim': [16,32, 64, 128], 
}

VAE_FIXED = {'lr': 3e-5}

TVAE_SEARCH_SPACE = {
    'latent_dim': [32, 64, 128],
    'beta': (1e-2, 0.12),
    'lr': (1.5e-4, 4e-4),
    'dropout': [0.0, 0.1],
    'd_model': [64, 128],
    'num_layers': [2, 3],
    'num_memory_tokens': [4, 8],
}

TVAE_SEARCH_SPACE_STAGE2 = {
    "latent_dim": [64, 128],
    "beta": (0.02, 0.08),
    "lr": (1.6e-4, 3.2e-4),
    "dropout": [0.0, 0.1],
    "d_model": [128],
    "num_layers": [2, 3],
    "num_memory_tokens": [8],
}

TVAE_FIXED = {
    'nhead': 4,
    'weight_decay': 1e-2,
}
