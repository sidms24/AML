# helper functions for the SIRC notebook — model loading, latent extraction,
# drift baselines, CDC data loading, and evaluation metrics.
# extracted from sirs_sirc_model.ipynb so the notebook stays clean

import io
from pathlib import Path
import requests
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import jax.numpy as jnp

from .dataloader import load_data
from .encoders import dna_one_hot
from .sirc import simulate_batch


HF_REPO = 'sidms/AML'
HF_BASE = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
LOCAL_WEIGHTS_DIR = ARTIFACTS_DIR / 'weights'
LOCAL_CDC_DIR = ARTIFACTS_DIR / 'cdc_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_checkpoint_bytes(weight_name, hf_base=None, local_dir=None):
    """we prefer local cached weights first so retrained models flow straight into SIRC"""
    search_dir = Path(local_dir) if local_dir is not None else LOCAL_WEIGHTS_DIR
    local_path = search_dir / weight_name
    if local_path.exists():
        return local_path.read_bytes()

    base = hf_base or HF_BASE
    url = f'{base}/weights/{weight_name}'
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _load_cdc_frame(name, hf_base=None, local_dir=None):
    """we keep CDC tables cacheable locally so offline reruns still work"""
    search_dir = Path(local_dir) if local_dir is not None else LOCAL_CDC_DIR
    local_path = search_dir / f'{name}.csv'
    if local_path.exists():
        return pd.read_csv(local_path)

    base = hf_base or HF_BASE
    url = f'{base}/cdc_data/{name}.csv'
    return pd.read_csv(url)


def load_vae_from_hf(weight_name, hf_base=None, dev=None):
    """download weights from HF and rebuild the VAE model"""
    from models.vae import VAE

    dev = dev or device
    checkpoint_bytes = _load_checkpoint_bytes(weight_name, hf_base=hf_base)
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=dev, weights_only=False)
    cfg = checkpoint['config']
    model = VAE(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        latent_dim=cfg['latent_dim'],
        seq_length=cfg['seq_length'],
    ).to(dev)

    # fix for _orig_mod. prefix when loading state_dict (torch.compile artifact)
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, cfg, checkpoint.get('history', {})


def load_tvae_from_hf(weight_name, hf_base=None, dev=None):
    """download weights from HF and rebuild the TransformerVAE model"""
    from models.transformer_vae import TransformerVAE

    dev = dev or device
    checkpoint_bytes = _load_checkpoint_bytes(weight_name, hf_base=hf_base)
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=dev, weights_only=False)
    cfg = checkpoint['config']
    model = TransformerVAE(
    input_dim=cfg['input_dim'],
    d_model=cfg['d_model'],
    latent_dim=cfg['latent_dim'],
    seq_length=cfg['seq_length'],
    nhead=cfg['nhead'],
    num_layers=cfg['num_layers'],
    dim_feedforward=cfg['dim_feedforward'],
    dropout=cfg['dropout'],
    num_memory_tokens=cfg.get('num_memory_tokens', 0),
    ).to(dev)

    # handle _orig_mod. prefix (torch.compile artifact)
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, cfg, checkpoint.get('history', {})



def extract_vae_latents(model, subtype, batch_size=64, dev=None):
    """Returns (latents, years, seasons, months).
    months is a numpy int array (1-12 = collection month, -1 = unknown)."""
    dev = dev or device
    data = load_data(subtype=subtype, encoder=dna_one_hot, batch_size=batch_size)
    train_loader, test_loader = data()
    all_latents, all_years, all_seasons, all_months = [], [], [], []
    model.eval()
    loaders = [train_loader]
    if test_loader is not None:
        loaders.append(test_loader)
    with torch.no_grad():
        for loader in loaders:
            for x, (years, ids, months, seasons) in loader:
                x = x.to(dev).float()
                expected_len = model.seq_length
                if x.shape[-1] < expected_len:
                    pad_len = expected_len - x.shape[-1]
                    x = torch.nn.functional.pad(x, (0, pad_len))
                elif x.shape[-1] > expected_len:
                    x = x[..., :expected_len]
                _, mu, _ = model(x)
                all_latents.append(mu.cpu().numpy())
                all_years.extend(years.numpy())
                all_seasons.extend(seasons)
                all_months.extend(months)  # list of Python ints from collate_fn
    return (np.concatenate(all_latents), np.array(all_years),
            np.array(all_seasons), np.array(all_months, dtype=np.int32))


def extract_tvae_latents(model, subtype, batch_size=64, dev=None):
    """Returns (latents, years, seasons, months).
    months is a numpy int array (1-12 = collection month, -1 = unknown)."""
    dev = dev or device
    data = load_data(subtype=subtype, encoder=dna_one_hot, batch_size=batch_size)
    train_loader, test_loader = data()
    all_latents, all_years, all_seasons, all_months = [], [], [], []
    model.eval()
    loaders = [train_loader]
    if test_loader is not None:
        loaders.append(test_loader)
    with torch.no_grad():
        for loader in loaders:
            for x, (years, ids, months, seasons) in loader:
                x = x.to(dev).float()
                expected_len = model.seq_length
                if x.shape[-1] < expected_len:
                    pad_len = expected_len - x.shape[-1]
                    x = torch.nn.functional.pad(x, (0, pad_len))
                elif x.shape[-1] > expected_len:
                    x = x[..., :expected_len]
                mu, _ = model.encode(x)
                all_latents.append(mu.cpu().numpy())
                all_years.extend(years.numpy())
                all_seasons.extend(seasons)
                all_months.extend(months)
    return (np.concatenate(all_latents), np.array(all_years),
            np.array(all_seasons), np.array(all_months, dtype=np.int32))



def compute_hamming_drift(seqs_onehot, seasons, lag=1):
    """Hamming distance between season consensus sequences (one-hot encoded).
    seqs_onehot: raw one-hot encoded sequences, not latent vectors.
    this is the "dumb baseline" our autoencoders need to beat"""
    unique_seasons = sorted(set(seasons), key=lambda s: int(s.split('-')[0]) if '-' in s else 0)
    consensus = {}
    for s in unique_seasons:
        mask = np.array([ss == s for ss in seasons])
        seqs = seqs_onehot[mask]
        # consensus = argmax per position across the 4 nucleotide channels
        consensus[s] = seqs.mean(axis=0).argmax(axis=0)

    records = []
    for i in range(lag, len(unique_seasons)):
        s_curr = unique_seasons[i]
        s_prev = unique_seasons[i - lag]
        h_dist = np.mean(consensus[s_curr] != consensus[s_prev])
        records.append({'season': s_curr, 'hamming_raw': h_dist})

    df = pd.DataFrame(records)
    vals = df['hamming_raw']
    normed = []
    for i in range(len(vals)):
        window = vals.iloc[:i + 1]
        mn, mx = window.min(), window.max()
        normed.append((vals.iloc[i] - mn) / (mx - mn) if mx > mn else 0.5)
    df['hamming_norm'] = normed
    return df

def get_raw_sequences(subtype, batch_size=64):
    data = load_data(subtype=subtype, encoder=dna_one_hot, batch_size=batch_size)
    train_loader, test_loader = data()
    all_seqs, all_seasons = [], []
    loaders = [train_loader]
    if test_loader is not None:
        loaders.append(test_loader)
    for loader in loaders:
        for x, (years, ids, months, seasons) in loader:
            all_seqs.append(x.numpy())
            all_seasons.extend(seasons)
    return np.concatenate(all_seqs), np.array(all_seasons)



def load_cdc_csv(name, hf_base=None):
    """grab a CDC CSV from the HuggingFace dataset"""
    return _load_cdc_frame(name, hf_base=hf_base)



def bootstrap_corr(x, y, n_boot=1000, seed=42):
    """bootstrap 95% CI for Spearman rho — following Perofsky et al.'s methodology"""
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if len(set(idx)) < 3:
            continue
        rho, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    return np.percentile(rhos, [2.5, 97.5]) if len(rhos) > 0 else (np.nan, np.nan)



def build_sim_results(params, calib_df, valid_df, drift_col='drift_blended',
                      ve_calib=None, ve_valid=None):
    """run SIRC simulation on calib + valid drift and return merged results dataframes"""
    calib_d = jnp.array(calib_df[drift_col].values, dtype=jnp.float32)
    valid_d = jnp.array(valid_df[drift_col].values, dtype=jnp.float32)
    hosp_c = np.array(simulate_batch(params, calib_d, ve_calib))
    hosp_v = np.array(simulate_batch(params, valid_d, ve_valid))

    sc = calib_df[['season', drift_col, 'hosp_rate_overall']].copy()
    sc['hosp_sim'] = hosp_c
    sc = sc.rename(columns={'hosp_rate_overall': 'hosp_obs'})

    sv = valid_df[['season', drift_col, 'hosp_rate_overall']].copy()
    sv['hosp_sim'] = hosp_v
    sv = sv.rename(columns={'hosp_rate_overall': 'hosp_obs'})

    return pd.concat([sc, sv]).sort_values('season').reset_index(drop=True), sc, sv


def eval_metrics(df, label):
    """print RMSE, MAE, Pearson r, Spearman rho for a sim-vs-obs dataframe"""
    obs, sim = df['hosp_obs'].values, df['hosp_sim'].values
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    mae = np.mean(np.abs(sim - obs))
    rrmse = np.sqrt(np.mean(((sim - obs) / obs) ** 2))
    r, p = pearsonr(obs, sim) if len(obs) >= 3 else (np.nan, np.nan)
    rho, _ = spearmanr(obs, sim) if len(obs) >= 3 else (np.nan, np.nan)
    print(f'{label}:')
    print(f'  RMSE  = {rmse:.1f} /100k')
    print(f'  RRMSE = {rrmse:.3f}')
    print(f'  MAE   = {mae:.1f} /100k')
    print(f'  r     = {r:.3f} (p={p:.4f})')
    print(f'  rho   = {rho:.3f}')
    return {'rmse': rmse, 'rrmse': rrmse, 'mae': mae, 'r': r, 'rho': rho}


def compute_metrics(df):
    """RMSE, RRMSE, MAE, Pearson r, Spearman rho — no printing, just returns a dict"""
    obs, sim = df['hosp_obs'].values, df['hosp_sim'].values
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    rrmse = np.sqrt(np.mean(((sim - obs) / obs) ** 2))
    mae = np.mean(np.abs(sim - obs))
    r, p = pearsonr(obs, sim) if len(obs) >= 3 else (np.nan, np.nan)
    rho, _ = spearmanr(obs, sim) if len(obs) >= 3 else (np.nan, np.nan)
    return {'RMSE': rmse, 'RRMSE': rrmse, 'MAE': mae, 'r': r, 'p': p, 'rho': rho}
