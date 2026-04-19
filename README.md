# Latent Sequence Drift for Influenza Surveillance

I built this project to test whether influenza HA nucleotide sequences can be turned into a useful epidemiological signal. The pipeline chains three steps: sequence autoencoding, latent drift estimation, and a season-level SIRC compartmental model for FluSurv-NET hospitalisation rates. I compare a Conv1D VAE against a Transformer VAE, validate both against a raw Hamming-distance baseline, and check whether learned drift scores improve forecasting.

The main result is mixed in a useful way: both neural models recover biologically sensible sequence drift, but the full VAE/TVAE-to-SIRC pipeline does not beat simple hospitalisation baselines. That makes the representation-learning part defensible while giving a clear negative result about where the bottleneck sits.

## Repository Layout

```text
.
├── run.py                          # single-command reproducible entry point
├── main.pdf                        # compiled report
├── models/
│   ├── __init__.py
│   ├── vae.py                      # Conv1D VAE
│   └── transformer_vae.py          # Transformer VAE with CLS token + memory bank decoder
├── notebooks/
│   ├── seq_data.ipynb              # raw FASTA -> aligned, deduplicated HF datasets
│   ├── vae_training_notebook.ipynb
│   ├── transformer_vae_training_notebook.ipynb
│   ├── hyperparameter_VAE.ipynb    # Optuna sweep (Conv1D VAE)
│   ├── hyperparameter_VAE_64.ipynb # Optuna sweep (Conv1D VAE, latent=64)
│   ├── hyperparameter_TVAE.ipynb   # Optuna sweep (Transformer VAE)
│   ├── VAE_hp_analysis.ipynb       # sweep visualisation + best config selection
│   ├── TVAE_hp_analysis.ipynb
│   ├── cdc_surveillance_pipeline.ipynb  # CDC FluSurv-NET / ILINet / NREVSS ingestion
│   └── sirs_sirc_model.ipynb       # SIRC calibration, forecasting, and evaluation
├── utils/                          # shared library code
│   ├── __init__.py
│   ├── dataloader.py               # HuggingFace dataset loader with one-hot encoding
│   ├── encoders.py                 # DNA one-hot encoder
│   ├── loss.py                     # beta-VAE loss (categorical reconstruction + KL)
│   ├── train.py                    # training loop with AMP, grad clipping, early stopping
│   ├── inference.py                # latent extraction from trained checkpoints
│   ├── drift.py                    # seasonal drift scoring
│   ├── param_sweep.py              # Optuna sweep runner
│   ├── sweep_config.py             # sweep search spaces
│   ├── sirc.py                     # JAX-based SIRC ODE solver
│   ├── sirc_helpers.py             # calibration, LOO-CV, sensitivity analysis
│   └── plot_funcs.py               # shared plotting utilities
├── requirements.txt
└── environment.yml                 # pinned conda environment
```

## Data

All large artefacts live on the HuggingFace dataset hub at [`sidms/AML`](https://huggingface.co/datasets/sidms/AML), keeping the Git repo lightweight. This includes:

| Artefact | Location |
|---|---|
| Processed sequence datasets | `H1N1_global`, `H3N2_global`, `H1N1_drift`, `H3N2_drift`, + temporal splits |
| Trained model weights | `weights/vae_h1n1_weights.pt`, `weights/vae_h3n2_weights.pt`, `weights/tvae_h1n1_weights.pt`, `weights/tvae_h3n2_weights.pt` |
| CDC surveillance tables | `cdc_data/*.csv` |
| Optuna sweep results | `sweep_results/`, `sweep_results_tvae/` |

## Quick Start

### Option 1: Run locally

```bash
pip install -r requirements.txt
python run.py                       # uses hosted weights, runs analysis notebooks
```

### Option 2: Google Colab

```python
!git clone https://github.com/sidms24/AML.git
%cd AML
!pip install -q -r requirements.txt
!python run.py
```

Make sure to select a **GPU runtime** (Runtime -> Change runtime type -> T4 GPU).

### Option 3: Conda (fully pinned)

```bash
conda env create -f environment.yml
conda activate aml-final-repro
python run.py
```

## `run.py` Flags

| Flag | Effect |
|---|---|
| *(default)* | Download hosted weights + data, run smoke test |
| `--data-clean` | Run CDC surveillance pipeline and sequence data notebooks |
| `--tuning` | Rerun Optuna hyperparameter sweeps and analysis notebooks |
| `--train` | Retrain all four models from scratch |
| `--sirc` | Run the SIRC calibration and evaluation notebook |
| `--smoke-test-only` | Run forward-pass shape checks and exit |
| `--skip-download` | Skip HuggingFace prefetch (use existing cache) |
| `--skip-smoke-test` | Skip the smoke test |

## Reproducing from Scratch

If i wanted to rebuild everything from raw inputs, I would run the notebooks in this order:

1. `seq_data.ipynb` — FASTA alignment and deduplication
2. `cdc_surveillance_pipeline.ipynb` — CDC data ingestion
3. `hyperparameter_VAE.ipynb` / `hyperparameter_VAE_64.ipynb` / `hyperparameter_TVAE.ipynb` — sweeps
4. `VAE_hp_analysis.ipynb` / `TVAE_hp_analysis.ipynb` — sweep analysis
5. `vae_training_notebook.ipynb` — Conv1D VAE training
6. `transformer_vae_training_notebook.ipynb` — Transformer VAE training
7. `sirs_sirc_model.ipynb` — SIRC calibration and evaluation

Or equivalently: `python run.py --data-clean --tuning --train --sirc`.

For verifying the final reported results only (using hosted weights and processed data): `python run.py --sirc`.

## Key Results

- Both VAE architectures learn latent representations whose seasonal drift correlates significantly with Hamming-distance drift for H1N1 and H3N2.
- The Transformer VAE is slower and reconstructs worse than the Conv1D VAE in this setup.
- The SIRC stage is the weak link: learned drift features do not beat simple hospitalisation baselines, while PCA-SIRC performs better.
- The project supports a useful conclusion even though the end-to-end forecasting result is negative.

