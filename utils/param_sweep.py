import gc
import torch
import numpy as np
import optuna
from sklearn.metrics import silhouette_score

from models.vae import VAE
from models.transformer_vae import TransformerVAE
from utils.loss import VAE_Loss
from utils.train import VAE_train
from utils.inference import extract_latents

from utils.sweep_config import (
    ABLATION_EPOCHS,
    ANNEAL_EPOCHS,
    PATIENCE,
    N_TRIALS,
    SEED,
    SAVE_EVERY,
    VAE_SEARCH_SPACE,
    VAE_FIXED,
    TVAE_SEARCH_SPACE,
    TVAE_FIXED,
    N_TRIALS_TVAE,
    ABLATION_EPOCHS_TVAE,
    PATIENCE_TVAE,
    ANNEAL_EPOCHS_TVAE, 
    TVAE_SEARCH_SPACE_STAGE2 
)


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _silhouette(model, loader, device):
    """extract latents and compute silhouette on true season labels"""
    model.eval()
    result = extract_latents(model, loader, device=device)
    latents = result[0]

    # prefer season labels if available, else fall back to years
    if len(result) > 4 and result[4] is not None and len(result[4]) > 0:
        labels = result[4]
    else:
        labels = result[1]

    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0

    label_map = {s: i for i, s in enumerate(unique)}
    int_labels = np.array([label_map[s] for s in labels])

    try:
        return float(silhouette_score(latents, int_labels))
    except Exception:
        return 0.0



def run_vae_sweep(
    train_loader,
    test_loader,
    input_dim,
    seq_length,
    device,
    n_trials=N_TRIALS,
    epochs=ABLATION_EPOCHS,
    patience=PATIENCE,
    anneal_epochs=ANNEAL_EPOCHS,
    hidden_dim=None,
    latent_dim=None,
    lr=None,
    collapse_threshold=1e-2,
    collapse_penalty=1e6,
    search_space=None,
):
    """multi-objective sweep over latent_dim, beta, and hidden_dim.
    any argument passed explicitly is fixed.
    pass search_space=VAE_SEARCH_SPACE_STAGE2 for a focused second run"""

    if search_space is None:
        search_space = VAE_SEARCH_SPACE

    fixed_lr = lr if lr is not None else VAE_FIXED.get("lr", 2e-4)

    def objective(trial):
        model = None

        lat = latent_dim if latent_dim is not None else trial.suggest_categorical(
            "latent_dim", search_space["latent_dim"]
        )

        hid = hidden_dim if hidden_dim is not None else trial.suggest_categorical(
            "hidden_dim", search_space["hidden_dim"]
        )

        beta_lo, beta_hi = search_space["beta"]
        beta = trial.suggest_float("beta", beta_lo, beta_hi, log=True)

        criterion = VAE_Loss(beta=beta)
        model = VAE(
            input_dim=input_dim,
            hidden_dim=hid,
            latent_dim=lat,
            seq_length=seq_length,
        ).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=fixed_lr)

        try:
            model, history = VAE_train(
                model,
                train_loader,
                test_loader,
                criterion,
                optimiser,
                device,
                epochs=epochs,
                patience=patience,
                anneal_epochs=anneal_epochs,
                save_every=SAVE_EVERY,
                tqd_bar=False,
            )

            test_recon = float(history["test_recon_loss"][-1])
            test_kl = float(history["test_kl_loss"][-1])
            test_total = float(history["test_tloss"][-1])

            if np.isnan(test_recon) or np.isinf(test_recon):
                trial.set_user_attr("diverged", True)
                return float("inf"), float("inf")

            sil = _silhouette(model, test_loader, device)
            collapsed = test_kl < collapse_threshold

            trial.set_user_attr("test_recon", test_recon)
            trial.set_user_attr("test_kl", test_kl)
            trial.set_user_attr("test_total", test_total)
            trial.set_user_attr("silhouette", sil)
            trial.set_user_attr("kl_collapsed", collapsed)
            trial.set_user_attr("epochs_trained", len(history["train_tloss"]))
            trial.set_user_attr("diverged", False)

            recon_objective = test_recon + (collapse_penalty if collapsed else 0.0)
            return recon_objective, -sil

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            trial.set_user_attr("diverged", True)
            return float("inf"), float("inf")

        finally:
            del model
            _clear_memory()

    study = optuna.create_study(
        study_name="vae_hparam_sweep",
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study



def run_tvae_sweep(
    train_loader,
    test_loader,
    input_dim,
    seq_length,
    device,
    n_trials=N_TRIALS_TVAE,
    epochs=ABLATION_EPOCHS_TVAE,
    patience=PATIENCE_TVAE,
    anneal_epochs=ANNEAL_EPOCHS_TVAE,
    latent_dim=None,
    d_model=None,
    nhead=None,
    num_layers=None,
    dim_feedforward=None,
    num_memory_tokens=None,
    weight_decay=None,
    collapse_threshold=1e-2,
    collapse_penalty=1e6,
    search_space=None,
):
    """multi-objective TVAE sweep over latent_dim, beta, lr, dropout, d_model, num_layers, num_memory_tokens.
    nhead and weight_decay are fixed by default.
    any argument passed explicitly is fixed.
    dim_feedforward is tied to d_model unless explicitly fixed (64->256, 128->512, else 4*d_model)"""

    if search_space is None:
        search_space = TVAE_SEARCH_SPACE

    fixed_nhead = nhead if nhead is not None else TVAE_FIXED.get("nhead", 4)
    fixed_wd = weight_decay if weight_decay is not None else TVAE_FIXED.get("weight_decay", 1e-2)

    def resolve_dim_feedforward(curr_d_model):
        if dim_feedforward is not None:
            return dim_feedforward
        if curr_d_model == 64:
            return 256
        if curr_d_model == 128:
            return 512
        return 4 * curr_d_model

    def objective(trial):
        model = None

        lat = latent_dim if latent_dim is not None else trial.suggest_categorical(
            "latent_dim", search_space["latent_dim"]
        )

        curr_d_model = d_model if d_model is not None else trial.suggest_categorical(
            "d_model", search_space["d_model"]
        )

        curr_num_layers = num_layers if num_layers is not None else trial.suggest_categorical(
            "num_layers", search_space["num_layers"]
        )

        curr_num_memory_tokens = (
            num_memory_tokens
            if num_memory_tokens is not None
            else trial.suggest_categorical("num_memory_tokens", search_space["num_memory_tokens"])
        )

        if curr_d_model % fixed_nhead != 0:
            raise ValueError(
                f"Invalid architecture: d_model={curr_d_model} must be divisible by nhead={fixed_nhead}"
            )

        curr_dim_ff = resolve_dim_feedforward(curr_d_model)

        beta_lo, beta_hi = search_space["beta"]
        beta = trial.suggest_float("beta", beta_lo, beta_hi, log=True)

        lr_lo, lr_hi = search_space["lr"]
        lr = trial.suggest_float("lr", lr_lo, lr_hi, log=True)

        dropout = trial.suggest_categorical("dropout", search_space["dropout"])

        criterion = VAE_Loss(beta=beta)

        model = TransformerVAE(
            input_dim=input_dim,
            d_model=curr_d_model,
            latent_dim=lat,
            seq_length=seq_length,
            nhead=fixed_nhead,
            num_layers=curr_num_layers,
            dim_feedforward=curr_dim_ff,
            dropout=dropout,
            num_memory_tokens=curr_num_memory_tokens,
        ).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=fixed_wd,
        )

        try:
            model, history = VAE_train(
                model,
                train_loader,
                test_loader,
                criterion,
                optimiser,
                device,
                epochs=epochs,
                patience=patience,
                anneal_epochs=anneal_epochs,
                save_every=SAVE_EVERY,
                tqd_bar=False,
            )

            test_recon = float(history["test_recon_loss"][-1])
            test_kl = float(history["test_kl_loss"][-1])
            test_total = float(history["test_tloss"][-1])

            if np.isnan(test_recon) or np.isinf(test_recon):
                trial.set_user_attr("diverged", True)
                return float("inf"), float("inf")

            sil = _silhouette(model, test_loader, device)
            collapsed = test_kl < collapse_threshold

            trial.set_user_attr("test_recon", test_recon)
            trial.set_user_attr("test_kl", test_kl)
            trial.set_user_attr("test_total", test_total)
            trial.set_user_attr("silhouette", sil)
            trial.set_user_attr("kl_collapsed", collapsed)
            trial.set_user_attr("epochs_trained", len(history["train_tloss"]))
            trial.set_user_attr("diverged", False)

            trial.set_user_attr("d_model", curr_d_model)
            trial.set_user_attr("nhead", fixed_nhead)
            trial.set_user_attr("num_layers", curr_num_layers)
            trial.set_user_attr("dim_feedforward", curr_dim_ff)
            trial.set_user_attr("num_memory_tokens", curr_num_memory_tokens)
            trial.set_user_attr("weight_decay", fixed_wd)
            trial.set_user_attr("sweep_epochs", epochs)
            trial.set_user_attr("sweep_patience", patience)
            trial.set_user_attr("sweep_anneal_epochs", anneal_epochs)

            recon_objective = test_recon + (collapse_penalty if collapsed else 0.0)
            return recon_objective, -sil

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            trial.set_user_attr("diverged", True)
            return float("inf"), float("inf")

        finally:
            del model
            _clear_memory()

    study = optuna.create_study(
        study_name="tvae_hparam_sweep",
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
