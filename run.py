from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download

from models.vae import VAE
from models.transformer_vae import TransformerVAE
from tests.test_smoke import run_smoke_test
from utils.dataloader import load_data
from utils.encoders import dna_one_hot
from utils.loss import VAE_Loss
from utils.train import VAE_train


ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"
CDC_DIR = ARTIFACTS_DIR / "cdc_data"
EXECUTED_NOTEBOOKS_DIR = ARTIFACTS_DIR / "executed_notebooks"
HF_REPO = "sidms/AML"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this is the one knob the marker asked for.
# leave it False to use hosted weights/data, flip it to True to retrain the four models locally
TRAIN = False

# if this is False we use the processed HF artefacts.
# flip it to True if we want to start from the raw CDC zip + raw sequence files instead
DATA_CLEAN = False

# if this is False we pull the hosted Optuna studies and just make the analysis plots.
# flip it to True only if we want to rerun the sweeps themselves, because those take much longer
Tuning = False

# SIRC notebook is off by default because it depends on CDC data and trained weights
RUN_SIRC_NOTEBOOK = False
RUN_SMOKE_TEST = True
DOWNLOAD_FROM_HF = True
SEED = 42


@dataclass(frozen=True)
class NotebookJob:
    path: Path
    replacements: dict[str, str] = field(default_factory=dict)


def set_seed(seed: int = SEED) -> None:
    """we pin every RNG we use here so reruns don't drift for silly reasons"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for path in (ARTIFACTS_DIR, WEIGHTS_DIR, CDC_DIR, EXECUTED_NOTEBOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def prefetch_hf_assets(train: bool, data_clean: bool, tuning: bool) -> None:
    """we cache the public artefacts up front so a fresh clone doesn't need manual HF steps"""
    patterns = ["weights/*", "cdc_data/*", "sweep_results/*", "sweep_results_tvae/*"]
    if data_clean:
        patterns.append("data/*")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(ARTIFACTS_DIR),
        allow_patterns=patterns,
    )

    dataset_configs = ["H1N1_drift", "H3N2_drift"]
    if train:
        dataset_configs.extend(["H1N1_global", "H3N2_global"])
    if data_clean or tuning:
        dataset_configs.extend(["H1N1_global_temporal", "H3N2_global_temporal"])
    if data_clean:
        dataset_configs.extend(["H1N1_global", "H3N2_global"])

    for config_name in sorted(set(dataset_configs)):
        # we only need to trigger datasets' own download/cache logic here
        load_dataset(HF_REPO, config_name)


def save_checkpoint(weight_name: str, model: torch.nn.Module, config: dict, history: dict, extra: dict | None = None) -> Path:
    """we mirror the notebook checkpoint shape so the existing helper loaders keep working"""
    payload = {
        "config": config,
        "model_state_dict": model.state_dict(),
        "history": history,
    }
    if extra:
        payload.update(extra)

    out_path = WEIGHTS_DIR / weight_name
    torch.save(payload, out_path)
    return out_path


def train_vae(subtype: str, weight_name: str) -> Path:
    """we train from the public processed HF datasets so retraining stays one command away"""
    best_config = {"latent_dim": 64, "beta": 0.010239273411172712, "hidden_dim": 128}
    batch_size = 512
    epochs = 200
    lr = 3e-5
    anneal_epochs = 10
    patience = 10

    data = load_data(subtype=subtype, encoder=dna_one_hot, batch_size=batch_size)
    train_loader, test_loader = data()
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    seq_length = sample_x.shape[2]

    model = VAE(
        input_dim=input_dim,
        hidden_dim=best_config["hidden_dim"],
        latent_dim=best_config["latent_dim"],
        seq_length=seq_length,
    ).to(DEVICE)
    criterion = VAE_Loss(beta=best_config["beta"])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training Conv1D VAE for {subtype} on {DEVICE} ...")
    model, history = VAE_train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimiser,
        DEVICE,
        epochs=epochs,
        patience=patience,
        anneal_epochs=anneal_epochs,
        save_every=0,
        save_path=str(WEIGHTS_DIR / f"checkpoint_{weight_name}"),
    )

    return save_checkpoint(
        weight_name=weight_name,
        model=model,
        config={
            "input_dim": input_dim,
            "hidden_dim": best_config["hidden_dim"],
            "latent_dim": best_config["latent_dim"],
            "seq_length": seq_length,
        },
        history=history,
        extra={"best_config": best_config},
    )


def train_tvae(subtype: str, weight_name: str) -> Path:
    """we keep the transformer defaults identical to the submitted notebook"""
    best_config = {
        "latent_dim": 128,
        "beta": 0.028630640852697415,
        "d_model": 128,
        "num_layers": 3,
        "num_memory_tokens": 8,
        "lr": 0.0002532556033608517,
        "dropout": 0.1,
    }
    batch_size = 512
    epochs = 200
    anneal_epochs = 10
    patience = 10
    nhead = 4
    dim_feedforward = 256

    data = load_data(subtype=subtype, encoder=dna_one_hot, batch_size=batch_size)
    train_loader, test_loader = data()
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    seq_length = sample_x.shape[2]

    model = TransformerVAE(
        input_dim=input_dim,
        d_model=best_config["d_model"],
        latent_dim=best_config["latent_dim"],
        seq_length=seq_length,
        nhead=nhead,
        num_layers=best_config["num_layers"],
        dim_feedforward=dim_feedforward,
        dropout=best_config["dropout"],
        num_memory_tokens=best_config["num_memory_tokens"],
    ).to(DEVICE)
    criterion = VAE_Loss(beta=best_config["beta"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=best_config["lr"], weight_decay=1e-2)

    print(f"Training Transformer VAE for {subtype} on {DEVICE} ...")
    model, history = VAE_train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimiser,
        DEVICE,
        epochs=epochs,
        patience=patience,
        anneal_epochs=anneal_epochs,
        save_every=0,
        save_path=str(WEIGHTS_DIR / f"checkpoint_{weight_name}"),
    )

    return save_checkpoint(
        weight_name=weight_name,
        model=model,
        config={
            "input_dim": input_dim,
            "d_model": best_config["d_model"],
            "latent_dim": best_config["latent_dim"],
            "seq_length": seq_length,
            "nhead": nhead,
            "num_layers": best_config["num_layers"],
            "dim_feedforward": dim_feedforward,
            "dropout": best_config["dropout"],
            "num_memory_tokens": best_config["num_memory_tokens"],
        },
        history=history,
        extra={"best_config": best_config},
    )


def retrain_all_models() -> list[Path]:
    return [
        train_vae("H1N1_global", "vae_h1n1_weights.pt"),
        train_vae("H3N2_global", "vae_h3n2_weights.pt"),
        train_tvae("H1N1_global", "tvae_h1n1_weights.pt"),
        train_tvae("H3N2_global", "tvae_h3n2_weights.pt"),
    ]


def should_skip_cell(source: str) -> bool:
    """we strip upload/install cells so the orchestration stays non-interactive"""
    skip_markers = (
        "upload_file(",
        "upload_folder(",
        ".push_to_hub(",
        "from huggingface_hub import login",
        "login(",
        "!pip install",
        "%pip install",
    )
    return any(marker in source for marker in skip_markers)


def execute_notebook(job: NotebookJob) -> Path:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError(
            "Notebook execution needs `nbformat` and `nbclient`. Use the pinned `environment.yml` "
            "or install the updated requirements.txt first."
        ) from exc

    print(f"Executing notebook: {job.path.name}")
    with job.path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    cleaned_cells = []
    skipped_cells = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            cleaned_cells.append(cell)
            continue
        source = cell.source
        if should_skip_cell(source):
            skipped_cells += 1
            continue
        for old, new in job.replacements.items():
            source = source.replace(old, new)
        cell.source = source
        cleaned_cells.append(cell)
    notebook.cells = cleaned_cells

    # we execute from the project root so notebook imports like `from models...` still resolve
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    out_path = EXECUTED_NOTEBOOKS_DIR / job.path.name
    with out_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"  saved executed copy -> {out_path}")
    if skipped_cells:
        print(f"  skipped {skipped_cells} upload/install cells")
    return out_path


def build_notebook_plan(train: bool, data_clean: bool, tuning: bool, run_sirc_notebook: bool) -> list[NotebookJob]:
    plan: list[NotebookJob] = []

    # 1. sequence data pipeline (only when rebuilding from raw FASTA)
    if data_clean:
        plan.append(NotebookJob(path=NOTEBOOKS_DIR / "seq_data.ipynb"))

    # 2. CDC surveillance ingestion (only when rebuilding from raw data)
    if data_clean:
        plan.append(NotebookJob(path=NOTEBOOKS_DIR / "cdc_surveillance_pipeline.ipynb"))

    # 3-4. hyperparameter sweeps and analysis (only when rerunning Optuna)
    if tuning:
        plan.extend(
            [
                NotebookJob(path=NOTEBOOKS_DIR / "hyperparameter_VAE.ipynb"),
                NotebookJob(path=NOTEBOOKS_DIR / "hyperparameter_VAE_64.ipynb"),
                NotebookJob(path=NOTEBOOKS_DIR / "hyperparameter_TVAE.ipynb"),
                NotebookJob(path=NOTEBOOKS_DIR / "VAE_hp_analysis.ipynb"),
                NotebookJob(path=NOTEBOOKS_DIR / "TVAE_hp_analysis.ipynb"),
            ]
        )

    # 5-6. training notebooks (only when retraining)
    if train:
        plan.append(NotebookJob(path=NOTEBOOKS_DIR / "vae_training_notebook.ipynb"))
        plan.append(NotebookJob(path=NOTEBOOKS_DIR / "transformer_vae_training_notebook.ipynb"))

    # 7. SIRC calibration and evaluation (only when explicitly requested)
    if run_sirc_notebook:
        plan.append(NotebookJob(path=NOTEBOOKS_DIR / "sirs_sirc_model.ipynb"))

    if train:
        print("train=True -> model retraining will run before notebook execution")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible entry point for AML_final")
    parser.add_argument("--train", action="store_true", help="override the file-level TRAIN flag and retrain all models")
    parser.add_argument(
        "--data-clean",
        action="store_true",
        help="start from the raw CDC zip and raw sequence files instead of processed hosted artefacts",
    )
    parser.add_argument("--tuning", action="store_true", help="rerun the Optuna sweeps instead of only pulling hosted studies for plots")
    parser.add_argument("--sirc", action="store_true", help="run the SIRC calibration and evaluation notebook")
    parser.add_argument("--skip-download", action="store_true", help="skip HF prefetching and rely on existing caches")
    parser.add_argument("--skip-smoke-test", action="store_true", help="skip the lightweight forward-pass test")
    parser.add_argument("--smoke-test-only", action="store_true", help="run only the smoke test and then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = TRAIN or args.train
    data_clean = DATA_CLEAN or args.data_clean
    tuning = Tuning or args.tuning
    run_sirc_notebook = RUN_SIRC_NOTEBOOK or args.sirc
    run_download = DOWNLOAD_FROM_HF and not args.skip_download
    run_smoke = RUN_SMOKE_TEST and not args.skip_smoke_test

    ensure_dirs()
    set_seed()

    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")
    print(f"train={train}")
    print(f"data_clean={data_clean}")
    print(f"tuning={tuning}")

    if run_download:
        print("Prefetching public Hugging Face artefacts ...")
        prefetch_hf_assets(train=train, data_clean=data_clean, tuning=tuning)

    if run_smoke or args.smoke_test_only:
        print("Running lightweight smoke test ...")
        run_smoke_test()
        print("Smoke test passed")
        if args.smoke_test_only:
            return

    if train:
        print("Retraining all four models ...")
        saved_paths = retrain_all_models()
        for path in saved_paths:
            print(f"  saved -> {path}")

    notebook_plan = build_notebook_plan(
        train=train,
        data_clean=data_clean,
        tuning=tuning,
        run_sirc_notebook=run_sirc_notebook,
    )
    for job in notebook_plan:
        execute_notebook(job)

    print("run.py completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
