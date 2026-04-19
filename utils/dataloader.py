import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_dataset_builder, DatasetDict, Features, Value
from .encoders import dna_one_hot


def _collate_mixed(batch: list) -> tuple:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    years = torch.tensor([b["year"] for b in batch], dtype=torch.long)
    ids = [b["id"] for b in batch]
    months = [b.get("month", -1) for b in batch]
    seasons = [b.get("season", "") for b in batch]
    return input_ids, (years, ids, months, seasons)


class load_data:
    """we wrap the HF dataset once here so the notebooks don't keep reimplementing it"""

    def __init__(self, subtype: str, encoder=dna_one_hot, batch_size=64):
        self.subtype = subtype
        self.encoder = encoder
        self.batch_size = batch_size
        builder = load_dataset_builder("sidms/AML", subtype)
        available = set(builder.info.features.keys())
        wanted = [c for c in ["id", "sequence", "year", "month", "season"] if c in available]
        forced_features_map = {
            "id": Value("string"),
            "sequence": Value("string"),
            "year": Value("int64"),
            "month": Value("int64"),
            "season": Value("string"),
        }
        forced_features = Features({c: forced_features_map[c] for c in wanted})
        raw = load_dataset(
            "sidms/AML",
            subtype,
            columns=wanted,
            features=forced_features,
            verification_mode="no_checks"
        )
        splits = {"train": raw["train"]}
        if "test" in raw:
            splits["test"] = raw["test"]
        self.dataset = DatasetDict(splits)
        self.encoder = self.encoder()

    def _encode(self):
        remove_cols = [c for c in ["sequence"] if c in self.dataset["train"].column_names]
        self.dataset = self.dataset.map(
            self.encoder,
            batched=True,
            num_proc=4,
            remove_columns=remove_cols,
        )
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "year"],
            output_all_columns=True,
        )

    def make_loaders(self):
        worker_kwargs = dict(
            collate_fn=_collate_mixed,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )
        train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True, **worker_kwargs)
        test_loader = (
            DataLoader(self.dataset["test"], batch_size=self.batch_size, shuffle=False, **worker_kwargs)
            if "test" in self.dataset else None
        )
        return train_loader, test_loader

    def __call__(self):
        self._encode()
        return self.make_loaders()
