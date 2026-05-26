from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from config import PGAEConfig


@dataclass
class NormalizationStats:
    pt_mean: torch.Tensor
    pt_std: torch.Tensor

    def to_dict(self) -> Dict[str, list]:
        return {
            "pt_mean": self.pt_mean.cpu().tolist(),
            "pt_std": self.pt_std.cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "NormalizationStats":
        return cls(
            pt_mean=torch.tensor(data["pt_mean"], dtype=torch.float32),
            pt_std=torch.tensor(data["pt_std"], dtype=torch.float32),
        )


class PGAEDataset(Dataset):
    def __init__(self, csv_path: str, stats: NormalizationStats | None = None, nc: int = 15):
        self.frame = pd.read_csv(csv_path)
        self.nc = nc
        z_cols = [f"z{i}" for i in range(1, nc + 1)]
        x_cols = [f"x{i}" for i in range(1, nc + 1)]
        y_cols = [f"y{i}" for i in range(1, nc + 1)]
        beta_col = "beta_V" if "beta_V" in self.frame.columns else "beta"

        required = ["P", "T", *z_cols, beta_col, *x_cols, *y_cols]
        missing = [col for col in required if col not in self.frame.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        pt = self.frame[["P", "T"]].to_numpy(dtype=np.float32)
        z = self._normalize_simplex(self.frame[z_cols].to_numpy(dtype=np.float32))
        beta = np.clip(self.frame[[beta_col]].to_numpy(dtype=np.float32), 0.0, 1.0)
        x = self._normalize_simplex(self.frame[x_cols].to_numpy(dtype=np.float32))
        y = self._normalize_simplex(self.frame[y_cols].to_numpy(dtype=np.float32))

        if "phase_label" in self.frame.columns:
            phase = self.frame["phase_label"].to_numpy(dtype=np.int64)
        else:
            # Encoding: 0=Liquid, 1=Vapour, 2=Two-Phase (matching data pipeline)
            phase = np.where(beta.reshape(-1) <= 1.0e-6, 0,
                    np.where(beta.reshape(-1) >= 1.0 - 1.0e-6, 1, 2)).astype(np.int64)

        self.pt_raw = torch.tensor(pt, dtype=torch.float32)
        self.z = torch.tensor(z, dtype=torch.float32)
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.phase_label = torch.tensor(phase.reshape(-1), dtype=torch.long)
        self.stats = stats

    @staticmethod
    def _normalize_simplex(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, 0.0, None)
        denom = np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)
        return values / denom

    def set_stats(self, stats: NormalizationStats) -> None:
        self.stats = stats

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.stats is None:
            raise RuntimeError("Normalization stats must be set before indexing the dataset.")
        pt_norm = (self.pt_raw[index] - self.stats.pt_mean) / self.stats.pt_std
        features = torch.cat([pt_norm, self.z[index]], dim=0)
        target = torch.cat([self.beta[index], self.x[index], self.y[index]], dim=0)
        return {
            "input": features,
            "pt_raw": self.pt_raw[index],
            "z": self.z[index],
            "beta": self.beta[index],
            "x": self.x[index],
            "y": self.y[index],
            "target": target,
            "phase_label": self.phase_label[index],
        }


def split_indices(n_items: int, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator)
    n_val = max(1, int(round(n_items * val_fraction)))
    n_val = min(n_val, n_items - 1)
    return indices[n_val:], indices[:n_val]


def make_dataloaders(config: PGAEConfig):
    dataset = PGAEDataset(str(config.dataset_path), nc=config.nc)
    train_idx, val_idx = split_indices(len(dataset), config.val_fraction, config.seed)
    pt_train = dataset.pt_raw[train_idx]
    stats = NormalizationStats(
        pt_mean=pt_train.mean(dim=0),
        pt_std=pt_train.std(dim=0, unbiased=False).clamp_min(1.0e-6),
    )
    dataset.set_stats(stats)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_idx) > config.batch_size,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, dataset, stats
