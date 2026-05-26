from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

from config import PGAEConfig
from dataset import NormalizationStats
from thermo_checks import run_thermo_checks
from train import build_model


class PGAEFlashSurrogate:
    def __init__(self, checkpoint_path: str | Path | None = None):
        self.config = PGAEConfig()
        path = Path(checkpoint_path) if checkpoint_path is not None else self.config.best_checkpoint_path
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.stats = NormalizationStats.from_dict(checkpoint["normalization"])
        self.model = build_model(self.config).to(self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _prepare_input(self, P: float, T: float, z: Iterable[float]) -> torch.Tensor:
        z_tensor = torch.tensor(list(z), dtype=torch.float32)
        if z_tensor.numel() != self.config.nc:
            raise ValueError(f"Expected {self.config.nc} composition entries, got {z_tensor.numel()}.")
        z_tensor = z_tensor.clamp_min(0.0)
        z_tensor = z_tensor / z_tensor.sum().clamp_min(1.0e-12)
        pt = torch.tensor([P, T], dtype=torch.float32)
        pt_norm = (pt - self.stats.pt_mean) / self.stats.pt_std
        return torch.cat([pt_norm, z_tensor], dim=0).unsqueeze(0).to(self.config.device)

    @torch.no_grad()
    def predict_flash(self, P: float, T: float, z: Iterable[float]) -> Dict[str, np.ndarray | float]:
        inputs = self._prepare_input(P, T, z)
        pred = self.model(inputs)
        beta = pred["beta"][0, 0].cpu().item()
        x = pred["x"][0].cpu().numpy()
        y = pred["y"][0].cpu().numpy()
        latent = pred["latent"][0].cpu().numpy()
        z_np = np.asarray(list(z), dtype=np.float32)
        z_np = np.clip(z_np, 0.0, None)
        z_np = z_np / max(float(z_np.sum()), 1.0e-12)
        z_hat = (1.0 - beta) * x + beta * y
        return {
            "beta": beta,
            "x": x,
            "y": y,
            "latent": latent,
            "z_reconstructed": z_hat,
            "mass_residual": float(np.mean(np.abs(z_hat - z_np))),
            "x_sum": float(x.sum()),
            "y_sum": float(y.sum()),
        }


def main() -> None:
    config = PGAEConfig()
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)
    df = pd.read_csv(config.dataset_path)
    z_cols = [f"z{i}" for i in range(1, config.nc + 1)]
    x_cols = [f"x{i}" for i in range(1, config.nc + 1)]
    y_cols = [f"y{i}" for i in range(1, config.nc + 1)]

    rows = []
    for _, row in df.iterrows():
        z = row[z_cols].to_numpy(dtype=np.float32)
        pred = surrogate.predict_flash(float(row["P"]), float(row["T"]), z)
        item = {
            "P": row["P"],
            "T": row["T"],
            "beta_pred": pred["beta"],
            "mass_residual": pred["mass_residual"],
            "x_sum": pred["x_sum"],
            "y_sum": pred["y_sum"],
            **{f"z{i + 1}": float(z[i]) for i in range(config.nc)},
            **{f"x_pred{i + 1}": pred["x"][i] for i in range(config.nc)},
            **{f"y_pred{i + 1}": pred["y"][i] for i in range(config.nc)},
            **{f"z_reconstructed{i + 1}": pred["z_reconstructed"][i] for i in range(config.nc)},
            **{f"latent{i + 1}": pred["latent"][i] for i in range(config.latent_dim)},
        }
        if "beta_V" in df.columns:
            item["beta_true"] = row["beta_V"]
            item["beta_abs_error"] = abs(item["beta_pred"] - item["beta_true"])
        if all(col in df.columns for col in x_cols + y_cols):
            x_true = row[x_cols].to_numpy(dtype=np.float32)
            y_true = row[y_cols].to_numpy(dtype=np.float32)
            item["phase_split_reconstruction_error"] = float(np.mean(np.abs((pred["y"] - pred["x"]) - (y_true - x_true))))
            item["thermodynamic_consistency_residual"] = float(np.mean(np.abs(np.log((pred["y"] + config.eps) / (pred["x"] + config.eps)) - np.log((y_true + config.eps) / (x_true + config.eps)))))
            for i in range(config.nc):
                item[f"x{i + 1}"] = float(x_true[i])
                item[f"y{i + 1}"] = float(y_true[i])
        rows.append(item)

    out_path = config.inference_dir / "pgae_predictions.csv"
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")
    print(f"Mean mass residual: {pred_df['mass_residual'].mean():.6e}")
    if "beta_abs_error" in pred_df.columns:
        print(f"Beta MAE: {pred_df['beta_abs_error'].mean():.6e}")
        print(f"Beta true mean={pred_df['beta_true'].mean():.6f}, pred mean={pred_df['beta_pred'].mean():.6f}")
    if "phase_split_reconstruction_error" in pred_df.columns:
        print(f"Phase split reconstruction error: {pred_df['phase_split_reconstruction_error'].mean():.6e}")
        print(f"Thermodynamic consistency residual: {pred_df['thermodynamic_consistency_residual'].mean():.6e}")

    # ---- Full thermodynamic consistency checks ----
    print("\n--- Thermodynamic Consistency Checks (Full Dataset) ---")
    z_cols_list = [f"z{i}" for i in range(1, config.nc + 1)]
    x_cols_list = [f"x{i}" for i in range(1, config.nc + 1)]
    y_cols_list = [f"y{i}" for i in range(1, config.nc + 1)]
    xp_cols = [f"x_pred{i}" for i in range(1, config.nc + 1)]
    yp_cols = [f"y_pred{i}" for i in range(1, config.nc + 1)]

    has_truth = all(col in pred_df.columns for col in x_cols_list + y_cols_list)
    P_vals = pred_df["P"].to_numpy(dtype=np.float64)
    T_vals = pred_df["T"].to_numpy(dtype=np.float64)
    z_vals = pred_df[z_cols_list].to_numpy(dtype=np.float64)
    beta_vals = pred_df["beta_pred"].to_numpy(dtype=np.float64)
    xp_vals = pred_df[xp_cols].to_numpy(dtype=np.float64)
    yp_vals = pred_df[yp_cols].to_numpy(dtype=np.float64)

    if has_truth:
        xt_vals = pred_df[x_cols_list].to_numpy(dtype=np.float64)
        yt_vals = pred_df[y_cols_list].to_numpy(dtype=np.float64)
    else:
        xt_vals = None
        yt_vals = None

    thermo_report = run_thermo_checks(
        P=P_vals, T=T_vals, z=z_vals,
        beta_pred=beta_vals, x_pred=xp_vals, y_pred=yp_vals,
        x_true=xt_vals, y_true=yt_vals,
        figure_dir=config.figure_dir,
    )
    print(thermo_report.summary())


if __name__ == "__main__":
    main()
