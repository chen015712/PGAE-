from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import PGAEConfig
from dataset import make_dataloaders
from loss import pgae_loss
from model import PGAE
from thermo_checks import evaluate_thermo_on_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def metric_to_float(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def average_metrics(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def build_model(config: PGAEConfig) -> PGAE:
    return PGAE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        nc=config.nc,
        encoder_dims=config.hidden_encoder,
        decoder_dims=config.hidden_decoder,
        dropout=config.residual_dropout,
        p_net_hidden=config.p_net_hidden,
    )


def run_epoch(model: PGAE, loader, config: PGAEConfig, optimizer=None, scaler=None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    all_metrics = []
    device = config.device
    amp_enabled = config.use_amp and device.type == "cuda"
    progress = tqdm(loader, desc="train" if is_train else "valid", leave=False)

    for batch in progress:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(batch["input"])
                loss, metrics = pgae_loss(batch, pred, config)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()

        metrics_float = metric_to_float(metrics)
        all_metrics.append(metrics_float)
        progress.set_postfix(loss=f"{metrics_float['loss']:.4e}", beta_mae=f"{metrics_float['beta_mae']:.3e}")

    return average_metrics(all_metrics)


@torch.no_grad()
def collect_predictions(model: PGAE, loader, config: PGAEConfig) -> Dict[str, np.ndarray]:
    model.eval()
    chunks = {"latent": [], "beta": [], "beta_true": [], "phase_label": [], "phase_error": [], "mass_residual": [], "thermo_residual": []}
    for batch in loader:
        batch = move_batch(batch, config.device)
        pred = model(batch["input"])
        beta = pred["beta"]
        x, y = pred["x"], pred["y"]
        x_true, y_true = batch["x"], batch["y"]
        z_hat = pred["z_flash"]
        chunks["latent"].append(pred["latent"].cpu())
        chunks["beta"].append(beta.cpu())
        chunks["beta_true"].append(batch["beta"].cpu())
        chunks["phase_label"].append(batch["phase_label"].cpu())
        chunks["phase_error"].append(torch.mean(torch.abs((y - x) - (y_true - x_true)), dim=-1).cpu())
        chunks["mass_residual"].append(torch.mean(torch.abs(z_hat - batch["z"]), dim=-1).cpu())
        chunks["thermo_residual"].append(torch.mean(torch.abs(torch.log((y + config.eps) / (x + config.eps)) - torch.log((y_true + config.eps) / (x_true + config.eps))), dim=-1).cpu())
    return {key: torch.cat(value, dim=0).numpy() for key, value in chunks.items()}


def save_history(history: list[Dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_outputs(preds: Dict[str, np.ndarray], config: PGAEConfig) -> None:
    latent = preds["latent"]
    beta = preds["beta"].reshape(-1)
    beta_true = preds["beta_true"].reshape(-1)
    phase_label = preds["phase_label"].reshape(-1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=beta_true, s=8, cmap="viridis", alpha=0.75)
    ax.set_xlabel("latent-1")
    ax.set_ylabel("latent-2")
    ax.set_zlabel("latent-3")
    fig.colorbar(scatter, ax=ax, label="WinProp beta")
    fig.tight_layout()
    fig.savefig(config.fig_training_dir / "latent_beta_3d.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=phase_label, s=8, cmap="coolwarm", alpha=0.75)
    ax.set_xlabel("latent-1")
    ax.set_ylabel("latent-2")
    ax.set_zlabel("latent-3")
    fig.colorbar(scatter, ax=ax, label="phase label")
    fig.tight_layout()
    fig.savefig(config.fig_training_dir / "latent_phase_3d.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(beta_true, beta, c=phase_label, s=8, cmap="coolwarm", alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("WinProp beta")
    ax.set_ylabel("PGAE beta")
    ax.set_title("Beta prediction parity")
    fig.tight_layout()
    fig.savefig(config.fig_training_dir / "beta_parity.png", dpi=220)
    plt.close(fig)

    for name, title in [
        ("phase_error", "Phase split reconstruction error"),
        ("mass_residual", "Mass conservation residual"),
        ("thermo_residual", "Thermodynamic consistency residual"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(preds[name].reshape(-1), bins=60, alpha=0.85)
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(config.fig_training_dir / f"{name}.png", dpi=220)
        plt.close(fig)


def save_checkpoint(path: Path, model: PGAE, optimizer, config: PGAEConfig, stats, epoch: int, metrics: Dict[str, float]) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict(),
        "normalization": stats.to_dict(),
        "epoch": epoch,
        "val_metrics": metrics,
    }, path)


def main(finetune: bool = False) -> None:
    config = PGAEConfig()
    set_seed(config.seed)
    train_loader, val_loader, _, stats = make_dataloaders(config)
    model = build_model(config).to(config.device)

    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0

    if finetune and config.best_checkpoint_path.exists():
        print(f"Fine-tuning from: {config.best_checkpoint_path}")
        ckpt = torch.load(config.best_checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        lr = config.learning_rate * 0.1  # lower LR for fine-tuning
        best_loss = float(ckpt.get("val_metrics", {}).get("loss", float("inf")))
        print(f"  Loaded checkpoint epoch={ckpt.get('epoch', '?')}, prev best_loss={best_loss:.5e}, finetune LR={lr:.1e}")
    else:
        lr = config.learning_rate

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.lr_patience)
    scaler = torch.amp.GradScaler(config.device.type, enabled=config.use_amp and config.device.type == "cuda")

    history = []
    n_epochs = config.epochs if not finetune else max(100, config.epochs // 3)
    print(f"Training PGAE on {config.device} | train batches={len(train_loader)} | val batches={len(val_loader)} | epochs={n_epochs}")

    for epoch in range(1, n_epochs + 1):
        actual_epoch = start_epoch + epoch
        train_metrics = run_epoch(model, train_loader, config, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, config)
        scheduler.step(val_metrics["loss"])

        row = {"epoch": actual_epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {actual_epoch:04d} | train_loss={train_metrics['loss']:.5e} | val_loss={val_metrics['loss']:.5e} | beta_mae={val_metrics['beta_mae']:.5e} | mass={val_metrics['mass_residual']:.5e} | mono={val_metrics.get('mono_loss', 0):.5e} | k_mono={val_metrics.get('k_mono_loss', 0):.5e}")

        save_checkpoint(config.last_checkpoint_path, model, optimizer, config, stats, actual_epoch, val_metrics)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = actual_epoch
            save_checkpoint(config.best_checkpoint_path, model, optimizer, config, stats, actual_epoch, val_metrics)

        if actual_epoch - best_epoch >= config.early_stop_patience:
            print(f"Early stopping at epoch {actual_epoch}; best epoch={best_epoch}, best val loss={best_loss:.5e}")
            break

    save_history(history, config.history_path)
    checkpoint = torch.load(config.best_checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    preds = collect_predictions(model, val_loader, config)
    plot_outputs(preds, config)

    print("\n--- Thermodynamic Consistency Checks (Validation Set) ---")
    thermo_results = evaluate_thermo_on_loader(model, val_loader, config, figure_dir=config.fig_thermo_dir)

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "beta_true_mean": float(preds["beta_true"].mean()),
        "beta_pred_mean": float(preds["beta"].mean()),
        "beta_mae": float(np.mean(np.abs(preds["beta"] - preds["beta_true"]))),
        "phase_split_reconstruction_error": float(preds["phase_error"].mean()),
        "thermodynamic_consistency_residual": float(preds["thermo_residual"].mean()),
        "mass_conservation_residual": float(preds["mass_residual"].mean()),
        "rachford_rice_residual_mean": thermo_results["rr_residual_mean"],
        "rachford_rice_pass_rate": thermo_results["rr_pass_rate_1e4"],
        "gibbs_dg_mean": thermo_results["gibbs_dg_mean"],
        "gibbs_violation_rate": thermo_results["gibbs_violation_rate"],
        "k_value_mae": thermo_results["k_mae"],
        "k_value_r2": thermo_results["k_r2"],
    }
    with open(config.validation_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBest model saved:", config.best_checkpoint_path)
    print("Training history:", config.history_path)
    print("Validation metrics:", config.validation_metrics_path)
    print("Figures:", config.fig_training_dir)
    print("Beta prediction statistics:")
    print(f"  true mean={summary['beta_true_mean']:.6f}, pred mean={summary['beta_pred_mean']:.6f}, MAE={summary['beta_mae']:.6e}")
    print(f"Phase split reconstruction error: {summary['phase_split_reconstruction_error']:.6e}")
    print(f"Thermodynamic consistency residual: {summary['thermodynamic_consistency_residual']:.6e}")
    print(f"Mass conservation residual: {summary['mass_conservation_residual']:.6e}")
    print(f"Rachford-Rice residual mean: {summary['rachford_rice_residual_mean']:.4e}  pass rate: {summary['rachford_rice_pass_rate']*100:.1f}%")
    print(f"Gibbs ΔG/RT mean: {summary['gibbs_dg_mean']:.6f}  violation rate: {summary['gibbs_violation_rate']*100:.2f}%")
    print(f"K-value MAE: {summary['k_value_mae']:.4e}  R²: {summary['k_value_r2']:.6f}")


if __name__ == "__main__":
    import sys
    finetune = "--finetune" in sys.argv
    main(finetune=finetune)
