"""Part 12: Latent Manifold Deep Analysis & Visualization.

Analyzes the PGAE 3D latent space:
  1. PCA / UMAP dimensionality reduction & visualization
  2. Latent–physical correlation analysis
  3. Phase clustering metrics
  4. Latent space interpolation & traversal
  5. Dimension-wise sensitivity analysis
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import PGAEConfig
from infer import PGAEFlashSurrogate
from dataset import NormalizationStats

COMP_NAMES = ["C10+", "N2", "CO2", "CH4", "C2H6", "C3H8", "IC4", "NC4", "IC5", "NC5",
              "FC6", "FC7", "FC8", "FC9", "FC10"]
NC = 15


# =============================================================================
# 1. Latent vector collection
# =============================================================================

@torch.no_grad()
def collect_latent_vectors(
    surrogate: PGAEFlashSurrogate,
    df: pd.DataFrame,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Encode the entire dataset and return latent vectors with metadata.

    Returns:
        latent: (N, latent_dim) latent codes
        P, T: (N,) physical conditions
        z: (N, NC) feed compositions
        beta: (N,) model-predicted β
        beta_true: (N,) ground-truth β
        phase_label: (N,) phase labels (0=L, 1=V, 2=TP)
        log_K: (N, NC) model-predicted log K-values
    """
    device = surrogate.config.device
    model = surrogate.model
    model.eval()
    stats = surrogate.stats
    config = surrogate.config

    z_cols = [f"z{i}" for i in range(1, NC + 1)]
    P_vals = df["P"].to_numpy(dtype=np.float64)
    T_vals = df["T"].to_numpy(dtype=np.float64)
    z_arr = df[z_cols].to_numpy(dtype=np.float64)
    beta_true_arr = df["beta_V"].to_numpy(dtype=np.float64)
    phase_label_arr = df["phase_label"].to_numpy(dtype=np.int64)

    N = len(df)
    latent_dim = config.latent_dim
    latents = np.zeros((N, latent_dim), dtype=np.float32)
    betas = np.zeros(N, dtype=np.float32)
    log_Ks = np.zeros((N, NC), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="Encoding dataset"):
        end = min(start + batch_size, N)
        batch_P = P_vals[start:end]
        batch_T = T_vals[start:end]
        batch_z = z_arr[start:end]

        pts = np.stack([batch_P, batch_T], axis=1)
        pts = torch.tensor(pts, dtype=torch.float32)
        pts_norm = (pts - stats.pt_mean) / stats.pt_std
        z_t = torch.tensor(batch_z, dtype=torch.float32).clamp_min(0)
        z_t = z_t / z_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
        inputs = torch.cat([pts_norm, z_t], dim=1).to(device)

        pred = model(inputs)
        latents[start:end] = pred["latent"].cpu().numpy()
        betas[start:end] = pred["beta"].squeeze(-1).cpu().numpy()
        log_Ks[start:end] = pred["log_K"].cpu().numpy()

    return {
        "latent": latents,
        "P": P_vals,
        "T": T_vals,
        "z": z_arr,
        "beta": betas,
        "beta_true": beta_true_arr,
        "phase_label": phase_label_arr,
        "log_K": log_Ks,
    }


# =============================================================================
# 2. PCA & UMAP dimensionality reduction
# =============================================================================

def run_pca_analysis(latent: np.ndarray) -> Dict:
    """PCA on latent space: explained variance, component loadings."""
    pca = PCA(n_components=min(latent.shape[1], 3))
    latent_pca = pca.fit_transform(latent)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "components": pca.components_.tolist(),
        "latent_pca": latent_pca,
    }


def run_umap_projection(latent: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    """UMAP 2D projection of latent space."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=42, verbose=False)
        return reducer.fit_transform(latent)
    except ImportError:
        print("  umap-learn not installed; using PCA 2D instead")
        pca = PCA(n_components=2)
        return pca.fit_transform(latent)


# =============================================================================
# 3. Physical correlation analysis
# =============================================================================

def compute_latent_correlations(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute Pearson correlation between each latent dim and physical quantities.

    Returns DataFrame with rows=latent dims, cols=physical features.
    """
    latent = data["latent"]
    latent_dim = latent.shape[1]

    # Physical features to correlate with
    phys_features = {
        "P": data["P"],
        "T": data["T"],
        "beta_true": data["beta_true"],
        "beta_pred": data["beta"],
        **{f"z_{COMP_NAMES[i]}": data["z"][:, i] for i in range(NC)},
    }

    rows = []
    for d in range(latent_dim):
        row = {"latent_dim": f"L{d+1}"}
        for name, values in phys_features.items():
            corr = np.corrcoef(latent[:, d], values)[0, 1]
            row[name] = corr
        rows.append(row)

    corr_df = pd.DataFrame(rows).set_index("latent_dim")
    return corr_df


def find_most_correlated(corr_df: pd.DataFrame, n_top: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """For each latent dimension, find the n_top most correlated physical features."""
    result = {}
    for dim_name in corr_df.index:
        sorted_corr = corr_df.loc[dim_name].abs().sort_values(ascending=False)
        top = [(name, corr_df.loc[dim_name, name]) for name in sorted_corr.index[:n_top]]
        result[dim_name] = top
    return result


# =============================================================================
# 4. Phase clustering metrics
# =============================================================================

def evaluate_latent_clustering(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Evaluate how well the latent space clusters by phase label."""
    latent = data["latent"]
    phase = data["phase_label"]

    # Use subset for silhouette (expensive O(N²))
    n_subset = min(2000, len(latent))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(latent), n_subset, replace=False)

    metrics = {}
    try:
        metrics["silhouette_score"] = float(silhouette_score(latent[idx], phase[idx]))
    except Exception:
        metrics["silhouette_score"] = float("nan")

    # Intra-class compactness ratio
    scaler = StandardScaler().fit(latent)
    latent_scaled = scaler.transform(latent)

    for label, name in [(0, "liquid"), (1, "vapour"), (2, "two_phase")]:
        mask = phase == label
        if mask.sum() > 10:
            points = latent_scaled[mask]
            centroid = points.mean(axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            metrics[f"{name}_n"] = int(mask.sum())
            metrics[f"{name}_mean_dist_to_centroid"] = float(dists.mean())
            metrics[f"{name}_std_dist_to_centroid"] = float(dists.std())

    # Inter-class separation (distance between phase centroids)
    centroids = {}
    for label in [0, 1, 2]:
        mask = phase == label
        if mask.sum() > 0:
            centroids[label] = latent[mask].mean(axis=0)

    for (l1, n1), (l2, n2) in [((0, "L"), (1, "V")), ((0, "L"), (2, "TP")), ((1, "V"), (2, "TP"))]:
        if l1 in centroids and l2 in centroids:
            metrics[f"centroid_dist_{n1}_{n2}"] = float(
                np.linalg.norm(centroids[l1] - centroids[l2])
            )

    return metrics


# =============================================================================
# 5. Latent space interpolation
# =============================================================================

@torch.no_grad()
def latent_interpolation(
    surrogate: PGAEFlashSurrogate,
    z_fixed: np.ndarray,
    latent_start: np.ndarray,
    latent_end: np.ndarray,
    n_steps: int = 50,
    P_kPa: float = 20000.0,
) -> Dict[str, np.ndarray]:
    """Linearly interpolate between two latent points and decode.

    Args:
        surrogate: loaded model
        z_fixed: fixed feed composition for decoding
        latent_start, latent_end: start and end points in latent space
        n_steps: number of interpolation steps

    Returns:
        dict with beta, x, y, K at each interpolation step
    """
    model = surrogate.model
    model.eval()
    device = surrogate.config.device

    # Normalise P for decoder P-modulation
    P_norm_val = (P_kPa - surrogate.stats.pt_mean[0].item()) / surrogate.stats.pt_std[0].item()
    P_norm = torch.tensor([[P_norm_val]], dtype=torch.float32).to(device)

    alphas = np.linspace(0, 1, n_steps)
    latent_start_t = torch.tensor(latent_start, dtype=torch.float32).reshape(1, -1)
    latent_end_t = torch.tensor(latent_end, dtype=torch.float32).reshape(1, -1)
    z_t = torch.tensor(z_fixed, dtype=torch.float32).reshape(1, -1).to(device)

    betas = np.zeros(n_steps)
    xs = np.zeros((n_steps, NC))
    ys = np.zeros((n_steps, NC))
    Ks = np.zeros((n_steps, NC))
    lats = np.zeros((n_steps, latent_start.shape[0]))

    for i, alpha in enumerate(alphas):
        latent_i = (1 - alpha) * latent_start_t + alpha * latent_end_t
        latent_i = latent_i.to(device)
        decoded = model.decode(latent_i, z_t, P_norm)
        betas[i] = decoded["beta"][0, 0].cpu().item()
        xs[i] = decoded["x"][0].cpu().numpy()
        ys[i] = decoded["y"][0].cpu().numpy()
        Ks[i] = decoded["K"][0].cpu().numpy()
        lats[i] = latent_i[0].cpu().numpy()

    return {"alpha": alphas, "beta": betas, "x": xs, "y": ys, "K": Ks, "latent": lats}


# =============================================================================
# 6. Dimension-wise sensitivity analysis
# =============================================================================

@torch.no_grad()
def latent_sensitivity(
    surrogate: PGAEFlashSurrogate,
    z_fixed: np.ndarray,
    base_latent: np.ndarray,
    dim: int,
    delta_range: float = 3.0,
    n_points: int = 51,
    P_kPa: float = 20000.0,
) -> Dict[str, np.ndarray]:
    """Vary one latent dimension while fixing others, decode output.

    Reveals what physical quantity each latent dimension controls.
    """
    model = surrogate.model
    model.eval()
    device = surrogate.config.device

    # Normalise P for decoder P-modulation
    P_norm_val = (P_kPa - surrogate.stats.pt_mean[0].item()) / surrogate.stats.pt_std[0].item()
    P_norm = torch.tensor([[P_norm_val]], dtype=torch.float32).to(device)

    deltas = np.linspace(-delta_range, delta_range, n_points)
    z_t = torch.tensor(z_fixed, dtype=torch.float32).reshape(1, -1).to(device)

    betas = np.zeros(n_points)
    xs = np.zeros((n_points, NC))
    ys = np.zeros((n_points, NC))
    Ks = np.zeros((n_points, NC))

    for i, delta in enumerate(deltas):
        latent_i = base_latent.copy()
        latent_i[dim] += delta
        latent_t = torch.tensor(latent_i, dtype=torch.float32).reshape(1, -1).to(device)
        decoded = model.decode(latent_t, z_t, P_norm)
        betas[i] = decoded["beta"][0, 0].cpu().item()
        xs[i] = decoded["x"][0].cpu().numpy()
        ys[i] = decoded["y"][0].cpu().numpy()
        Ks[i] = decoded["K"][0].cpu().numpy()

    return {"delta": deltas, "beta": betas, "x": xs, "y": ys, "K": Ks}


# =============================================================================
# 7. Publication-quality plots
# =============================================================================

def plot_pca_3d(data: Dict[str, np.ndarray], output_dir: Path) -> None:
    """3D PCA scatter colored by phase label and beta."""
    latent = data["latent"]
    pca_results = run_pca_analysis(latent)
    latent_pca = pca_results["latent_pca"]

    fig = plt.figure(figsize=(14, 5.5))

    # PCA colored by phase
    ax1 = fig.add_subplot(121, projection="3d")
    phase = data["phase_label"]
    colors = {0: "blue", 1: "red", 2: "green"}
    labels = {0: "Liquid", 1: "Vapour", 2: "Two-Phase"}
    for ph in [0, 1, 2]:
        mask = phase == ph
        if mask.sum() > 0:
            ax1.scatter(latent_pca[mask, 0], latent_pca[mask, 1], latent_pca[mask, 2],
                       c=colors[ph], label=labels[ph], s=3, alpha=0.5)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("PCA: Colored by Phase Label")
    ax1.legend(markerscale=5, fontsize=8)

    # PCA colored by beta_true
    ax2 = fig.add_subplot(122, projection="3d")
    sc = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2],
                    c=data["beta_true"], s=3, cmap="viridis", alpha=0.5)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("PCA: Colored by β_true")
    fig.colorbar(sc, ax=ax2, label="β")

    fig.suptitle("PCA of PGAE Latent Space (3D)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_pca_3d.png", dpi=220)
    plt.close(fig)


def plot_umap_2d(data: Dict[str, np.ndarray], output_dir: Path) -> None:
    """2D UMAP projection with multiple colorings."""
    latent = data["latent"]
    print("  Running UMAP on %d points..." % len(latent))
    umap_2d = run_umap_projection(latent)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # By phase
    ax = axes[0, 0]
    phase = data["phase_label"]
    colors = {0: "blue", 1: "red", 2: "green"}
    labels = {0: "Liquid", 1: "Vapour", 2: "Two-Phase"}
    for ph in [0, 1, 2]:
        mask = phase == ph
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                      c=colors[ph], label=labels[ph], s=2, alpha=0.5)
    ax.set_title("UMAP: Phase Label")
    ax.legend(markerscale=6, fontsize=8)

    # By beta
    ax = axes[0, 1]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=data["beta_true"], s=2, cmap="viridis", alpha=0.5)
    ax.set_title("UMAP: β_true")
    fig.colorbar(sc, ax=ax, label="β")

    # By temperature
    ax = axes[1, 0]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=data["T"], s=2, cmap="plasma", alpha=0.5)
    ax.set_title("UMAP: Temperature (°C)")
    fig.colorbar(sc, ax=ax, label="T (°C)")

    # By pressure
    ax = axes[1, 1]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=data["P"], s=2, cmap="inferno", alpha=0.5,
                   norm=plt.matplotlib.colors.LogNorm())
    ax.set_title("UMAP: Pressure (kPa)")
    fig.colorbar(sc, ax=ax, label="P (kPa)")

    fig.suptitle("UMAP Projection of PGAE Latent Space", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_umap_2d.png", dpi=220)
    plt.close(fig)


def plot_latent_correlation_heatmap(corr_df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of latent dimension vs physical feature correlations."""
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=10)
    ax.set_title("Latent–Physical Feature Pearson Correlation", fontsize=13, fontweight="bold")

    # Annotate
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_correlation_heatmap.png", dpi=220)
    plt.close(fig)


def plot_phase_clustering(data: Dict[str, np.ndarray], cluster_metrics: Dict[str, float],
                          output_dir: Path) -> None:
    """Bar chart of inter-class centroid distances and intra-class compactness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Intra-class compactness
    classes = ["liquid", "vapour", "two_phase"]
    means = [cluster_metrics.get(f"{c}_mean_dist_to_centroid", 0) for c in classes]
    stds = [cluster_metrics.get(f"{c}_std_dist_to_centroid", 0) for c in classes]
    ns = [cluster_metrics.get(f"{c}_n", 0) for c in classes]
    bars = ax1.bar(classes, means, yerr=stds, capsize=5, color=["blue", "red", "green"], alpha=0.7)
    for bar, n in zip(bars, ns):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", fontsize=9)
    ax1.set_ylabel("Mean Distance to Centroid (σ-normalized)")
    ax1.set_title("Intra-Class Compactness")

    # Inter-class separation
    pairs = [("L_V", "Liquid–Vapour"), ("L_TP", "Liquid–TwoPhase"), ("V_TP", "Vapour–TwoPhase")]
    dists = [cluster_metrics.get(f"centroid_dist_{p[0]}", 0) for p in pairs]
    ax2.bar([p[1] for p in pairs], dists, color=["purple", "orange", "cyan"], alpha=0.7)
    ax2.set_ylabel("Centroid Distance")
    ax2.set_title("Inter-Class Separation")

    fig.suptitle("Latent Space Phase Clustering", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_phase_clustering.png", dpi=220)
    plt.close(fig)


def plot_interpolation(interp_data: Dict[str, np.ndarray],
                       z_fixed: np.ndarray, output_dir: Path) -> None:
    """Plot interpolation results: β, key K-values vs alpha."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    alpha = interp_data["alpha"]
    ax1.plot(alpha, interp_data["beta"], "b-", linewidth=2)
    ax1.set_xlabel("Interpolation α")
    ax1.set_ylabel("β")
    ax1.set_title("β Along Latent Interpolation")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)

    # Key component K-values
    key_idx = [3, 4, 2, 0]  # CH4, C2H6, CO2, C10+
    key_names = [COMP_NAMES[i] for i in key_idx]
    for idx, name in zip(key_idx, key_names):
        ax2.plot(alpha, interp_data["K"][:, idx], linewidth=1.5, label=name, alpha=0.8)
    ax2.set_xlabel("Interpolation α")
    ax2.set_ylabel("K-value")
    ax2.set_title("K-Values Along Latent Interpolation")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Latent Space Interpolation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_interpolation.png", dpi=220)
    plt.close(fig)


def plot_sensitivity(sensitivity_results: List[Dict], z_fixed: np.ndarray,
                     output_dir: Path) -> None:
    """Plot latent dimension sensitivity: β vs delta for each dimension."""
    latent_dim = len(sensitivity_results)
    fig, axes = plt.subplots(1, latent_dim, figsize=(5 * latent_dim, 4.5))
    if latent_dim == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        res = sensitivity_results[dim]
        ax.plot(res["delta"], res["beta"], "b-", linewidth=2)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(f"Δ L{dim+1}")
        ax.set_ylabel("β")
        ax.set_title(f"Sensitivity: Latent Dim {dim+1}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latent Dimension Sensitivity Analysis (β response)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_sensitivity_beta.png", dpi=220)
    plt.close(fig)

    # K-value sensitivity per latent dim
    fig, axes = plt.subplots(1, latent_dim, figsize=(5 * latent_dim, 4.5))
    if latent_dim == 1:
        axes = [axes]
    key_idx = [3, 4, 2, 0]  # CH4, C2H6, CO2, C10+
    key_names = [COMP_NAMES[i] for i in key_idx]

    for dim, ax in enumerate(axes):
        res = sensitivity_results[dim]
        for idx, name in zip(key_idx, key_names):
            ax.plot(res["delta"], res["K"][:, idx], linewidth=1.5, label=name, alpha=0.8)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(f"Δ L{dim+1}")
        ax.set_ylabel("K-value")
        ax.set_title(f"K Sensitivity: Latent Dim {dim+1}")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latent Dimension Sensitivity: K-Value Response", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_sensitivity_K.png", dpi=220)
    plt.close(fig)


# =============================================================================
# 8. Main analysis
# =============================================================================

def main() -> None:
    config = PGAEConfig()
    if not config.best_checkpoint_path.exists():
        print(f"Checkpoint not found: {config.best_checkpoint_path}")
        return

    print("Loading model...")
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)

    print("Loading dataset...")
    df = pd.read_csv(config.dataset_path)
    print(f"  {len(df)} samples")

    # ---- Collect latent vectors ----
    print("\n" + "=" * 60)
    print("1. Collecting latent vectors for full dataset...")
    print("=" * 60)
    data = collect_latent_vectors(surrogate, df)
    latent = data["latent"]
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
    print(f"  Latent mean: {latent.mean(axis=0)}")
    print(f"  Latent std: {latent.std(axis=0)}")

    # ---- PCA Analysis ----
    print("\n" + "=" * 60)
    print("2. PCA Analysis...")
    print("=" * 60)
    pca_results = run_pca_analysis(latent)
    print(f"  Explained variance ratio: {pca_results['explained_variance_ratio']}")
    print(f"  Cumulative: {np.cumsum(pca_results['explained_variance_ratio'])}")
    print(f"  PC components:")
    for i, comp in enumerate(pca_results["components"]):
        print(f"    PC{i+1}: {np.array2string(np.array(comp), precision=3)}")

    # ---- UMAP Projection ----
    print("\n" + "=" * 60)
    print("3. UMAP 2D Projection...")
    print("=" * 60)

    # ---- Correlation Analysis ----
    print("\n" + "=" * 60)
    print("4. Latent–Physical Correlation Analysis...")
    print("=" * 60)
    corr_df = compute_latent_correlations(data)
    print(corr_df.to_string())
    top_corrs = find_most_correlated(corr_df, n_top=5)
    print("\n  Top correlations per latent dimension:")
    for dim, tops in top_corrs.items():
        print(f"    {dim}: {', '.join(f'{name}({corr:+.3f})' for name, corr in tops)}")

    # ---- Phase Clustering ----
    print("\n" + "=" * 60)
    print("5. Phase Clustering Metrics...")
    print("=" * 60)
    cluster_metrics = evaluate_latent_clustering(data)
    for k, v in cluster_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ---- Latent Interpolation ----
    print("\n" + "=" * 60)
    print("6. Latent Space Interpolation...")
    print("=" * 60)
    # Pick two samples: one liquid (β≈0), one vapour (β≈1)
    liq_mask = data["beta_true"] < 0.01
    vap_mask = data["beta_true"] > 0.99
    if liq_mask.sum() > 0 and vap_mask.sum() > 0:
        liq_idx = np.where(liq_mask)[0][0]
        vap_idx = np.where(vap_mask)[0][0]
        z_mid = data["z"][liq_idx]  # use liquid sample's z
        interp_data = latent_interpolation(
            surrogate, z_mid,
            latent[liq_idx], latent[vap_idx],
            n_steps=80,
        )
        print(f"  Liquid (idx={liq_idx}) → Vapour (idx={vap_idx})")
        print(f"  β range along interpolation: [{interp_data['beta'].min():.4f}, {interp_data['beta'].max():.4f}]")
        print(f"  β monotonic: {np.all(np.diff(interp_data['beta']) >= -1e-6)}")
    else:
        interp_data = None
        print("  No pure liquid/vapour samples for interpolation")

    # ---- Sensitivity Analysis ----
    print("\n" + "=" * 60)
    print("7. Latent Dimension Sensitivity...")
    print("=" * 60)
    # Use median latent point as base
    base_latent = np.median(latent, axis=0)
    med_z = np.median(data["z"], axis=0)
    med_z = med_z / med_z.sum()

    sens_results = []
    for dim in range(config.latent_dim):
        sens = latent_sensitivity(surrogate, med_z, base_latent, dim, delta_range=3.0, n_points=51)
        sens_results.append(sens)
        beta_range = sens["beta"]
        print(f"  Dim {dim+1}: β range [{beta_range.min():.4f}, {beta_range.max():.4f}] "
              f"(Δβ={beta_range.max() - beta_range.min():.4f})")

    # ---- Generate Plots ----
    print("\n" + "=" * 60)
    print("8. Generating publication-quality figures...")
    print("=" * 60)
    plot_pca_3d(data, config.fig_latent_dir)
    print("  → latent_pca_3d.png")
    plot_umap_2d(data, config.fig_latent_dir)
    print("  → latent_umap_2d.png")
    plot_latent_correlation_heatmap(corr_df, config.fig_latent_dir)
    print("  → latent_correlation_heatmap.png")
    plot_phase_clustering(data, cluster_metrics, config.fig_latent_dir)
    print("  → latent_phase_clustering.png")
    if interp_data is not None:
        plot_interpolation(interp_data, z_mid, config.fig_latent_dir)
        print("  → latent_interpolation.png")
    plot_sensitivity(sens_results, med_z, config.fig_latent_dir)
    print("  → latent_sensitivity_beta.png, latent_sensitivity_K.png")

    # ---- Save Metrics ----
    summary = {
        "n_samples": int(len(latent)),
        "latent_dim": int(config.latent_dim),
        "latent_mean": latent.mean(axis=0).tolist(),
        "latent_std": latent.std(axis=0).tolist(),
        "pca_explained_variance_ratio": pca_results["explained_variance_ratio"],
        "pca_cumulative_variance": np.cumsum(pca_results["explained_variance_ratio"]).tolist(),
        "top_correlations": top_corrs,
        "cluster_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in cluster_metrics.items()},
    }

    summary_path = config.metric_dir / "latent_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Part 12: Latent Manifold Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
