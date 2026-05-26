"""Thermodynamic consistency verification for PGAE flash predictions.

Implements:
  1. Rachford-Rice residual check
  2. Gibbs free energy of mixing (PR-EOS fugacity)
  3. K-value parity analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# PR-EOS parameters (hard-coded from fluid_15comp.dat for reproducibility)
# ---------------------------------------------------------------------------

COMPONENT_NAMES = [
    "C10+", "N2", "CO2", "CH4", "C2H6",
    "C3H8", "IC4", "NC4", "IC5", "NC5",
    "FC6", "FC7", "FC8", "FC9", "FC10",
]

# Critical pressure (atm) — from *PCRIT
PC_ATM = np.array([
    15.0, 33.5, 72.8, 45.4, 48.2,
    41.9, 36.0, 37.5, 33.4, 33.3,
    32.46, 30.97, 29.12, 26.94, 25.01,
], dtype=np.float64)

# Critical temperature (K) — from *TCRIT
TC_K = np.array([
    700.0, 126.2, 304.2, 190.6, 305.4,
    369.8, 408.1, 425.2, 460.4, 469.6,
    507.5, 543.2, 570.5, 598.5, 622.1,
], dtype=np.float64)

# Acentric factor — from *AC
ACENTRIC = np.array([
    0.6, 0.04, 0.225, 0.008, 0.098,
    0.152, 0.176, 0.193, 0.227, 0.251,
    0.27504, 0.308301, 0.351327, 0.390781, 0.443774,
], dtype=np.float64)

# Molecular weight (g/mol) — from *MW
MW = np.array([
    200.0, 28.013, 44.01, 16.043, 30.07,
    44.097, 58.124, 58.124, 72.151, 72.151,
    86.0, 96.0, 107.0, 121.0, 134.0,
], dtype=np.float64)

R_GAS = 82.05746  # cm³·atm/(mol·K)
R_GAS_SI = 8.314462618  # J/(mol·K)

# BIP matrix (15×15, symmetric, diagonal=0)
# Only C10+ (col 0) and N2 (col 1) have non-zero BIP with other components.
BIP = np.zeros((15, 15), dtype=np.float64)
_bip_rows = [
    [0.0],                            # N2
    [0.0, 0.0],                       # CO2
    [0.025, 0.105],                   # CH4
    [0.01, 0.13],                     # C2H6
    [0.09, 0.125],                    # C3H8
    [0.095, 0.12],                    # IC4
    [0.095, 0.115],                   # NC4
    [0.1, 0.115],                     # IC5
    [0.11, 0.115],                    # NC5
    [0.11, 0.115],                    # FC6
    [0.11, 0.115],                    # FC7
    [0.11, 0.115],                    # FC8
    [0.11, 0.115],                    # FC9
    [0.11, 0.115],                    # FC10
]
for i, row in enumerate(_bip_rows):
    for j, val in enumerate(row):
        BIP[i + 1, j] = val
        BIP[j, i + 1] = val


def _pr_alpha(Tr: np.ndarray, omega: float) -> np.ndarray:
    """Peng-Robinson alpha(T) function."""
    if omega <= 0.49:
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    else:
        m = 0.3796 + 1.485 * omega - 0.1644 * omega ** 2 + 0.01667 * omega ** 3
    return (1.0 + m * (1.0 - np.sqrt(Tr))) ** 2


def _solve_cubic_z(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve PR-EOS cubic for Z.

    Returns (Z_vapour, Z_liquid) — largest and smallest real roots.
    """
    c2 = B - 1.0
    c1 = A - 2.0 * B - 3.0 * B ** 2
    c0 = -(A * B - B ** 2 - B ** 3)

    p = c1 - c2 ** 2 / 3.0
    q = c0 - c2 * c1 / 3.0 + 2.0 * c2 ** 3 / 27.0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

    Z_vap = np.zeros_like(A)
    Z_liq = np.zeros_like(A)

    # Three real roots (disc ≤ 0): trigonometric solution
    mask_neg = disc <= 0
    if mask_neg.any():
        r = np.sqrt(-p[mask_neg] / 3.0)
        phi = np.arccos(np.clip(-q[mask_neg] / (2.0 * r ** 3), -1.0, 1.0))
        Z_vap[mask_neg] = 2.0 * r * np.cos(phi / 3.0) - c2[mask_neg] / 3.0
        Z_liq[mask_neg] = 2.0 * r * np.cos((phi + 4.0 * np.pi) / 3.0) - c2[mask_neg] / 3.0

    # One real root (disc > 0): Cardano
    mask_pos = ~mask_neg
    if mask_pos.any():
        sqrt_disc = np.sqrt(disc[mask_pos])
        term1 = -q[mask_pos] / 2.0 + sqrt_disc
        term2 = -q[mask_pos] / 2.0 - sqrt_disc
        Z_single = (np.cbrt(term1) + np.cbrt(term2) - c2[mask_pos] / 3.0)
        Z_vap[mask_pos] = Z_single
        Z_liq[mask_pos] = Z_single

    Z_vap = np.clip(Z_vap, B + 1e-10, None)
    Z_liq = np.clip(Z_liq, B + 1e-10, None)
    return Z_vap, Z_liq


def _pr_ab(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pure-component PR a and b parameters."""
    Tr = T[:, np.newaxis] / TC_K[np.newaxis, :]  # (N, 15)
    alpha = np.array([_pr_alpha(Tr[:, i], ACENTRIC[i]) for i in range(15)]).T
    a_i = 0.45724 * R_GAS ** 2 * TC_K ** 2 / PC_ATM * alpha  # (N, 15)
    b_i = 0.07780 * R_GAS * TC_K / PC_ATM  # (15,)
    return a_i, b_i


def fugacity_coefficients(
    P_kPa: np.ndarray,
    T_K: np.ndarray,
    z: np.ndarray,
    phase: str = "vapour",
) -> np.ndarray:
    """Compute PR-EOS fugacity coefficients for each component.

    Args:
        P_kPa: pressure in kPa, shape (N,)
        T_K: temperature in K, shape (N,)
        z: mole fractions, shape (N, 15)
        phase: 'vapour' (largest Z) or 'liquid' (smallest Z)

    Returns:
        ln_phi: log fugacity coefficients, shape (N, 15)
    """
    P_atm = np.asarray(P_kPa, dtype=np.float64) / 101.325
    T = np.asarray(T_K, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    N = len(P_atm)

    a_i, b_i = _pr_ab(T)  # (N, 15), (15,)

    # Mixture parameters
    sqrt_a = np.sqrt(a_i)  # (N, 15)
    a_mix = np.zeros(N, dtype=np.float64)
    for i in range(15):
        for j in range(15):
            a_mix += z[:, i] * z[:, j] * (1.0 - BIP[i, j]) * sqrt_a[:, i] * sqrt_a[:, j]

    b_mix = z @ b_i  # (N,)

    A = a_mix * P_atm / (R_GAS * T) ** 2
    B = b_mix * P_atm / (R_GAS * T)

    Z_vap, Z_liq = _solve_cubic_z(A, B)
    Z = Z_vap if phase == "vapour" else Z_liq

    # Fugacity coefficients
    sqrt2 = np.sqrt(2.0)
    ln_phi = np.zeros((N, 15), dtype=np.float64)
    for i in range(15):
        sum_zj_aij = np.zeros(N, dtype=np.float64)
        for j in range(15):
            sum_zj_aij += z[:, j] * (1.0 - BIP[i, j]) * sqrt_a[:, i] * sqrt_a[:, j]

        ln_phi[:, i] = (
            b_i[i] / b_mix * (Z - 1.0)
            - np.log(Z - B)
            - A / (2.0 * sqrt2 * B) * (2.0 * sum_zj_aij / a_mix - b_i[i] / b_mix)
            * np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B + 1e-15))
        )
    return ln_phi


def gibbs_mixing_energy(
    P_kPa: np.ndarray,
    T_K: np.ndarray,
    z: np.ndarray,
    phase: str = "vapour",
) -> np.ndarray:
    """Dimensionless Gibbs free energy of mixing: ΔG_mix / (RT).

    ΔG_mix = RT Σ z_i ln(z_i φ_i)
    """
    z = np.clip(np.asarray(z, dtype=np.float64), 1e-15, None)
    ln_phi = fugacity_coefficients(P_kPa, T_K, z, phase)
    return np.sum(z * (np.log(z) + ln_phi), axis=-1)


def rachford_rice_residual(beta: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute Rachford-Rice equation residual.

    RR(beta) = Σ z_i (K_i - 1) / (1 + beta (K_i - 1))

    For a valid flash solution, RR(beta) should be 0.
    """
    beta = np.clip(np.asarray(beta, dtype=np.float64), 0.0, 1.0)
    z = np.asarray(z, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    denom = 1.0 + beta[:, np.newaxis] * (K - 1.0)
    denom = np.where(np.abs(denom) < 1e-15, np.sign(denom) * 1e-15, denom)
    return np.sum(z * (K - 1.0) / denom, axis=-1)


# ---------------------------------------------------------------------------
# High-level verification interface
# ---------------------------------------------------------------------------

@dataclass
class ThermoReport:
    rr_residual_mean: float
    rr_residual_median: float
    rr_residual_p99: float
    rr_pass_rate_1e4: float
    gibbs_dg_mean: float
    gibbs_dg_median: float
    gibbs_violation_rate: float
    k_mae: float
    k_r2: float
    k_light_mae: float
    k_heavy_mae: float
    figure_dir: Optional[Path] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Thermodynamic Consistency Report",
            "=" * 60,
            f"Rachford-Rice residual  mean={self.rr_residual_mean:.4e}  median={self.rr_residual_median:.4e}  p99={self.rr_residual_p99:.4e}",
            f"Rachford-Rice pass rate (|RR|<1e-4): {self.rr_pass_rate_1e4 * 100:.1f}%",
            f"Gibbs ΔG/RT  mean={self.gibbs_dg_mean:.6f}  median={self.gibbs_dg_median:.6f}",
            f"Gibbs violation rate (ΔG>0): {self.gibbs_violation_rate * 100:.2f}%",
            f"K-value MAE  all={self.k_mae:.4e}  light={self.k_light_mae:.4e}  heavy={self.k_heavy_mae:.4e}",
            f"K-value R² = {self.k_r2:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_thermo_checks(
    P: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    beta_pred: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    x_true: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    K_pred_direct: np.ndarray | None = None,
    figure_dir: Path | None = None,
) -> ThermoReport:
    """Run all thermodynamic consistency checks.

    Args:
        P: pressure (kPa), shape (N,)
        T: temperature (°C), will be converted to K
        z: feed composition, shape (N, 15)
        beta_pred: predicted vapour fraction, shape (N,)
        x_pred: predicted liquid composition, shape (N, 15)
        y_pred: predicted vapour composition, shape (N, 15)
        x_true: ground-truth liquid composition (optional)
        y_true: ground-truth vapour composition (optional)
        K_pred_direct: model-predicted K-values (optional, preferred over y/x)
        figure_dir: if provided, save K-value parity plots

    Returns:
        ThermoReport with all metrics
    """
    N = len(P)
    T_K = np.asarray(T, dtype=np.float64) + 273.15
    P_arr = np.asarray(P, dtype=np.float64)
    z = np.clip(np.asarray(z, dtype=np.float64), 1e-15, None)
    xp = np.clip(np.asarray(x_pred, dtype=np.float64), 1e-15, None)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-15, None)
    beta = np.clip(np.asarray(beta_pred, dtype=np.float64).reshape(-1), 0.0, 1.0)

    # ---- Rachford-Rice (use effective K from flash variables) ----
    K_eff = yp / (xp + 1e-15)  # implied K from predicted mole fractions
    rr = rachford_rice_residual(beta, z, K_eff)
    rr_abs = np.abs(rr)
    rr_pass = rr_abs < 1e-4

    # ---- Gibbs free energy ----
    # Compute single-phase feed Gibbs for both vapour and liquid roots.
    # The stable single-phase reference is the minimum of the two.
    g_single_v = gibbs_mixing_energy(P_arr, T_K, z, phase="vapour")
    g_single_l = gibbs_mixing_energy(P_arr, T_K, z, phase="liquid")
    g_single = np.minimum(g_single_v, g_single_l)  # stable phase minimises G

    # Two-phase Gibbs: (1-β)·G(x, liquid root) + β·G(y, vapour root)
    g_x = gibbs_mixing_energy(P_arr, T_K, xp, phase="liquid")
    g_y = gibbs_mixing_energy(P_arr, T_K, yp, phase="vapour")
    g_two = (1.0 - beta) * g_x + beta * g_y
    dg = g_two - g_single  # should be ≤ 0 for spontaneous phase split
    dg_violation = dg > 1e-9

    # ---- K-value accuracy (use model K if available, else effective K) ----
    if x_true is not None and y_true is not None:
        K_for_accuracy = K_pred_direct if K_pred_direct is not None else K_eff
        xt = np.clip(np.asarray(x_true, dtype=np.float64), 1e-15, None)
        yt = np.clip(np.asarray(y_true, dtype=np.float64), 1e-15, None)
        K_true = yt / (xt + 1e-15)
        k_abs_err = np.abs(np.log(K_for_accuracy + 1e-15) - np.log(K_true + 1e-15))
        k_mae = float(np.mean(k_abs_err))
        # R² in log-K space
        logK_p_flat = np.log(K_for_accuracy + 1e-15).ravel()
        logK_t_flat = np.log(K_true + 1e-15).ravel()
        ss_res = np.sum((logK_p_flat - logK_t_flat) ** 2)
        ss_tot = np.sum((logK_t_flat - np.mean(logK_t_flat)) ** 2)
        k_r2 = float(1.0 - ss_res / (ss_tot + 1e-15))
        # Light (C1-C3, indices 3,4,5) vs heavy (C7+, indices 10-14)
        light_idx = [3, 4, 5]
        heavy_idx = [10, 11, 12, 13, 14]
        k_light_mae = float(np.mean(k_abs_err[:, light_idx]))
        k_heavy_mae = float(np.mean(k_abs_err[:, heavy_idx]))
    else:
        k_mae = k_r2 = k_light_mae = k_heavy_mae = float("nan")

    report = ThermoReport(
        rr_residual_mean=float(np.mean(rr_abs)),
        rr_residual_median=float(np.median(rr_abs)),
        rr_residual_p99=float(np.percentile(rr_abs, 99)),
        rr_pass_rate_1e4=float(np.mean(rr_pass)),
        gibbs_dg_mean=float(np.mean(dg)),
        gibbs_dg_median=float(np.median(dg)),
        gibbs_violation_rate=float(np.mean(dg_violation)),
        k_mae=k_mae,
        k_r2=k_r2,
        k_light_mae=k_light_mae,
        k_heavy_mae=k_heavy_mae,
        figure_dir=figure_dir,
    )

    # ---- K-value parity plot ----
    if figure_dir is not None and x_true is not None and y_true is not None:
        _plot_kvalue_parity(K_true, K_for_accuracy, figure_dir)
        _plot_rr_histogram(rr_abs, figure_dir)
        _plot_gibbs_histogram(dg, figure_dir)

    return report


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_kvalue_parity(K_true: np.ndarray, K_pred: np.ndarray, out_dir: Path) -> None:
    logKt = np.log10(np.clip(K_true, 1e-10, None)).ravel()
    logKp = np.log10(np.clip(K_pred, 1e-10, None)).ravel()
    mask = np.isfinite(logKt) & np.isfinite(logKp)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(logKt[mask], logKp[mask], gridsize=80, cmap="Blues", mincnt=1, bins="log")
    lims = [min(logKt[mask].min(), logKp[mask].min()), max(logKt[mask].max(), logKp[mask].max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="y=x")
    ax.plot(lims, [l + 1 for l in lims], "gray", linewidth=0.5, alpha=0.5)
    ax.plot(lims, [l - 1 for l in lims], "gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("log₁₀ K (WinProp EOS)")
    ax.set_ylabel("log₁₀ K (PGAE)")
    ax.set_title("K-Value Parity (15 components × N samples)")
    ax.legend()
    cb = plt.colorbar(ax.collections[0], ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_dir / "kvalue_parity.png", dpi=220)
    plt.close(fig)


def _plot_rr_histogram(rr_abs: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(rr_abs + 1e-20), bins=80, alpha=0.85, color="steelblue", edgecolor="white")
    ax.axvline(-4, color="red", linestyle="--", label="1e-4 threshold")
    ax.set_xlabel("log₁₀ |RR residual|")
    ax.set_ylabel("count")
    ax.set_title("Rachford-Rice Residual Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rachford_rice_histogram.png", dpi=220)
    plt.close(fig)


def _plot_gibbs_histogram(dg: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dg, bins=80, alpha=0.85, color="darkorange", edgecolor="white")
    ax.axvline(0.0, color="red", linestyle="--", label="ΔG=0")
    ax.set_xlabel("ΔG_mix / (RT)  (two-phase − single-phase)")
    ax.set_ylabel("count")
    ax.set_title("Gibbs Free Energy of Mixing\n(negative = phase separation favourable)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gibbs_energy_histogram.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch evaluation helper (for train.py integration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_thermo_on_loader(model, loader, config, figure_dir: Path | None = None) -> Dict:
    """Run thermo checks on a full DataLoader and return structured results."""
    from train import move_batch

    model.eval()
    all_P, all_T, all_z, all_beta, all_x, all_y, all_K = [], [], [], [], [], [], []
    all_x_true, all_y_true = [], []

    for batch in loader:
        batch = move_batch(batch, config.device)
        pred = model(batch["input"])
        beta_p = pred["beta"].cpu().numpy().reshape(-1)
        x_p = pred["x"].cpu().numpy()
        y_p = pred["y"].cpu().numpy()
        P = batch["pt_raw"][:, 0].cpu().numpy()
        T = batch["pt_raw"][:, 1].cpu().numpy()
        z_np = batch["z"].cpu().numpy()
        x_t = batch["x"].cpu().numpy()
        y_t = batch["y"].cpu().numpy()

        all_P.append(P)
        all_T.append(T)
        all_z.append(z_np)
        all_beta.append(beta_p)
        all_x.append(x_p)
        all_y.append(y_p)
        all_x_true.append(x_t)
        all_y_true.append(y_t)
        if "K" in pred:
            all_K.append(pred["K"].cpu().numpy())

    P_arr = np.concatenate(all_P)
    T_arr = np.concatenate(all_T)
    z_arr = np.concatenate(all_z)
    beta_arr = np.concatenate(all_beta)
    x_arr = np.concatenate(all_x)
    y_arr = np.concatenate(all_y)
    xt_arr = np.concatenate(all_x_true)
    yt_arr = np.concatenate(all_y_true)
    K_arr = np.concatenate(all_K) if all_K else None

    report = run_thermo_checks(
        P=P_arr, T=T_arr, z=z_arr,
        beta_pred=beta_arr, x_pred=x_arr, y_pred=y_arr,
        x_true=xt_arr, y_true=yt_arr,
        K_pred_direct=K_arr,
        figure_dir=figure_dir,
    )
    print(report.summary())
    return {
        "rr_residual_mean": report.rr_residual_mean,
        "rr_residual_median": report.rr_residual_median,
        "rr_residual_p99": report.rr_residual_p99,
        "rr_pass_rate_1e4": report.rr_pass_rate_1e4,
        "gibbs_dg_mean": report.gibbs_dg_mean,
        "gibbs_dg_median": report.gibbs_dg_median,
        "gibbs_violation_rate": report.gibbs_violation_rate,
        "k_mae": report.k_mae,
        "k_r2": report.k_r2,
        "k_light_mae": report.k_light_mae,
        "k_heavy_mae": report.k_heavy_mae,
    }
