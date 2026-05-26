"""
Part 15: Robustness Testing for PGAE Flash Surrogate
======================================================
Comprehensive robustness evaluation across 6 dimensions:
  1. Input noise sensitivity (Monte Carlo perturbation)
  2. Extrapolation behavior (P, T beyond training range)
  3. Composition sensitivity (Lipschitz constant estimation)
  4. Physical constraint stress test (extreme inputs)
  5. Monte Carlo reliability analysis (uncertainty quantification)
  6. Adversarial perturbation search (worst-case sensitivity)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PGAEConfig
from infer import PGAEFlashSurrogate

# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10})
plt.rcParams["font.family"] = "DejaVu Sans"

_NC = 15
_Z_COLS = [f"z{i}" for i in range(1, _NC + 1)]


def _normalize_z(z: np.ndarray) -> np.ndarray:
    """Clip and re-normalize a composition vector to the simplex."""
    z = np.clip(z, 0.0, None)
    s = z.sum()
    return z / s if s > 1e-12 else z


def _random_z_on_simplex(rng: np.random.Generator) -> np.ndarray:
    """Sample a random point on the 15-simplex via Dirichlet(1,...,1)."""
    return rng.dirichlet(np.ones(_NC))


def _perturb_z(z: np.ndarray, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise then re-project onto simplex."""
    dz = rng.normal(0, sigma, size=_NC)
    return _normalize_z(z + dz)


def _predict(surrogate: PGAEFlashSurrogate, P: float, T: float,
             z: np.ndarray) -> Dict:
    return surrogate.predict_flash(P, T, z)


# ---------------------------------------------------------------------------
# Representative compositions (from training-data percentiles)
# ---------------------------------------------------------------------------

def _load_reference_compositions(
    csv_path: str,
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Return (z, CH4_fraction) for three representative compositions."""
    df = pd.read_csv(csv_path)
    z_arr = df[_Z_COLS].to_numpy(dtype=np.float32)
    # Ensure valid simplex
    z_arr = np.clip(z_arr, 0, None)
    z_arr = z_arr / z_arr.sum(axis=1, keepdims=True)
    ch4 = z_arr[:, 3]  # z4 = CH4

    def _select(label: str, frac: float) -> Tuple[np.ndarray, float]:
        idx = np.argmin(np.abs(ch4 - frac))
        return z_arr[idx].copy(), float(ch4[idx])

    return {
        "oil_rich": _select("Oil-rich (CH4≈20%)", 0.20),
        "typical": _select("Typical (CH4≈37%)", 0.368),
        "gas_rich": _select("Gas-rich (CH4≈63%)", 0.628),
    }


# ===================================================================
# Test 1: Input Noise Sensitivity
# ===================================================================

@dataclass
class NoiseSensitivityResult:
    label: str
    P_nominal: float
    T_nominal: float
    beta_nominal: float
    sigma_P_pct: float
    sigma_T: float
    sigma_z: float
    beta_mean: float
    beta_std: float
    beta_min: float
    beta_max: float
    mass_residual_mean: float
    n_violations: int  # β outside [0,1]
    n_samples: int = 500


def run_noise_sensitivity(
    surrogate: PGAEFlashSurrogate,
    ref_z: Dict[str, Tuple[np.ndarray, float]],
    n_mc: int = 500,
    seed: int = 2026,
) -> List[NoiseSensitivityResult]:
    """Monte Carlo noise injection at three (P,T) states per composition."""
    rng = np.random.default_rng(seed)
    conditions = [
        ("low", 5000.0, 60.0),
        ("mid", 20000.0, 100.0),
        ("high", 35000.0, 150.0),
    ]
    noise_levels = [
        ("low_noise", 0.001, 0.1, 0.0001),
        ("med_noise", 0.01, 1.0, 0.001),
        ("high_noise", 0.05, 5.0, 0.01),
    ]

    results: List[NoiseSensitivityResult] = []
    for comp_name, (z_ref, _) in ref_z.items():
        for cond_name, P0, T0 in conditions:
            # Nominal prediction
            nom = _predict(surrogate, P0, T0, z_ref)
            beta0 = nom["beta"]
            for nl_name, sp, sT, sz in noise_levels:
                betas, masses = [], []
                violations = 0
                for _ in range(n_mc):
                    P_n = P0 * (1.0 + rng.normal(0, sp))
                    P_n = max(P_n, 1.0)
                    T_n = T0 + rng.normal(0, sT)
                    z_n = _perturb_z(z_ref, sz, rng)
                    pred = _predict(surrogate, P_n, T_n, z_n)
                    b = pred["beta"]
                    betas.append(b)
                    masses.append(pred["mass_residual"])
                    if b < -1e-9 or b > 1.0 + 1e-9:
                        violations += 1
                betas = np.array(betas)
                results.append(NoiseSensitivityResult(
                    label=f"{comp_name}/{cond_name}/{nl_name}",
                    P_nominal=P0, T_nominal=T0, beta_nominal=beta0,
                    sigma_P_pct=sp, sigma_T=sT, sigma_z=sz,
                    beta_mean=float(betas.mean()),
                    beta_std=float(betas.std()),
                    beta_min=float(betas.min()),
                    beta_max=float(betas.max()),
                    mass_residual_mean=float(np.mean(masses)),
                    n_violations=violations,
                    n_samples=n_mc,
                ))
    return results


# ===================================================================
# Test 2: Extrapolation Stress Test
# ===================================================================

@dataclass
class ExtrapolationResult:
    comp_name: str
    sweep_type: str  # "P" or "T"
    fixed_val: float  # the value of the fixed variable
    sweep_vals: np.ndarray
    betas: np.ndarray
    mass_residuals: np.ndarray
    in_range_mask: np.ndarray  # True where sweep_val is in training range
    monotonicity_violations: int


def run_extrapolation(
    surrogate: PGAEFlashSurrogate,
    ref_z: Dict[str, Tuple[np.ndarray, float]],
) -> List[ExtrapolationResult]:
    """Sweep P and T far beyond training range for each reference composition."""
    results: List[ExtrapolationResult] = []

    # Training ranges
    P_train = (100.0, 50000.0)
    T_train = (40.0, 180.0)

    for comp_name, (z_ref, _) in ref_z.items():
        # --- P-sweep (fixed T=100°C) ---
        P_vals = np.logspace(np.log10(1), np.log10(100_000), 150)
        T_fixed = 100.0
        betas_p, masses_p = [], []
        for P in P_vals:
            pred = _predict(surrogate, float(P), T_fixed, z_ref)
            betas_p.append(pred["beta"])
            masses_p.append(pred["mass_residual"])
        betas_p = np.array(betas_p)
        masses_p = np.array(masses_p)
        in_range_p = (P_vals >= P_train[0]) & (P_vals <= P_train[1])
        # Monotonicity violations (β should decrease with P, per Part 10 fix)
        mono_v = int(np.sum(np.diff(betas_p) > 0))
        results.append(ExtrapolationResult(
            comp_name=comp_name, sweep_type="P", fixed_val=T_fixed,
            sweep_vals=P_vals, betas=betas_p, mass_residuals=masses_p,
            in_range_mask=in_range_p, monotonicity_violations=mono_v,
        ))

        # --- T-sweep (fixed P=20000 kPa) ---
        T_vals = np.linspace(-20, 250, 150)
        P_fixed = 20000.0
        betas_t, masses_t = [], []
        for T in T_vals:
            pred = _predict(surrogate, P_fixed, float(T), z_ref)
            betas_t.append(pred["beta"])
            masses_t.append(pred["mass_residual"])
        betas_t = np.array(betas_t)
        masses_t = np.array(masses_t)
        in_range_t = (T_vals >= T_train[0]) & (T_vals <= T_train[1])
        results.append(ExtrapolationResult(
            comp_name=comp_name, sweep_type="T", fixed_val=P_fixed,
            sweep_vals=T_vals, betas=betas_t, mass_residuals=masses_t,
            in_range_mask=in_range_t, monotonicity_violations=0,
        ))
    return results


# ===================================================================
# Test 3: Composition Sensitivity — Lipschitz Analysis
# ===================================================================

@dataclass
class LipschitzResult:
    comp_name: str
    P: float
    T: float
    beta_nominal: float
    L_mean: float
    L_max: float
    L_p95: float
    n_dirs: int


def run_lipschitz_analysis(
    surrogate: PGAEFlashSurrogate,
    ref_z: Dict[str, Tuple[np.ndarray, float]],
    n_dirs: int = 200,
    seed: int = 2026,
) -> List[LipschitzResult]:
    """Estimate local Lipschitz constant  |Δβ| / ‖Δz‖  for random simplex directions."""
    rng = np.random.default_rng(seed)
    states = [(5000.0, 60.0), (20000.0, 100.0), (35000.0, 150.0)]
    step_sizes = [1e-4, 1e-3, 1e-2]
    results: List[LipschitzResult] = []

    for comp_name, (z0, _) in ref_z.items():
        for P0, T0 in states:
            nom = _predict(surrogate, P0, T0, z0)
            beta0 = nom["beta"]
            L_vals_all = []
            for eps in step_sizes:
                for _ in range(n_dirs // len(step_sizes)):
                    # Random direction on simplex: difference of two Dirichlet samples
                    d = rng.dirichlet(np.ones(_NC)) - rng.dirichlet(np.ones(_NC))
                    d = d / (np.linalg.norm(d) + 1e-16)
                    dz = eps * d
                    z_p = _normalize_z(z0 + dz)
                    z_m = _normalize_z(z0 - dz)
                    bp = _predict(surrogate, P0, T0, z_p)["beta"]
                    bm = _predict(surrogate, P0, T0, z_m)["beta"]
                    L = abs(bp - bm) / (np.linalg.norm(z_p - z_m) + 1e-16)
                    L_vals_all.append(L)
            L_arr = np.array(L_vals_all)
            results.append(LipschitzResult(
                comp_name=comp_name, P=P0, T=T0, beta_nominal=beta0,
                L_mean=float(L_arr.mean()), L_max=float(L_arr.max()),
                L_p95=float(np.percentile(L_arr, 95)), n_dirs=n_dirs,
            ))
    return results


# ===================================================================
# Test 4: Physical Constraint Stress Test
# ===================================================================

@dataclass
class ConstraintStressResult:
    label: str
    P: float
    T: float
    z_desc: str
    beta: float
    x_sum: float
    y_sum: float
    mass_residual: float
    x_neg_count: int
    y_neg_count: int
    valid: bool  # all constraints satisfied


def run_constraint_stress(
    surrogate: PGAEFlashSurrogate,
) -> List[ConstraintStressResult]:
    """Feed extreme but physically legal inputs; check physical-constraint compliance."""
    results: List[ConstraintStressResult] = []

    # Build extreme compositions
    extreme_z_specs: List[Tuple[str, np.ndarray]] = []

    # Pure CH4
    z_pure_ch4 = np.zeros(_NC)
    z_pure_ch4[3] = 1.0
    extreme_z_specs.append(("pure_CH4", z_pure_ch4))

    # Pure CO2
    z_pure_co2 = np.zeros(_NC)
    z_pure_co2[1] = 1.0
    extreme_z_specs.append(("pure_CO2", z_pure_co2))

    # Pure C10+ (heaviest)
    z_heavy = np.zeros(_NC)
    z_heavy[0] = 1.0
    extreme_z_specs.append(("pure_C10plus", z_heavy))

    # Equal molar (15 comps)
    z_equal = np.ones(_NC) / _NC
    extreme_z_specs.append(("equal_molar", z_equal))

    # CH4-dominated (95% CH4 + 5% others)
    z_ch4_dom = np.ones(_NC) * 0.05 / (_NC - 1)
    z_ch4_dom[3] = 0.95
    extreme_z_specs.append(("CH4_dominant", z_ch4_dom))

    # Heavy-dominated (80% C10+)
    z_heavy_dom = np.ones(_NC) * 0.20 / (_NC - 1)
    z_heavy_dom[0] = 0.80
    extreme_z_specs.append(("heavy_dominant", z_heavy_dom))

    # Extreme P,T
    extreme_conditions = [
        ("ultra_low_P", 1.0, 50.0),
        ("ultra_high_P", 100_000.0, 100.0),
        ("ultra_low_T", 10_000.0, -50.0),
        ("ultra_high_T", 10_000.0, 300.0),
        ("extreme_all", 80_000.0, 250.0),
        ("mid_range", 20_000.0, 100.0),  # control
    ]

    for z_desc, z in extreme_z_specs:
        for cond_label, P, T in extreme_conditions:
            pred = _predict(surrogate, P, T, z)
            x = pred["x"]
            y = pred["y"]
            valid = True
            if pred["beta"] < -1e-9 or pred["beta"] > 1.0 + 1e-9:
                valid = False
            if abs(x.sum() - 1.0) > 1e-6:
                valid = False
            if abs(y.sum() - 1.0) > 1e-6:
                valid = False
            results.append(ConstraintStressResult(
                label=f"{z_desc}/{cond_label}",
                P=P, T=T, z_desc=z_desc,
                beta=pred["beta"],
                x_sum=float(x.sum()),
                y_sum=float(y.sum()),
                mass_residual=pred["mass_residual"],
                x_neg_count=int(np.sum(x < -1e-12)),
                y_neg_count=int(np.sum(y < -1e-12)),
                valid=valid,
            ))
    return results


# ===================================================================
# Test 5: Monte Carlo Reliability Analysis
# ===================================================================

@dataclass
class ReliabilityResult:
    sample_id: int
    P: float
    T: float
    z_ch4: float
    beta_true: float
    beta_mc_mean: float
    beta_mc_std: float
    beta_ci_95_low: float
    beta_ci_95_high: float
    ci_contains_true: bool
    n_mc: int


def run_reliability_analysis(
    surrogate: PGAEFlashSurrogate,
    csv_path: str,
    n_test: int = 50,
    n_mc: int = 200,
    seed: int = 2026,
) -> List[ReliabilityResult]:
    """
    For random test-set points, add realistic measurement noise
    (σ_P=1%, σ_T=0.5°C, σ_z=0.001) and compute β confidence intervals.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)
    z_arr = df[_Z_COLS].to_numpy(dtype=np.float32)
    z_arr = np.clip(z_arr, 0, None)
    z_arr = z_arr / z_arr.sum(axis=1, keepdims=True)

    # Select test points, prefer two-phase samples
    two_phase_mask = (df["beta_V"] > 0.01) & (df["beta_V"] < 0.99)
    idx_two = np.where(two_phase_mask.to_numpy())[0]
    idx_all = np.arange(len(df))
    chosen: List[int] = []
    if len(idx_two) >= n_test // 2:
        chosen = list(rng.choice(idx_two, size=n_test // 2, replace=False))
        remaining = list(set(idx_all) - set(chosen))
        chosen += list(rng.choice(remaining, size=n_test - len(chosen), replace=False))
    else:
        chosen = list(rng.choice(idx_all, size=n_test, replace=False))
    rng.shuffle(chosen)

    noise_sigma_P_pct = 0.01
    noise_sigma_T = 0.5
    noise_sigma_z = 0.001

    results: List[ReliabilityResult] = []
    for sid, idx in enumerate(chosen):
        row = df.iloc[idx]
        P0 = float(row["P"])
        T0 = float(row["T"])
        z0 = z_arr[idx]
        beta_true = float(row["beta_V"])

        mc_betas = []
        for _ in range(n_mc):
            P_n = P0 * (1.0 + rng.normal(0, noise_sigma_P_pct))
            P_n = max(P_n, 1.0)
            T_n = T0 + rng.normal(0, noise_sigma_T)
            z_n = _perturb_z(z0, noise_sigma_z, rng)
            mc_betas.append(_predict(surrogate, P_n, T_n, z_n)["beta"])

        mc_arr = np.array(mc_betas)
        ci_lo = float(np.percentile(mc_arr, 2.5))
        ci_hi = float(np.percentile(mc_arr, 97.5))
        results.append(ReliabilityResult(
            sample_id=sid, P=P0, T=T0,
            z_ch4=float(z0[3]),
            beta_true=beta_true,
            beta_mc_mean=float(mc_arr.mean()),
            beta_mc_std=float(mc_arr.std()),
            beta_ci_95_low=ci_lo,
            beta_ci_95_high=ci_hi,
            ci_contains_true=ci_lo <= beta_true <= ci_hi,
            n_mc=n_mc,
        ))
    return results


# ===================================================================
# Test 6: Adversarial Perturbation Search
# ===================================================================

@dataclass
class AdversarialResult:
    comp_name: str
    P: float
    T: float
    beta_original: float
    z_perturbed: np.ndarray
    beta_perturbed: float
    delta_beta: float
    delta_z_norm: float
    n_trials: int
    most_sensitive_comp_idx: int
    most_sensitive_comp_name: str


_COMPONENT_NAMES = [
    "C10+", "N2", "CO2", "CH4", "C2H6", "C3H8",
    "IC4", "NC4", "IC5", "NC5", "FC6", "FC7", "FC8", "FC9", "FC10",
]


def run_adversarial_search(
    surrogate: PGAEFlashSurrogate,
    ref_z: Dict[str, Tuple[np.ndarray, float]],
    n_trials: int = 5000,
    target_dz: float = 0.02,
    seed: int = 2026,
) -> List[AdversarialResult]:
    """
    Random search for worst-case z perturbation that maximises Δβ
    for a given ‖Δz‖ budget. Focus on two-phase states where
    the model is most sensitive.
    """
    rng = np.random.default_rng(seed)
    # Use (P,T) states in the two-phase region
    states = [(10000.0, 80.0), (20000.0, 100.0), (30000.0, 120.0)]
    results: List[AdversarialResult] = []

    for comp_name, (z0, _) in ref_z.items():
        for P0, T0 in states:
            beta_orig = _predict(surrogate, P0, T0, z0)["beta"]

            best_dbeta = -1.0
            best_zp = None
            best_dznorm = 0.0
            comp_sensitivity = np.zeros(_NC)

            for _ in range(n_trials):
                # Generate a random direction and scale to target ‖Δz‖
                d = rng.dirichlet(np.ones(_NC)) - rng.dirichlet(np.ones(_NC))
                dn = np.linalg.norm(d)
                if dn < 1e-16:
                    continue
                d = d / dn * target_dz * rng.uniform(0.5, 2.0)
                zp = _normalize_z(z0 + d)
                bp = _predict(surrogate, P0, T0, zp)["beta"]
                dbeta = abs(bp - beta_orig)
                dznorm = np.linalg.norm(zp - z0)

                # Accumulate per-component sensitivity (weighted by dbeta)
                comp_sensitivity += dbeta * np.abs(zp - z0)

                if dbeta > best_dbeta:
                    best_dbeta = dbeta
                    best_zp = zp.copy()
                    best_dznorm = dznorm

            # Normalize component sensitivity
            comp_sensitivity /= max(comp_sensitivity.sum(), 1e-16)
            most_sensitive = int(np.argmax(comp_sensitivity))

            results.append(AdversarialResult(
                comp_name=comp_name, P=P0, T=T0,
                beta_original=beta_orig,
                z_perturbed=best_zp,
                beta_perturbed=_predict(surrogate, P0, T0, best_zp)["beta"],
                delta_beta=best_dbeta,
                delta_z_norm=best_dznorm,
                n_trials=n_trials,
                most_sensitive_comp_idx=most_sensitive,
                most_sensitive_comp_name=_COMPONENT_NAMES[most_sensitive],
            ))
    return results


# ===================================================================
# Visualisation
# ===================================================================

def plot_noise_heatmap(results: List[NoiseSensitivityResult],
                       save_to: Path) -> None:
    """Heatmap of β std vs noise level × composition × condition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [1, 1, 0.08]})

    # Build matrix: rows=composition×condition, cols=noise_level
    rows_labels = []
    matrix = []
    noise_labels = ["low", "med", "high"]
    for comp in ["oil_rich", "typical", "gas_rich"]:
        for cond in ["low", "mid", "high"]:
            row = []
            rows_labels.append(f"{comp}/{cond}")
            for nl in noise_labels:
                match = [r for r in results
                         if comp in r.label and f"/{cond}/" in r.label
                         and f"/{nl}" in r.label.replace("low_noise", "low")
                                 .replace("med_noise", "med")
                                 .replace("high_noise", "high")]
                # More robust matching
                val = np.nan
                for r2 in results:
                    parts = r2.label.split("/")
                    if len(parts) == 3:
                        c, cd, nl2 = parts
                        if c == comp and cd == cond:
                            nlm = nl2.replace("_noise", "")
                            if nlm == nl:
                                val = r2.beta_std
                                break
                row.append(val)
            matrix.append(row)

    matrix = np.array(matrix)
    im = axes[0].imshow(matrix, aspect="auto", cmap="YlOrRd")
    axes[0].set_xticks(range(3), noise_labels)
    axes[0].set_yticks(range(9), rows_labels, fontsize=7)
    axes[0].set_title("β std under input noise")
    for i in range(9):
        for j in range(3):
            v = matrix[i, j]
            axes[0].text(j, i, f"{v:.4f}" if not np.isnan(v) else "N/A",
                         ha="center", va="center", fontsize=6,
                         color="white" if (not np.isnan(v) and v > 0.05) else "black")

    # β range bar chart for highest noise
    high_noise = [r for r in results if "high_noise" in r.label]
    labels_short = [r.label.replace("/high_noise", "").replace("_", " ") for r in high_noise]
    betas_nom = [r.beta_nominal for r in high_noise]
    betas_lo = [r.beta_min for r in high_noise]
    betas_hi = [r.beta_max for r in high_noise]
    y_pos = range(len(high_noise))
    axes[1].barh(y_pos, [h - l for l, h in zip(betas_lo, betas_hi)],
                 left=betas_lo, height=0.6, color="steelblue", alpha=0.7)
    axes[1].scatter(betas_nom, y_pos, color="red", s=30, zorder=5, label="nominal")
    axes[1].set_yticks(y_pos, labels_short, fontsize=6)
    axes[1].set_xlabel("β range (5% noise)")
    axes[1].set_title("β spread under max noise")
    axes[1].legend(fontsize=7)
    axes[1].set_xlim(0, 1)

    plt.colorbar(im, cax=axes[2], label="β std")
    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_extrapolation(results: List[ExtrapolationResult],
                       save_to: Path) -> None:
    """P-sweep and T-sweep for each composition, with training-range shading."""
    comps = sorted(set(r.comp_name for r in results))
    fig, axes = plt.subplots(2, len(comps), figsize=(5 * len(comps), 8))

    for ci, comp in enumerate(comps):
        # P-sweep
        ax_p = axes[0, ci] if len(comps) > 1 else axes[0]
        p_res = [r for r in results if r.comp_name == comp and r.sweep_type == "P"][0]
        ax_p.plot(p_res.sweep_vals, p_res.betas, "b-", lw=1.2)
        ax_p.axvspan(100, 50000, alpha=0.08, color="green", label="training P")
        ax_p.set_xscale("log")
        ax_p.set_xlabel("P (kPa)")
        ax_p.set_ylabel("β")
        ax_p.set_title(f"{comp} P-sweep (T={p_res.fixed_val}°C)")
        ax_p.legend(fontsize=7)
        ax_p.set_ylim(-0.05, 1.05)
        in_p = p_res.in_range_mask
        ax_p.axhline(0, color="gray", ls="--", lw=0.5)
        ax_p.axhline(1, color="gray", ls="--", lw=0.5)

        # T-sweep
        ax_t = axes[1, ci] if len(comps) > 1 else axes[1]
        t_res = [r for r in results if r.comp_name == comp and r.sweep_type == "T"][0]
        ax_t.plot(t_res.sweep_vals, t_res.betas, "r-", lw=1.2)
        ax_t.axvspan(40, 180, alpha=0.08, color="green", label="training T")
        ax_t.set_xlabel("T (°C)")
        ax_t.set_ylabel("β")
        ax_t.set_title(f"{comp} T-sweep (P={t_res.fixed_val} kPa)")
        ax_t.legend(fontsize=7)
        ax_t.set_ylim(-0.05, 1.05)
        ax_t.axhline(0, color="gray", ls="--", lw=0.5)
        ax_t.axhline(1, color="gray", ls="--", lw=0.5)

    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lipschitz(results: List[LipschitzResult], save_to: Path) -> None:
    """Grouped bar chart of Lipschitz constants."""
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"{r.comp_name}\nP={r.P:.0f} T={r.T:.0f}°C" for r in results]
    x = np.arange(len(results))
    w = 0.25
    ax.bar(x - w, [r.L_mean for r in results], w, label="L_mean", color="steelblue")
    ax.bar(x, [r.L_p95 for r in results], w, label="L_p95", color="darkorange")
    ax.bar(x + w, [r.L_max for r in results], w, label="L_max", color="firebrick")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Lipschitz constant |Δβ|/‖Δz‖")
    ax.set_title("Composition Sensitivity (Lipschitz Analysis)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(results: List[ReliabilityResult], save_to: Path) -> None:
    """Error-bar plot: β_true vs β_mc with 95% CI."""
    results_sorted = sorted(results, key=lambda r: r.beta_true)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(results_sorted))
    yerr_lo = [max(0, r.beta_mc_mean - r.beta_ci_95_low) for r in results_sorted]
    yerr_hi = [max(0, r.beta_ci_95_high - r.beta_mc_mean) for r in results_sorted]
    ax.errorbar(x, [r.beta_mc_mean for r in results_sorted],
                yerr=[yerr_lo, yerr_hi],
                fmt="o", color="steelblue", capsize=2, ms=4, label="MC mean ± 95% CI")
    ax.scatter(x, [r.beta_true for r in results_sorted], color="red", s=15,
               zorder=5, label="true β")
    ax.set_xlabel("Sample (sorted by β_true)")
    ax.set_ylabel("β")
    ax.set_title("Monte Carlo Reliability: 95% CI coverage")
    ax.legend(fontsize=8)
    coverage = sum(1 for r in results_sorted if r.ci_contains_true)
    ax.text(0.98, 0.02, f"CI coverage: {coverage}/{len(results_sorted)}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial(results: List[AdversarialResult], save_to: Path) -> None:
    """Adversarial sensitivity summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Δβ vs ‖Δz‖
    ax = axes[0]
    labels = [f"{r.comp_name[:6]}\nP={r.P:.0f}" for r in results]
    x = np.arange(len(results))
    ax.bar(x, [r.delta_beta for r in results], color="tomato", alpha=0.7,
           label=f"Δβ (‖Δz‖≈{results[0].delta_z_norm:.3f})")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Δβ")
    ax.set_title("Worst-case β perturbation")
    ax.legend(fontsize=7)

    # Right: Most sensitive component
    ax2 = axes[1]
    comp_counts = {}
    for r in results:
        name = r.most_sensitive_comp_name
        comp_counts[name] = comp_counts.get(name, 0) + 1
    sorted_comps = sorted(comp_counts.items(), key=lambda x: -x[1])
    names, counts = zip(*sorted_comps)
    ax2.bar(names, counts, color="steelblue", alpha=0.7)
    ax2.set_ylabel("Times most sensitive")
    ax2.set_title("Most Sensitive Component (adversarial)")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_stress(results: List[ConstraintStressResult],
                           save_to: Path) -> None:
    """Heatmap of constraint violations, labelled by result."""
    fig, ax = plt.subplots(figsize=(16, 8))
    rows = sorted(set(r.z_desc for r in results))
    cols = sorted(set(r.label.split("/")[1] for r in results))
    # Build violation matrix
    viol = np.zeros((len(rows), len(cols)))
    for r in results:
        ri = rows.index(r.z_desc)
        ci = cols.index(r.label.split("/")[1])
        viol[ri, ci] = 0 if r.valid else 1 + r.mass_residual * 100

    im = ax.imshow(viol, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=2)
    ax.set_xticks(range(len(cols)), cols, fontsize=7, rotation=30)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_title("Constraint Violations (0=valid, >0=failed, darker=worse)")
    plt.colorbar(im, ax=ax, label="violation severity")
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            r_match = [r for r in results
                       if r.z_desc == rows[ri] and cols[ci] in r.label]
            if r_match:
                r2 = r_match[0]
                status = "✓" if r2.valid else "✗"
                ax.text(ci, ri, f"{status}\nβ={r2.beta:.3f}",
                        ha="center", va="center", fontsize=5)
    fig.tight_layout()
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(
    noise_results: List[NoiseSensitivityResult],
    extrap_results: List[ExtrapolationResult],
    lip_results: List[LipschitzResult],
    reli_results: List[ReliabilityResult],
    adv_results: List[AdversarialResult],
    save_to: Path,
) -> None:
    """A single summary dashboard with key robustness metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # (0,0): β std distribution across noise tests
    ax1 = fig.add_subplot(gs[0, 0])
    all_std = [r.beta_std for r in noise_results]
    ax1.hist(all_std, bins=25, color="steelblue", edgecolor="white")
    ax1.axvline(np.mean(all_std), color="red", ls="--", label=f"mean={np.mean(all_std):.3f}")
    ax1.set_xlabel("β std")
    ax1.set_title("Noise Sensitivity Distribution")
    ax1.legend(fontsize=7)

    # (0,1): Extrapolation summary — β range
    ax2 = fig.add_subplot(gs[0, 1])
    comps_disp = {"oil_rich": "Oil", "typical": "Typ", "gas_rich": "Gas"}
    for ci, comp in enumerate(["oil_rich", "typical", "gas_rich"]):
        p_res = [r for r in extrap_results if r.comp_name == comp and r.sweep_type == "P"][0]
        in_range = p_res.in_range_mask
        out_range = ~in_range
        bp_in = p_res.betas[in_range]
        bp_out = p_res.betas[out_range]
        ax2.scatter([ci - 0.15] * len(bp_in), bp_in, s=3, alpha=0.3,
                    color="green", label="in-range" if ci == 0 else "")
        ax2.scatter([ci + 0.15] * len(bp_out), bp_out, s=3, alpha=0.3,
                    color="red", label="out-of-range" if ci == 0 else "")
    ax2.set_xticks(range(3), [comps_disp[c] for c in ["oil_rich", "typical", "gas_rich"]])
    ax2.set_ylabel("β (P-sweep)")
    ax2.set_title("Extrapolation: β scatter")
    ax2.legend(fontsize=6)

    # (0,2): Lipschitz summary
    ax3 = fig.add_subplot(gs[0, 2])
    L_means = [r.L_mean for r in lip_results]
    ax3.bar(range(len(L_means)), L_means, color="steelblue")
    ax3.set_xlabel("Test #")
    ax3.set_ylabel("L_mean")
    ax3.set_title(f"Lipschitz: mean={np.mean(L_means):.1f}, max={max(L_means):.1f}")
    ax3.set_yscale("log")

    # (1,0): Reliability coverage
    ax4 = fig.add_subplot(gs[1, 0])
    ci_widths = [r.beta_ci_95_high - r.beta_ci_95_low for r in reli_results]
    ax4.hist(ci_widths, bins=20, color="darkorange", edgecolor="white")
    ax4.set_xlabel("95% CI width")
    ax4.set_title(f"Reliability: CI width mean={np.mean(ci_widths):.3f}")

    # (1,1): Adversarial Δβ summary
    ax5 = fig.add_subplot(gs[1, 1])
    labels_adv = [f"{r.comp_name[:8]}\n{r.P:.0f}kPa" for r in adv_results]
    ax5.barh(range(len(adv_results)), [r.delta_beta for r in adv_results],
             color="tomato", alpha=0.7)
    ax5.set_yticks(range(len(adv_results)), labels_adv, fontsize=6)
    ax5.set_xlabel("Δβ (adversarial)")
    ax5.set_title("Worst-case β change")

    # (1,2): Constraint stress pass rate summary
    ax6 = fig.add_subplot(gs[1, 2])
    # We'll compute after collecting constraint results
    ax6.text(0.5, 0.5, "See constraint\nstress chart", ha="center", va="center",
             transform=ax6.transAxes, fontsize=12)
    ax6.set_title("Constraint Stress")

    # (2,0)-(2,2): Robustness score card
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    # Compute scores
    noise_score = max(0, 1 - np.mean(all_std) / 0.1)  # lower std = better
    mono_ok = 0
    for comp in ["oil_rich", "typical", "gas_rich"]:
        p_res = [r for r in extrap_results if r.comp_name == comp and r.sweep_type == "P"]
        if p_res:
            in_r = p_res[0].in_range_mask
            b_in = p_res[0].betas[in_r]
            if len(b_in) > 1 and np.all(np.diff(b_in) <= 0):
                mono_ok += 1
    mono_score = mono_ok / 3
    lip_score = max(0, 1 - np.mean(L_means) / 50)  # L<50 is reasonable
    coverage = sum(1 for r in reli_results if r.ci_contains_true) / len(reli_results)
    adv_score = max(0, 1 - np.mean([r.delta_beta for r in adv_results]) / 0.3)

    overall = 0.20 * noise_score + 0.25 * mono_score + 0.20 * lip_score + 0.20 * coverage + 0.15 * adv_score

    score_lines = [
        "ROBUSTNESS SCORE CARD",
        "=" * 60,
        f"  Input Noise Stability:       {noise_score:.2f}  (β std under 5% noise)",
        f"  Extrapolation Monotonicity:   {mono_score:.2f}  (in-range β-P direction correct)",
        f"  Composition Smoothness:       {lip_score:.2f}  (Lipschitz constant reasonable)",
        f"  Reliability Coverage:         {coverage:.2f}  (95% CI captures true β)",
        f"  Adversarial Robustness:       {adv_score:.2f}  (max Δβ under small Δz)",
        "-" * 60,
        f"  OVERALL ROBUSTNESS SCORE:     {overall:.2f} / 1.00",
        "=" * 60,
    ]
    for i, line in enumerate(score_lines):
        color = "black"
        if "OVERALL" in line:
            color = "green" if overall >= 0.7 else ("orange" if overall >= 0.4 else "red")
        ax7.text(0.05, 0.95 - i * 0.12, line, transform=ax7.transAxes,
                 fontsize=9, fontfamily="monospace", color=color,
                 fontweight="bold" if "OVERALL" in line or "SCORE CARD" in line else "normal")

    fig.suptitle("PGAE Robustness Dashboard", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    config = PGAEConfig()
    print("=" * 64)
    print("Part 15: PGAE Robustness Testing")
    print("=" * 64)

    t0 = time.perf_counter()

    # --- Init ---
    print("\n[1/6] Loading surrogate model ...")
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)
    print(f"  Model: {sum(p.numel() for p in surrogate.model.parameters()):,} params")
    print(f"  Device: {config.device}")

    # --- Reference compositions ---
    ref_z = _load_reference_compositions(str(config.dataset_path))
    for name, (z, ch4) in ref_z.items():
        print(f"  {name}: CH4={ch4:.3f}, z_sum={z.sum():.4f}")

    # --- Ensure output dirs ---
    fig_dir = config.fig_robustness_dir
    metric_dir = config.metric_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict = {}

    # ================================================================
    # Test 1: Input Noise Sensitivity
    # ================================================================
    print("\n[2/6] Input noise sensitivity (MC=500, 3×3×3=27 tests) ...")
    noise_results = run_noise_sensitivity(surrogate, ref_z, n_mc=500)
    max_std = max(r.beta_std for r in noise_results)
    mean_std = np.mean([r.beta_std for r in noise_results])
    print(f"  β std range: [{min(r.beta_std for r in noise_results):.5f}, {max_std:.4f}]")
    print(f"  β std mean: {mean_std:.5f}")
    n_viol = sum(r.n_violations for r in noise_results)
    print(f"  Total β violations: {n_viol} / {sum(r.n_samples for r in noise_results)}")

    all_metrics["noise_sensitivity"] = {
        "n_tests": len(noise_results),
        "beta_std_mean": float(mean_std),
        "beta_std_max": float(max_std),
        "total_violations": n_viol,
        "worst_case": max(noise_results, key=lambda r: r.beta_std).label,
        "details": [
            {
                "label": r.label,
                "beta_nominal": r.beta_nominal,
                "beta_std": r.beta_std,
                "beta_range": [r.beta_min, r.beta_max],
                "n_violations": r.n_violations,
            }
            for r in noise_results
        ],
    }

    plot_noise_heatmap(noise_results, fig_dir / "robustness_noise.png")

    # ================================================================
    # Test 2: Extrapolation
    # ================================================================
    print("\n[3/6] Extrapolation stress test (P: 1→100k kPa, T: -20→250°C) ...")
    extrap_results = run_extrapolation(surrogate, ref_z)
    for r in extrap_results:
        in_range = r.in_range_mask
        if r.sweep_type == "P":
            b_in = r.betas[in_range]
            beta_range_in = f"[{b_in.min():.3f}, {b_in.max():.3f}]"
            print(f"  {r.comp_name} P-sweep: in-range β{beta_range_in}, "
                  f"mono_viol={r.monotonicity_violations}, "
                  f"all β∈[{r.betas.min():.3f},{r.betas.max():.3f}]")
        else:
            b_in = r.betas[in_range]
            beta_range_in = f"[{b_in.min():.3f}, {b_in.max():.3f}]"
            print(f"  {r.comp_name} T-sweep: in-range β{beta_range_in}, "
                  f"all β∈[{r.betas.min():.3f},{r.betas.max():.3f}]")

    all_metrics["extrapolation"] = {
        "n_sweeps": len(extrap_results),
        "details": [
            {
                "comp": r.comp_name,
                "sweep_type": r.sweep_type,
                "beta_in_range_min": float(r.betas[r.in_range_mask].min()),
                "beta_in_range_max": float(r.betas[r.in_range_mask].max()),
                "beta_all_min": float(r.betas.min()),
                "beta_all_max": float(r.betas.max()),
                "mono_violations": r.monotonicity_violations,
            }
            for r in extrap_results
        ],
    }

    plot_extrapolation(extrap_results, fig_dir / "robustness_extrapolation.png")

    # ================================================================
    # Test 3: Composition Sensitivity (Lipschitz)
    # ================================================================
    print("\n[4/6] Composition sensitivity (Lipschitz, 200 dir × 3 eps × 9 pts) ...")
    lip_results = run_lipschitz_analysis(surrogate, ref_z)
    L_all_mean = np.mean([r.L_mean for r in lip_results])
    L_all_max = max(r.L_max for r in lip_results)
    print(f"  L_mean range: [{min(r.L_mean for r in lip_results):.1f}, "
          f"{max(r.L_mean for r in lip_results):.1f}]")
    print(f"  L_max overall: {L_all_max:.1f}")
    print(f"  L_mean overall: {L_all_mean:.1f}")

    all_metrics["lipschitz"] = {
        "n_tests": len(lip_results),
        "L_mean_overall": float(L_all_mean),
        "L_max_overall": float(L_all_max),
        "details": [
            {
                "comp": r.comp_name, "P": r.P, "T": r.T,
                "L_mean": r.L_mean, "L_max": r.L_max, "L_p95": r.L_p95,
            }
            for r in lip_results
        ],
    }

    plot_lipschitz(lip_results, fig_dir / "robustness_lipschitz.png")

    # ================================================================
    # Test 4: Physical Constraint Stress
    # ================================================================
    print("\n[5/6] Physical constraint stress (extreme P,T,z) ...")
    constraint_results = run_constraint_stress(surrogate)
    n_valid = sum(1 for r in constraint_results if r.valid)
    n_total = len(constraint_results)
    print(f"  Valid: {n_valid}/{n_total}")
    for r in constraint_results:
        if not r.valid or r.mass_residual > 0.01:
            print(f"  ! {r.label}: β={r.beta:.4f}, mass_res={r.mass_residual:.2e}, "
                  f"x_neg={r.x_neg_count}, y_neg={r.y_neg_count}, valid={r.valid}")

    all_metrics["constraint_stress"] = {
        "n_tests": n_total,
        "n_valid": n_valid,
        "pass_rate": n_valid / n_total,
        "details": [
            {
                "label": r.label, "beta": r.beta,
                "x_sum": r.x_sum, "y_sum": r.y_sum,
                "mass_residual": r.mass_residual,
                "x_neg": r.x_neg_count, "y_neg": r.y_neg_count,
                "valid": r.valid,
            }
            for r in constraint_results
        ],
    }

    plot_constraint_stress(constraint_results, fig_dir / "robustness_constraint.png")

    # ================================================================
    # Test 5: Monte Carlo Reliability
    # ================================================================
    print("\n[6/6] Monte Carlo reliability (50 pts × 200 MC) ...")
    reli_results = run_reliability_analysis(
        surrogate, str(config.dataset_path), n_test=50, n_mc=200,
    )
    coverage = sum(1 for r in reli_results if r.ci_contains_true) / len(reli_results)
    mean_ci_w = np.mean([r.beta_ci_95_high - r.beta_ci_95_low for r in reli_results])
    print(f"  95% CI coverage: {coverage:.2%}")
    print(f"  Mean CI width: {mean_ci_w:.4f}")
    print(f"  Mean β MC std: {np.mean([r.beta_mc_std for r in reli_results]):.5f}")

    all_metrics["reliability"] = {
        "n_test_points": len(reli_results),
        "n_mc_per_point": reli_results[0].n_mc if reli_results else 0,
        "ci_coverage_95": float(coverage),
        "mean_ci_width": float(mean_ci_w),
        "mean_beta_mc_std": float(np.mean([r.beta_mc_std for r in reli_results])),
    }

    plot_reliability(reli_results, fig_dir / "robustness_reliability.png")

    # ================================================================
    # Test 6 (extra): Adversarial Search
    # ================================================================
    print("\n[extra] Adversarial perturbation search (5000 trials × 9 pts) ...")
    adv_results = run_adversarial_search(surrogate, ref_z, n_trials=5000)
    for r in adv_results:
        print(f"  {r.comp_name} P={r.P:.0f}: Δβ={r.delta_beta:.4f} "
              f"(‖Δz‖={r.delta_z_norm:.4f}), most_sensitive={r.most_sensitive_comp_name}")

    all_metrics["adversarial"] = {
        "n_tests": len(adv_results),
        "max_delta_beta": float(max(r.delta_beta for r in adv_results)),
        "mean_delta_beta": float(np.mean([r.delta_beta for r in adv_results])),
        "avg_dz_norm": float(np.mean([r.delta_z_norm for r in adv_results])),
    }

    plot_adversarial(adv_results, fig_dir / "robustness_adversarial.png")

    # ================================================================
    # Summary Dashboard
    # ================================================================
    plot_summary_dashboard(
        noise_results, extrap_results, lip_results,
        reli_results, adv_results,
        fig_dir / "robustness_dashboard.png",
    )

    # ================================================================
    # Save metrics & report
    # ================================================================
    metrics_path = metric_dir / "robustness_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")
    print(f"Metrics saved: {metrics_path}")
    print(f"Figures saved: {fig_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
