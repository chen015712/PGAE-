"""Part 10: Phase Boundary Continuity Verification.

Fixed-composition pressure scan:
  - PGAE β-P curve (500 points, instant)
  - WinProp EOS benchmark (coarse grid, ~30 points)
  - Bubble/dew point binary search
  - Continuity, monotonicity, smoothness evaluation
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PGAEConfig
from infer import PGAEFlashSurrogate

WINPROP_EXE = r"D:\CMG\WINPROP\2022.10\Win_x64\EXE\pr202210.exe"
TEMPLATE_DAT = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\fluid_15comp.dat")
WORK_DIR = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\sim")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Light component range for C1-C3, heavy for C7+
COMP_NAMES = ["C10+", "N2", "CO2", "CH4", "C2H6", "C3H8", "IC4", "NC4", "IC5", "NC5",
              "FC6", "FC7", "FC8", "FC9", "FC10"]
NC = 15

# ---------------------------------------------------------------------------
# 1. PGAE pressure scan
# ---------------------------------------------------------------------------

def pgae_pressure_scan(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    T: float,
    P_range: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run PGAE on a pressure sweep at fixed composition and temperature.

    Args:
        surrogate: loaded PGAEFlashSurrogate
        z: feed composition (15,)
        T: temperature (°C)
        P_range: array of pressures (kPa)

    Returns:
        dict with keys: P, beta, x, y, latent, mass_residual, x_sum, y_sum
    """
    N = len(P_range)
    beta = np.zeros(N)
    x = np.zeros((N, NC))
    y = np.zeros((N, NC))
    latent = np.zeros((N, surrogate.config.latent_dim))
    mass_res = np.zeros(N)
    x_sum = np.zeros(N)
    y_sum = np.zeros(N)

    for i, P in enumerate(tqdm(P_range, desc="PGAE P-scan", leave=False)):
        pred = surrogate.predict_flash(float(P), float(T), z)
        beta[i] = pred["beta"]
        x[i] = pred["x"]
        y[i] = pred["y"]
        latent[i] = pred["latent"]
        mass_res[i] = pred["mass_residual"]
        x_sum[i] = pred["x_sum"]
        y_sum[i] = pred["y_sum"]

    return {
        "P": P_range,
        "beta": beta,
        "x": x,
        "y": y,
        "latent": latent,
        "mass_residual": mass_res,
        "x_sum": x_sum,
        "y_sum": y_sum,
    }


# ---------------------------------------------------------------------------
# 2. Bubble / Dew point binary search
# ---------------------------------------------------------------------------

def find_bubble_point(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    T: float,
    P_low: float = 100.0,
    P_high: float = 50000.0,
    tol: float = 1.0,
    max_iter: int = 60,
) -> Optional[float]:
    """Binary search for bubble point (β → 0).

    If binary search fails (β never crosses threshold), returns the P
    with minimum β as a best-effort estimate.
    """
    beta_low = surrogate.predict_flash(P_low, T, z)["beta"]
    beta_high = surrogate.predict_flash(P_high, T, z)["beta"]

    if beta_low < 1e-4:
        return P_low
    if beta_high < 1e-4:
        return P_high

    # Binary search
    lo, hi = P_low, P_high
    for _ in range(max_iter):
        P_mid = (lo + hi) / 2.0
        if (hi - lo) < tol:
            return P_mid
        beta_mid = surrogate.predict_flash(P_mid, T, z)["beta"]
        if beta_mid < 1e-4:
            hi = P_mid
        else:
            lo = P_mid

    # If binary search didn't converge, use grid search for min beta
    P_test = np.linspace(P_low, P_high, 100)
    betas = np.array([surrogate.predict_flash(float(p), T, z)["beta"] for p in P_test])
    min_idx = np.argmin(betas)
    if betas[min_idx] < 0.05:
        return float(P_test[min_idx])
    return None  # beta never approaches 0


def find_dew_point(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    T: float,
    P_low: float = 100.0,
    P_high: float = 50000.0,
    tol: float = 1.0,
    max_iter: int = 60,
) -> Optional[float]:
    """Binary search for dew point (β → 1).

    If binary search fails, returns the P with maximum β as a best-effort estimate.
    """
    beta_low = surrogate.predict_flash(P_low, T, z)["beta"]
    beta_high = surrogate.predict_flash(P_high, T, z)["beta"]

    if beta_low > 1.0 - 1e-4:
        return P_low
    if beta_high > 1.0 - 1e-4:
        return P_high

    # Binary search
    lo, hi = P_low, P_high
    for _ in range(max_iter):
        P_mid = (lo + hi) / 2.0
        if (hi - lo) < tol:
            return P_mid
        beta_mid = surrogate.predict_flash(P_mid, T, z)["beta"]
        if beta_mid > 1.0 - 1e-4:
            lo = P_mid
        else:
            hi = P_mid

    # If binary search didn't converge, use grid search for max beta
    P_test = np.linspace(P_low, P_high, 100)
    betas = np.array([surrogate.predict_flash(float(p), T, z)["beta"] for p in P_test])
    max_idx = np.argmax(betas)
    if betas[max_idx] > 0.95:
        return float(P_test[max_idx])
    return None  # beta never approaches 1


# ---------------------------------------------------------------------------
# 3. WinProp ground-truth scan (coarse grid)
# ---------------------------------------------------------------------------

def _make_single_flash_dat(P: float, T: float, z: np.ndarray, template_lines: List[str]) -> str:
    """Generate a WinProp DAT file with a single fixed-PT flash at given (P, T, z)."""
    lines_out = []
    in_envelope = False
    in_flash = False
    composition_injected = False
    i = 0
    while i < len(template_lines):
        line = template_lines[i]
        upper = line.strip().upper()

        # Detect section boundaries
        if upper.startswith("*ENVELOPE"):
            in_envelope = True
            i += 1
            continue
        if upper.startswith("*FLASH"):
            in_envelope = False
            in_flash = True
            i += 1
            continue
        if upper.startswith("**=-=-="):
            in_envelope = False
            in_flash = False
            # END marker: flush our FLASH section if not yet done
            if upper.startswith("**=-=-=     END") and not composition_injected:
                # shouldn't happen, but safety
                pass
            lines_out.append(line)
            i += 1
            continue

        # Skip everything inside ENVELOPE and FLASH sections (replaced below)
        if in_envelope:
            i += 1
            continue
        if in_flash:
            i += 1
            continue

        # Outside special sections: normal processing
        if upper.startswith("*PLOT"):
            i += 1
            continue

        if "*PRIMARY" in upper and not composition_injected:
            lines_out.append(line)
            for j in range(0, NC, 5):
                lines_out.append("   ".join(f"{v:.6f}" for v in z[j:j+5]))
            composition_injected = True
            i += 1
            # Skip original composition lines
            while i < len(template_lines) and not template_lines[i].strip().startswith("*"):
                i += 1
            continue

        lines_out.append(line)
        i += 1

    # Inject single fixed-PT flash using QNSS with zero deltas
    flash_section = [
        "*FLASH",
        '*LABEL ""',
        "*FEED *MIXED 1.0",
        "*KVALUE *INTERNAL",
        "*LEVEL 1",
        "*OUTPUT 1",
        "*TYPE *QNSS",
        f"*PRES {P:.2f}",
        f"*TEMP {T:.2f}",
        "*DELP 0.0",
        "*DELT 0.0",
        "*STEPP 1",
        "*STEPT 1",
        "",
    ]
    # Find **=-=-=     END and insert before it
    end_idx = None
    for idx, line in enumerate(lines_out):
        if line.strip().upper().startswith("**=-=-=     END"):
            end_idx = idx
            break
    if end_idx is not None:
        lines_out = lines_out[:end_idx] + flash_section + lines_out[end_idx:]
    else:
        lines_out.extend(flash_section)

    return "\n".join(lines_out)


def _parse_ptflash_out(out_text: str) -> Optional[Dict]:
    """Parse a single PT-flash result from WinProp .out file.

    WinProp QNSS output has two composition tables:
      Table 1 (before "Phase Mole %"): component  Feed  Phase01  Phase02
        — Phase01/Phase02 values are mole % within each phase.
      Table 2 (after "Phase Mole %"):  component  Feed  Phase01  Phase02
        — component distribution across phases (different basis).
    We use Table 1 for phase compositions and "Phase Mole %" for β.
    """
    beta = None
    x = np.zeros(NC)
    y = np.zeros(NC)

    # Extract vapour fraction
    # Two-phase: "Phase Mole %   liquid_pct   vapour_pct"
    m = re.search(r'Phase\s+Mole\s*%\s+([\d.]+)\s+([\d.]+)', out_text)
    if m:
        vapour_pct = float(m.group(2))
        beta = vapour_pct / 100.0
    else:
        # Single-phase: Phase Mole % shows 100 or 0 for the non-existent phase
        # Determine by "Phase Split:" label
        split_label = ""
        split_match = re.search(r'Phase\s+Split:\s*(\S+)', out_text)
        if split_match:
            split_label = split_match.group(1).strip()
        if split_label == "Liquid":
            beta = 0.0
        elif split_label == "Vapour":
            beta = 1.0
        else:
            # Fallback: check Phase Mole % value
            m = re.search(r'Phase\s+Mole\s*%\s+([\d.]+)', out_text)
            if m:
                beta = 0.0 if float(m.group(1)) > 99.0 else 1.0
            else:
                m = re.search(r'Vap(?:our)?\s+(?:Mole\s+)?Fraction\s*[=:]\s*([\d.]+)', out_text, re.IGNORECASE)
                if m:
                    beta = float(m.group(1))

    # Parse the FIRST composition table (before "Phase Mole %")
    # Strategy: find the "Phase Split:" marker, then parse until "Phase Mole %"
    split_marker = out_text.find("Phase Split:")
    mole_marker = out_text.find("Phase Mole %")
    if split_marker >= 0:
        table_text = out_text[split_marker:mole_marker] if mole_marker > split_marker else out_text[split_marker:]
    else:
        table_text = out_text[:mole_marker] if mole_marker > 0 else out_text

    # Detect single-phase vs two-phase
    is_single_phase = "Liquid-Vapour" not in out_text[split_marker:split_marker+100] if split_marker >= 0 else False

    liq_vals = []
    vap_vals = []
    in_header = False
    for line in table_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if "component" in lower or "feed" in lower:
            in_header = True
            continue
        if not in_header:
            continue
        if "Phase" in stripped or lower.startswith("---"):
            in_header = False
            break
        if "mass percent" in lower or "ln (fug" in lower:
            in_header = False
            break
        toks = stripped.split()
        nums = []
        for t in toks:
            try:
                nums.append(float(t))
            except ValueError:
                pass
        if is_single_phase:
            # Only 2 numeric cols: feed, Phase01
            if len(nums) >= 2:
                liq_vals.append(nums[1])
                vap_vals.append(nums[1])  # same as liquid for single-phase
        else:
            # 3 numeric cols: feed, Phase01 (liquid), Phase02 (vapour)
            if len(nums) >= 3:
                liq_vals.append(nums[1])
                vap_vals.append(nums[2])

    if len(liq_vals) >= NC:
        x = np.array(liq_vals[:NC], dtype=np.float64) / 100.0
        y = np.array(vap_vals[:NC], dtype=np.float64) / 100.0
    else:
        return None

    if beta is None:
        return None

    # Normalize to ensure sum=1
    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)
    x_sum = max(x.sum(), 1e-12)
    y_sum = max(y.sum(), 1e-12)
    if x_sum > 0:
        x = x / x_sum
    if y_sum > 0:
        y = y / y_sum

    return {"beta": beta, "x": x, "y": y}


def run_winprop_scan(
    z: np.ndarray,
    T: float,
    P_points: np.ndarray,
    template_path: Path = TEMPLATE_DAT,
    work_dir: Path = WORK_DIR,
) -> Dict[str, np.ndarray]:
    """Run WinProp EOS on a pressure grid for ground-truth comparison.

    Args:
        z: feed composition (15,)
        T: temperature (°C)
        P_points: array of pressures (kPa), typically 20-40 points
        template_path: path to fluid_15comp.dat template
        work_dir: working directory for temp files

    Returns:
        dict with: P, beta, x, y, converged (bool mask)
    """
    if not os.path.exists(WINPROP_EXE):
        print(f"WinProp not found at {WINPROP_EXE}, skipping ground truth.")
        return {"P": P_points, "beta": np.full_like(P_points, np.nan),
                "x": np.full((len(P_points), NC), np.nan),
                "y": np.full((len(P_points), NC), np.nan),
                "converged": np.zeros(len(P_points), dtype=bool)}

    with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
        template_lines = f.readlines()

    N = len(P_points)
    beta = np.full(N, np.nan)
    x = np.full((N, NC), np.nan)
    y = np.full((N, NC), np.nan)
    converged = np.zeros(N, dtype=bool)

    print(f"Running WinProp EOS on {N} pressure points...")
    for i, P in enumerate(tqdm(P_points, desc="WinProp P-scan")):
        dat_content = _make_single_flash_dat(P, T, z, template_lines)
        dat_path = work_dir / f"_phase_boundary_{i:03d}.dat"
        dat_path.write_text(dat_content, encoding="utf-8")

        try:
            result = subprocess.run(
                [WINPROP_EXE],
                cwd=str(work_dir),
                input=f"{dat_path}\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore",
                timeout=60.0,
            )
            out_path = Path(str(dat_path).replace(".dat", ".out"))
            if out_path.exists():
                out_text = out_path.read_text(encoding="utf-8", errors="ignore")
                parsed = _parse_ptflash_out(out_text)
                if parsed is not None:
                    beta[i] = parsed["beta"]
                    x[i] = parsed["x"]
                    y[i] = parsed["y"]
                    converged[i] = True
        except (subprocess.TimeoutExpired, PermissionError, OSError):
            pass

    n_conv = converged.sum()
    print(f"  WinProp converged: {n_conv}/{N} points")
    return {"P": P_points, "beta": beta, "x": x, "y": y, "converged": converged}


# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_continuity(beta: np.ndarray) -> Dict[str, float]:
    """Check β-P curve continuity: max jump, total variation, monotonicity violations."""
    dbeta = np.diff(beta)
    max_jump = float(np.max(np.abs(dbeta)))
    # Monotonicity: β should decrease as P increases (more liquid at higher P)
    # Check for positive jumps (β increases with P → violation)
    mono_violations = int(np.sum(dbeta > 1e-6))
    total_variation = float(np.sum(np.abs(dbeta)))
    return {
        "max_jump": max_jump,
        "monotonicity_violations": mono_violations,
        "total_variation": total_variation,
    }


def evaluate_boundary_deviation(
    P_pgae: np.ndarray, beta_pgae: np.ndarray,
    P_eos: np.ndarray, beta_eos: np.ndarray,
    converged: np.ndarray,
) -> Dict[str, float]:
    """Compare PGAE vs EOS on converged EOS points (interpolate PGAE to EOS P)."""
    mask = converged & ~np.isnan(beta_eos)
    if mask.sum() < 2:
        return {"beta_mae": float("nan"), "beta_rmse": float("nan"), "beta_max_err": float("nan")}
    beta_pgae_at_eos = np.interp(P_eos[mask], P_pgae, beta_pgae)
    err = np.abs(beta_pgae_at_eos - beta_eos[mask])
    return {
        "beta_mae": float(np.mean(err)),
        "beta_rmse": float(np.sqrt(np.mean(err ** 2))),
        "beta_max_err": float(np.max(err)),
    }


# ---------------------------------------------------------------------------
# 5. Main experiment & plotting
# ---------------------------------------------------------------------------

def run_phase_boundary_experiment(
    surrogate: PGAEFlashSurrogate,
    z_fixed: np.ndarray,
    T_fixed: float,
    z_label: str = "",
    P_min: float = 100.0,
    P_max: float = 50000.0,
    n_pgae: int = 500,
    n_eos: int = 30,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run full phase boundary continuity experiment.

    Args:
        surrogate: loaded PGAE model
        z_fixed: feed composition (15,)
        T_fixed: temperature (°C)
        z_label: label for the composition (for plot titles)
        P_min, P_max: pressure range (kPa)
        n_pgae: number of PGAE scan points (dense)
        n_eos: number of WinProp points (coarse)
        output_dir: directory for figures, skips saving if None

    Returns:
        dict with all metrics and results
    """
    print(f"\n{'='*60}")
    print(f"Phase Boundary Continuity Experiment")
    print(f"  T = {T_fixed} °C,  P ∈ [{P_min:.0f}, {P_max:.0f}] kPa")
    print(f"  z: {z_label}")
    print(f"{'='*60}")

    # --- PGAE dense scan ---
    print("\n[1/4] PGAE pressure scan (500 points)...")
    P_pgae = np.linspace(P_min, P_max, n_pgae)
    pgae_results = pgae_pressure_scan(surrogate, z_fixed, T_fixed, P_pgae)

    # --- Bubble & dew point search ---
    print("\n[2/4] Binary search for bubble/dew points...")
    bubble_p = find_bubble_point(surrogate, z_fixed, T_fixed, P_min, P_max)
    dew_p = find_dew_point(surrogate, z_fixed, T_fixed, P_min, P_max)
    print(f"  Bubble point (PGAE): {bubble_p:.1f} kPa" if bubble_p else "  No bubble point found")
    print(f"  Dew point (PGAE):   {dew_p:.1f} kPa" if dew_p else "  No dew point found")

    # --- WinProp ground truth ---
    print("\n[3/4] WinProp EOS ground truth...")
    P_eos = np.linspace(P_min, P_max, n_eos)
    eos_results = run_winprop_scan(z_fixed, T_fixed, P_eos)

    # --- Evaluate ---
    print("\n[4/4] Evaluating continuity metrics...")
    cont_metrics = evaluate_continuity(pgae_results["beta"])

    mask = eos_results["converged"]
    dev_metrics = {}
    if mask.sum() > 1:
        dev_metrics = evaluate_boundary_deviation(
            P_pgae, pgae_results["beta"],
            P_eos, eos_results["beta"], mask,
        )

    summary = {
        "T": T_fixed,
        "z": z_fixed,
        "P_range": [P_min, P_max],
        "bubble_point_pgae": bubble_p,
        "dew_point_pgae": dew_p,
        **cont_metrics,
        **dev_metrics,
    }

    print(f"\n  Max β jump:         {cont_metrics['max_jump']:.4e}")
    print(f"  Monotonicity violations: {cont_metrics['monotonicity_violations']}/{n_pgae-1}")
    if dev_metrics:
        print(f"  β MAE vs EOS:       {dev_metrics.get('beta_mae', float('nan')):.4e}")
    print(f"  Bubble point:       {bubble_p:.1f} kPa" if bubble_p else "  No bubble point")
    print(f"  Dew point:          {dew_p:.1f} kPa" if dew_p else "  No dew point")

    # --- Plot ---
    if output_dir is not None:
        # Create a safe filename suffix from the experiment label
        suffix = z_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        _plot_phase_boundary(
            P_pgae, pgae_results["beta"],
            P_eos, eos_results["beta"], eos_results["converged"],
            bubble_p, dew_p,
            T_fixed, z_label,
            output_dir, suffix,
        )
        _plot_phase_compositions(P_pgae, pgae_results["x"], pgae_results["y"],
                                 COMP_NAMES, T_fixed, z_label, output_dir, suffix)

    return {"pgae": pgae_results, "eos": eos_results, "summary": summary}


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def _plot_phase_boundary(
    P_pgae: np.ndarray,
    beta_pgae: np.ndarray,
    P_eos: np.ndarray,
    beta_eos: np.ndarray,
    eos_converged: np.ndarray,
    bubble_p: Optional[float],
    dew_p: Optional[float],
    T_fixed: float,
    z_label: str,
    out_dir: Path,
    suffix: str = "",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Main β-P curve
    ax1.plot(P_pgae, beta_pgae, "b-", linewidth=1.5, label="PGAE", zorder=3)

    # EOS ground truth (converged points only)
    mask = eos_converged & ~np.isnan(beta_eos)
    if mask.sum() > 0:
        ax1.scatter(P_eos[mask], beta_eos[mask], c="red", s=35, marker="o",
                    edgecolors="darkred", linewidths=0.5, zorder=4, label="WinProp EOS")

    # Bubble/dew points
    if bubble_p is not None:
        ax1.axvline(bubble_p, color="green", linestyle="--", alpha=0.7,
                    label=f"Bubble P={bubble_p:.0f} kPa")
    if dew_p is not None:
        ax1.axvline(dew_p, color="orange", linestyle="--", alpha=0.7,
                    label=f"Dew P={dew_p:.0f} kPa")

    ax1.set_xlabel("Pressure (kPa)")
    ax1.set_ylabel("β (Vapour Mole Fraction)")
    ax1.set_title(f"β–P Phase Boundary  (T={T_fixed}°C)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Zoom: near-critical / phase boundary region
    # Find interesting region: where β changes from 0 to 1
    trans_mask = (beta_pgae > 0.01) & (beta_pgae < 0.99)
    if trans_mask.sum() > 10:
        P_zoom_min = P_pgae[trans_mask].min()
        P_zoom_max = P_pgae[trans_mask].max()
        pad = (P_zoom_max - P_zoom_min) * 0.15
        zoom_mask = (P_pgae >= P_zoom_min - pad) & (P_pgae <= P_zoom_max + pad)

        ax2.plot(P_pgae[zoom_mask], beta_pgae[zoom_mask], "b-", linewidth=1.8, zorder=3)
        if mask.sum() > 0:
            eos_zoom = mask & (P_eos >= P_zoom_min - pad) & (P_eos <= P_zoom_max + pad)
            if eos_zoom.sum() > 0:
                ax2.scatter(P_eos[eos_zoom], beta_eos[eos_zoom], c="red", s=50,
                           marker="o", edgecolors="darkred", linewidths=0.5, zorder=4)
        if bubble_p is not None and P_zoom_min - pad <= bubble_p <= P_zoom_max + pad:
            ax2.axvline(bubble_p, color="green", linestyle="--", alpha=0.7)
        if dew_p is not None and P_zoom_min - pad <= dew_p <= P_zoom_max + pad:
            ax2.axvline(dew_p, color="orange", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Pressure (kPa)")
        ax2.set_ylabel("β")
        ax2.set_title(f"Zoom: Phase Transition Region")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No phase transition in range", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Zoom (N/A)")

    fname = f"phase_boundary_beta_P_{suffix}.png" if suffix else "phase_boundary_beta_P.png"
    fig.suptitle(f"PGAE Phase Boundary Continuity  |  {z_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=220)
    plt.close(fig)


def _plot_phase_compositions(
    P: np.ndarray, x: np.ndarray, y: np.ndarray,
    comp_names: List[str], T_fixed: float, z_label: str, out_dir: Path,
    suffix: str = "",
) -> None:
    """Plot x and y compositions vs pressure (key components only)."""
    # Select key components: CH4 (3), CO2 (2), C2H6 (4), C3H8 (5), C10+ (0), N2 (1)
    key_idx = [3, 2, 4, 5, 0, 1]  # CH4, CO2, C2H6, C3H8, C10+, N2
    key_names = [comp_names[i] for i in key_idx]
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_idx)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for k, (idx, name, c) in enumerate(zip(key_idx, key_names, colors)):
        ax1.plot(P, x[:, idx], color=c, linewidth=1.2, label=name, alpha=0.85)
        ax2.plot(P, y[:, idx], color=c, linewidth=1.2, label=name, alpha=0.85)

    ax1.set_xlabel("Pressure (kPa)")
    ax1.set_ylabel("Liquid Mole Fraction xᵢ")
    ax1.set_title("Liquid Composition vs Pressure")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Pressure (kPa)")
    ax2.set_ylabel("Vapour Mole Fraction yᵢ")
    ax2.set_title("Vapour Composition vs Pressure")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fname = f"phase_boundary_compositions_{suffix}.png" if suffix else "phase_boundary_compositions.png"
    fig.suptitle(f"Phase Composition Continuity  (T={T_fixed}°C, {z_label})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Master entry: run on representative compositions
# ---------------------------------------------------------------------------

def main() -> None:
    config = PGAEConfig()
    if not config.best_checkpoint_path.exists():
        print(f"Checkpoint not found: {config.best_checkpoint_path}")
        print("Please run train.py first.")
        return

    print("Loading PGAE surrogate model...")
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)

    # Pick representative compositions from the dataset
    df = pd.read_csv(config.dataset_path)
    z_cols = [f"z{i}" for i in range(1, NC + 1)]

    # Strategy: pick compositions with different characteristics
    all_z = df[z_cols].to_numpy(dtype=np.float64)

    # 1. Median CH4 composition
    ch4_frac = all_z[:, 3]  # CH4 is index 3
    med_idx = np.argmin(np.abs(ch4_frac - np.median(ch4_frac)))
    z_typical = all_z[med_idx].copy()

    # 2. High CH4 (gas-rich)
    high_ch4_idx = np.argmax(ch4_frac)
    z_gas_rich = all_z[high_ch4_idx].copy()

    # 3. Low CH4 (oil-rich)
    low_ch4_idx = np.argmin(ch4_frac)
    z_oil_rich = all_z[low_ch4_idx].copy()

    experiments = [
        (z_typical, 100.0, f"Typical (CH4={z_typical[3]:.3f})"),
        (z_typical, 200.0, f"High-T (CH4={z_typical[3]:.3f})"),
        (z_gas_rich, 150.0, f"Gas-rich (CH4={z_gas_rich[3]:.3f})"),
        (z_oil_rich, 100.0, f"Oil-rich (CH4={z_oil_rich[3]:.3f})"),
    ]

    all_summaries = []
    for z, T, label in experiments:
        result = run_phase_boundary_experiment(
            surrogate, z, T, z_label=label,
            output_dir=config.fig_phase_boundary_dir,
        )
        all_summaries.append({"label": label, **result["summary"]})

    # Save summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_path = config.metric_dir / "phase_boundary_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nPhase boundary summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
