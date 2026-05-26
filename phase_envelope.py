"""Part 11: Phase Envelope Automatic Reconstruction.

For a fixed feed composition z, trace the full P-T phase envelope:
  - PGAE: binary search for bubble/dew point at each T
  - WinProp EOS: *ENVELOPE ground truth
  - Critical point estimation, overlap metrics, publication-quality plots
"""
from __future__ import annotations

import json
import os
import re
import subprocess
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
NC = 15
COMP_NAMES = ["C10+", "N2", "CO2", "CH4", "C2H6", "C3H8", "IC4", "NC4", "IC5", "NC5",
              "FC6", "FC7", "FC8", "FC9", "FC10"]


# =============================================================================
# 1. PGAE Phase Envelope via P-scan (robust to model extrapolation limits)
# =============================================================================

def _pgae_beta_scan(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    T_C: float,
    P_scan: np.ndarray,
) -> np.ndarray:
    """Compute β at each pressure in P_scan for a fixed (z, T)."""
    betas = np.zeros(len(P_scan))
    for i, P in enumerate(P_scan):
        betas[i] = surrogate.predict_flash(float(P), T_C, z)["beta"]
    return betas


def _extract_saturation_pressures(
    P_scan: np.ndarray,
    betas: np.ndarray,
    beta_lo: float = 0.01,
    beta_hi: float = 0.99,
) -> Tuple[Optional[float], Optional[float]]:
    """Extract bubble and dew pressures from a β-P scan.

    Enhanced with three detection strategies (fallback chain):
      1. Direct crossing: β crosses beta_lo/beta_hi threshold.
      2. Gradient inflection: β begins to deviate from plateau.
      3. Two-phase peak: middle of the two-phase dome when β∈[0.01,0.99] region exists.

    Bubble = liquid saturation (high-P side of dome). Dew = vapour saturation (low-P).
    """
    beta_max = betas.max()
    beta_min = betas.min()
    max_idx = int(np.argmax(betas))
    n = len(P_scan)

    # --- Strategy 1: Direct threshold crossing ---
    above_lo = betas > beta_lo
    above_hi = betas > beta_hi

    # Bubble: highest P where β crosses above beta_lo
    bubble_P = None
    for i in range(n - 1, 0, -1):
        if above_lo[i] and not above_lo[i - 1]:
            frac = (beta_lo - betas[i - 1]) / (betas[i] - betas[i - 1] + 1e-15)
            bubble_P = float(P_scan[i - 1] + frac * (P_scan[i] - P_scan[i - 1]))
            break
    if bubble_P is None and above_lo[0] and not above_lo.any():
        bubble_P = float(P_scan[0])

    # Dew: lowest P where β crosses above beta_hi
    dew_P = None
    if beta_max >= beta_hi:
        for i in range(0, n - 1):
            if above_hi[i + 1] and not above_hi[i]:
                frac = (beta_hi - betas[i]) / (betas[i + 1] - betas[i] + 1e-15)
                dew_P = float(P_scan[i] + frac * (P_scan[i + 1] - P_scan[i]))
                break
        if dew_P is None and above_hi[-1] and not above_hi.any():
            dew_P = float(P_scan[-1])
        if dew_P is None and beta_max >= beta_hi:
            # β≥beta_hi at lowest P scanned → dew is at or below P_min
            dew_P = float(P_scan[0])

    # --- Strategy 2: Gradient inflection (for shallow transitions) ---
    if bubble_P is None and max_idx < n - 1:
        # Look for large negative dβ/d(log P) indicating liquid→two-phase transition
        logP = np.log10(P_scan + 1e-10)
        dbeta_dlogP = np.gradient(betas, logP)
        # Bubble: sharp β decrease at high P
        for i in range(max_idx + 1, min(max_idx + 15, n)):
            if betas[i] < 0.05 and dbeta_dlogP[i] < -0.1:
                bubble_P = float(P_scan[i])
                break

    if dew_P is None and max_idx > 0:
        logP = np.log10(P_scan + 1e-10)
        dbeta_dlogP = np.gradient(betas, logP)
        # Dew: sharp β decrease at low P (entering two-phase from vapour side)
        for i in range(max(0, max_idx - 15), max_idx):
            if betas[i] > 0.05 and dbeta_dlogP[i] < -0.2:
                dew_P = float(P_scan[i])
                break

    # --- Strategy 3: Two-phase region bounds ---
    two_phase_mask = (betas > 0.005) & (betas < 0.995)
    if two_phase_mask.any():
        tp_indices = np.where(two_phase_mask)[0]
        if bubble_P is None and len(tp_indices) > 0:
            # Bubble = highest P in two-phase zone
            bubble_P = float(P_scan[tp_indices[-1]])
        if dew_P is None and len(tp_indices) > 0:
            # Dew = lowest P in two-phase zone
            dew_P = float(P_scan[tp_indices[0]])

    # --- Strategy 4: Pseudo-points from max-β ---
    if dew_P is None and beta_max > 0.40:
        dew_P = float(P_scan[max_idx])
    if bubble_P is None and beta_max > 0.40:
        bubble_P = float(P_scan[max_idx])

    return bubble_P, dew_P


def pgae_phase_envelope(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    T_range: np.ndarray,
    P_min: float = 1.0,
    P_max: float = 50000.0,
    n_P: int = 80,
) -> Dict[str, np.ndarray]:
    """Generate full P-T phase envelope via PGAE P-scan.

    For each T, scans n_P log-spaced pressures and extracts saturation
    pressures from the β-P curve. This is more robust than binary search
    when the model does not cleanly reach β=0 or β=1 within the P range.

    Args:
        surrogate: loaded PGAE model
        z: feed composition (15,)
        T_range: array of temperatures (°C) for envelope tracing
        P_min, P_max: pressure bounds (kPa)
        n_P: number of pressure points per temperature

    Returns:
        dict with: T_bubble, P_bubble, T_dew, P_dew, T_crit, P_crit,
                   beta_curves (T×P grid), P_scan
    """
    P_scan = np.logspace(np.log10(P_min), np.log10(P_max), n_P)
    T_bubble, P_bubble = [], []
    T_dew, P_dew = [], []
    all_betas = np.zeros((len(T_range), n_P))

    for ti, T in enumerate(tqdm(T_range, desc="PGAE envelope", leave=False)):
        betas = _pgae_beta_scan(surrogate, z, T, P_scan)
        all_betas[ti] = betas
        bp, dp = _extract_saturation_pressures(P_scan, betas)

        if bp is not None:
            T_bubble.append(T)
            P_bubble.append(bp)
        if dp is not None:
            T_dew.append(T)
            P_dew.append(dp)

    T_bubble_arr = np.array(T_bubble)
    P_bubble_arr = np.array(P_bubble)
    T_dew_arr = np.array(T_dew)
    P_dew_arr = np.array(P_dew)

    # Estimate critical point: where bubble and dew curves meet
    T_crit, P_crit = _estimate_critical_point(T_bubble_arr, P_bubble_arr, T_dew_arr, P_dew_arr)

    return {
        "T_bubble": T_bubble_arr,
        "P_bubble": P_bubble_arr,
        "T_dew": T_dew_arr,
        "P_dew": P_dew_arr,
        "T_crit": T_crit,
        "P_crit": P_crit,
        "beta_curves": all_betas,
        "P_scan": P_scan,
    }


def _estimate_critical_point(
    T_b: np.ndarray, P_b: np.ndarray,
    T_d: np.ndarray, P_d: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate critical point as the intersection of bubble and dew curves.

    Uses three strategies (best result wins):
      1. Power-law gap extrapolation: ΔP ∝ (T_c − T)^β (β≈2 near critical).
      2. Quadratic gap fit with intersection solve.
      3. Direct convergence: bubble/dew P difference at closest approach.
    """
    if len(T_b) < 2 or len(T_d) < 2:
        # Single-curve fallback: use max T/P of combined set
        if len(T_b) >= 2:
            return float(T_b.max()), float(P_b.max())
        if len(T_d) >= 2:
            return float(T_d.max()), float(P_d.max())
        return None, None

    candidates = []  # list of (T_crit, P_crit, quality_score)

    # --- Strategy 1: Power-law gap extrapolation ---
    # Combine bubble and dew sorted by T, fit ΔP vs T
    T_all = np.concatenate([T_b, T_d])
    P_all = np.concatenate([P_b, P_d])
    order = np.argsort(T_all)
    T_sorted = T_all[order]
    P_sorted = P_all[order]

    # Estimate gap by computing std(P) in T bins, or ΔP between upper/lower envelope
    if len(T_sorted) >= 6:
        # Build upper and lower P envelopes
        n_bins = min(15, len(T_sorted) // 2)
        bin_edges = np.linspace(T_sorted.min(), T_sorted.max(), n_bins + 1)
        T_mid = []
        P_upper, P_lower = [], []
        for k in range(n_bins):
            mask = (T_sorted >= bin_edges[k]) & (T_sorted < bin_edges[k + 1])
            if mask.sum() >= 2:
                T_mid.append((bin_edges[k] + bin_edges[k + 1]) / 2)
                P_upper.append(P_sorted[mask].max())
                P_lower.append(P_sorted[mask].min())

        if len(T_mid) >= 4:
            T_mid = np.array(T_mid)
            gap = np.array(P_upper) - np.array(P_lower)
            # gap ∝ (T_c - T)^β → gap^(1/β) ∝ (T_c - T)
            # Use β=2 (mean-field critical exponent)
            sqrt_gap = np.sqrt(np.clip(gap, 0, None))
            valid = sqrt_gap > 0

            if valid.sum() >= 3 and T_mid[valid].max() - T_mid[valid].min() > 5:
                coeffs = np.polyfit(T_mid[valid], sqrt_gap[valid], 1)
                T_crit = -coeffs[1] / coeffs[0]  # sqrt_gap = 0 at T_crit
                P_crit = float(np.interp(T_crit, T_mid, (np.array(P_upper) + np.array(P_lower)) / 2))
                T_range = T_mid[valid].max() - T_mid[valid].min()
                quality = float(gap.min() / (gap.max() + 1e-10))
                if T_crit > T_mid[valid].max() - 10 and T_crit < T_mid[valid].max() + 100:
                    candidates.append((T_crit, P_crit, 0.0))  # lowest quality = best

    # --- Strategy 2: Quadratic fit to overlapping region ---
    T_min = max(T_b.min(), T_d.min())
    T_max = min(T_b.max(), T_d.max())
    if T_max > T_min + 5:
        T_common = np.linspace(T_min, T_max, 100)
        P_b_interp = np.interp(T_common, T_b, P_b)
        P_d_interp = np.interp(T_common, T_d, P_d)
        gap = np.abs(P_b_interp - P_d_interp)

        if len(gap) > 5:
            coeffs = np.polyfit(T_common, gap, 2)
            if abs(coeffs[0]) > 1e-12:
                T_crit = -coeffs[1] / (2 * coeffs[0])
                P_crit = float(np.interp(T_crit, T_common, (P_b_interp + P_d_interp) / 2))
                # Quality: min gap as fraction of max gap
                quality = 1.0 - gap.min() / (gap.max() + 1e-10)
                candidates.append((T_crit, P_crit, quality))
            else:
                min_idx = np.argmin(gap)
                candidates.append((float(T_common[min_idx]),
                                   float((P_b_interp[min_idx] + P_d_interp[min_idx]) / 2), 0.5))

    # --- Strategy 3: Nearest approach (direct) ---
    if len(T_b) > 0 and len(T_d) > 0:
        T_min = max(T_b.min(), T_d.min())
        T_max = min(T_b.max(), T_d.max())
        if T_max > T_min:
            T_common = np.linspace(T_min, T_max, 100)
            P_b_i = np.interp(T_common, T_b, P_b)
            P_d_i = np.interp(T_common, T_d, P_d)
            gap = np.abs(P_b_i - P_d_i)
            min_idx = np.argmin(gap)
            candidates.append((float(T_common[min_idx]),
                               float((P_b_i[min_idx] + P_d_i[min_idx]) / 2), 0.9))

    if not candidates:
        return None, None

    # Pick best: prefer Strategy 2 (quadratic) if reasonable, else Strategy 3
    candidates.sort(key=lambda x: x[2])  # lowest quality = best
    T_crit, P_crit, _ = candidates[0]

    # Sanity bounds
    T_data_max = max(T_b.max(), T_d.max())
    if T_crit < T_data_max - 30 or T_crit > T_data_max + 100:
        # Use the last candidate that's in bounds
        for Tc, Pc, q in candidates:
            if Tc >= T_data_max - 30 and Tc <= T_data_max + 100:
                return Tc, Pc
        # Fallback: max T with corresponding P
        return float(T_data_max), float(np.interp(T_data_max,
            np.concatenate([T_b, T_d]), np.concatenate([P_b, P_d])))

    return float(T_crit), float(P_crit)


# =============================================================================
# 3. WinProp EOS Envelope (Ground Truth)
# =============================================================================

def _make_envelope_dat(
    z: np.ndarray,
    T_min_C: float,
    T_max_C: float,
    P_max_kPa: float,
    template_lines: List[str],
) -> str:
    """Generate a WinProp DAT file with only *ENVELOPE section.

    Replaces the composition and envelope settings while preserving EOS params.
    """
    lines_out = []
    in_envelope = False
    in_flash = False
    comp_injected = False
    envelope_injected = False
    i = 0
    while i < len(template_lines):
        line = template_lines[i]
        upper = line.strip().upper()

        if upper.startswith("*ENVELOPE"):
            in_envelope = True
            i += 1
            continue
        if upper.startswith("*FLASH"):
            in_envelope = False
            in_flash = True
            i += 1
            continue
        if upper.startswith("**=-=-=     END"):
            # Inject envelope before END if not done
            if not envelope_injected:
                lines_out.append("*ENVELOPE")
                lines_out.append("*LABEL    ''")
                lines_out.append("*FEED  *MIXED 1.0")
                lines_out.append("*KVALUE  *INTERNAL")
                lines_out.append("*OUTPUT 1")
                lines_out.append("*STABCHECK *YES")
                lines_out.append("*TRACEBOTH *NO")
                lines_out.append("*X-AXIS     *TEMP")
                lines_out.append(f"*RANGT     {T_min_C:.1f}    {T_max_C:.1f}")
                lines_out.append("*Y-AXIS     *PRES")
                lines_out.append(f"*RANGP     0.1    {P_max_kPa:.1f}")
                lines_out.append("*PRES    10000.0")
                lines_out.append("*TEMP    50.0")
                lines_out.append("*MAXSP   199")
                lines_out.append("*STEPDIR   0.1")
                lines_out.append("*RANGFV    -10.0    10.0")
                lines_out.append("*NTAB   0")
                lines_out.append("*NQUALITY    2")
                lines_out.append("*QUALITY")
                lines_out.append("0.0   1.0")
                lines_out.append("")
                envelope_injected = True
            lines_out.append(line)
            in_envelope = False
            in_flash = False
            i += 1
            continue

        if in_envelope:
            i += 1
            continue
        if in_flash:
            i += 1
            continue

        if upper.startswith("*PLOT"):
            i += 1
            continue

        if "*PRIMARY" in upper and not comp_injected:
            lines_out.append(line)
            for j in range(0, NC, 5):
                lines_out.append("   ".join(f"{v:.6f}" for v in z[j:j+5]))
            comp_injected = True
            i += 1
            while i < len(template_lines) and not template_lines[i].strip().startswith("*"):
                i += 1
            continue

        lines_out.append(line)
        i += 1

    return "\n".join(lines_out)


def _parse_envelope_out(out_text: str) -> Dict[str, np.ndarray]:
    """Parse WinProp *ENVELOPE output.

    WinProp traces the envelope by stepping along quality=const lines.
    With *NQUALITY 2, two sections are produced:
      - Section 1: quality=0 (bubble / liquid saturation)
      - Section 2: quality=1 (dew / vapour saturation)

    Each section contains step lines like:
      IT NV ST   P,kPa  T,deg C   ZX   ZY  ln(K)...

    We extract P and T from every line that starts with integers (IT NV ST).
    """
    lines = out_text.split("\n")
    sections = []  # list of (T_list, P_list) for each section
    current_section = []
    in_construction = False

    for line in lines:
        stripped = line.strip()

        # Detect start of envelope construction section
        if "Two-phase phase diagram construction" in stripped:
            if current_section:
                sections.append(current_section)
                current_section = []
            in_construction = True
            continue

        # End of construction: next major section starts (another construction or end)
        if in_construction and stripped.startswith("**=-=-="):
            in_construction = False
            if current_section:
                sections.append(current_section)
                current_section = []
            continue

        if not in_construction:
            continue

        # Skip blank lines within construction
        if not stripped:
            continue

        # Parse step lines: "  IT NV ST   P,kPa  T,deg C  ZX  ZY  ..."
        toks = stripped.split()
        if len(toks) < 5:
            continue

        # First three tokens should be integers (IT, NV, ST)
        try:
            int(toks[0])
            int(toks[1])
            int(toks[2])
        except (ValueError, IndexError):
            continue

        # Next tokens should be floats: P, T, ZX, ZY
        try:
            P = float(toks[3])
            T = float(toks[4])
        except (ValueError, IndexError):
            continue

        # Validate: P and T should be within reasonable ranges
        if not (0.01 <= P <= 200000 and -100 <= T <= 500):
            continue

        # Check that ZX, ZY are also floats (additional validation)
        try:
            float(toks[5])
            float(toks[6])
        except (ValueError, IndexError):
            continue

        current_section.append((T, P))

    # Don't forget last section
    if current_section:
        sections.append(current_section)

    if not sections:
        # Fallback: try to find P,T pairs from the old table format
        return _parse_envelope_out_fallback(out_text)

    # Extract bubble and dew curves from sections
    # Section 0 = quality=0 (bubble curve), Section 1 = quality=1 (dew curve)
    T_bubble, P_bubble = [], []
    T_dew, P_dew = [], []
    T_all, P_all = [], []

    for si, sec in enumerate(sections):
        if len(sec) < 2:
            continue
        T_arr = np.array([p[0] for p in sec])
        P_arr = np.array([p[1] for p in sec])

        # Sort by T for consistency
        order = np.argsort(T_arr)
        T_sorted = T_arr[order]
        P_sorted = P_arr[order]

        T_all.extend(T_sorted)
        P_all.extend(P_sorted)

        if si == 0:  # quality=0 → bubble
            T_bubble = T_sorted.tolist()
            P_bubble = P_sorted.tolist()
        elif si == 1:  # quality=1 → dew
            T_dew = T_sorted.tolist()
            P_dew = P_sorted.tolist()

    # Estimate critical point from max T of bubble or max P of envelope
    T_crit, P_crit = None, None
    if T_bubble and P_bubble:
        # Critical point is at max T on the envelope
        max_idx = np.argmax(T_all)
        T_crit = float(T_all[max_idx])
        P_crit = float(P_all[max_idx])

    return {
        "T_all": np.array(T_all),
        "P_all": np.array(P_all),
        "T_bubble": np.array(T_bubble),
        "P_bubble": np.array(P_bubble),
        "T_dew": np.array(T_dew),
        "P_dew": np.array(P_dew),
        "T_crit": T_crit,
        "P_crit": P_crit,
    }


def _parse_envelope_out_fallback(out_text: str) -> Dict[str, np.ndarray]:
    """Fallback parser: try to find a table with Temp/Pres columns."""
    T_vals, P_vals = [], []
    lines = out_text.split("\n")
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        if "Temp" in stripped and "Pres" in stripped:
            in_table = True
            continue
        if in_table:
            toks = stripped.split()
            if len(toks) >= 2:
                try:
                    T_vals.append(float(toks[0]))
                    P_vals.append(float(toks[1]))
                except ValueError:
                    in_table = False

    T_arr = np.array(T_vals)
    P_arr = np.array(P_vals)
    return {
        "T_all": T_arr, "P_all": P_arr,
        "T_bubble": T_arr, "P_bubble": P_arr,
        "T_dew": np.array([]), "P_dew": np.array([]),
        "T_crit": None, "P_crit": None,
    }


def run_winprop_envelope(
    z: np.ndarray,
    T_min_C: float = -50.0,
    T_max_C: float = 200.0,
    P_max_kPa: float = 50000.0,
    template_path: Path = TEMPLATE_DAT,
    work_dir: Path = WORK_DIR,
) -> Optional[Dict[str, np.ndarray]]:
    """Run WinProp *ENVELOPE to get ground-truth phase envelope.

    Returns None if WinProp is not available or fails.
    """
    if not os.path.exists(WINPROP_EXE):
        print(f"WinProp not found at {WINPROP_EXE}")
        return None

    with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
        template_lines = f.readlines()

    dat_content = _make_envelope_dat(z, T_min_C, T_max_C, P_max_kPa, template_lines)
    dat_path = work_dir / "_phase_envelope_temp.dat"
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
            timeout=120.0,
        )
        out_path = Path(str(dat_path).replace(".dat", ".out"))
        if out_path.exists():
            out_text = out_path.read_text(encoding="utf-8", errors="ignore")
            parsed = _parse_envelope_out(out_text)
            if len(parsed["T_all"]) > 0:
                print(f"  WinProp envelope: {len(parsed['T_all'])} points, "
                      f"T_crit={parsed['T_crit']:.1f}°C" if parsed['T_crit'] else "")
                return parsed
    except (subprocess.TimeoutExpired, PermissionError, OSError) as e:
        print(f"  WinProp envelope failed: {e}")

    return None


# =============================================================================
# 4. Comparison & Metrics
# =============================================================================

def compute_envelope_overlap(
    pgae_envelope: Dict[str, np.ndarray],
    eos_envelope: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute quantitative overlap metrics between PGAE and EOS envelopes.

    Uses interpolation of PGAE envelope onto EOS envelope points.
    """
    metrics = {}

    # Critical point deviation
    if pgae_envelope.get("T_crit") and eos_envelope.get("T_crit"):
        T_crit_pgae = pgae_envelope["T_crit"]
        P_crit_pgae = pgae_envelope["P_crit"]
        T_crit_eos = eos_envelope["T_crit"]
        P_crit_eos = eos_envelope["P_crit"]
        metrics["T_crit_deviation"] = float(abs(T_crit_pgae - T_crit_eos))
        metrics["P_crit_deviation"] = float(abs(P_crit_pgae - P_crit_eos))
        metrics["T_crit_relative_error"] = float(abs(T_crit_pgae - T_crit_eos) / (abs(T_crit_eos) + 1e-10))
        metrics["P_crit_relative_error"] = float(abs(P_crit_pgae - P_crit_eos) / (abs(P_crit_eos) + 1e-10))

    # Bubble curve MAE
    if len(pgae_envelope["T_bubble"]) > 2 and len(eos_envelope["T_bubble"]) > 2:
        T_b_eos = eos_envelope["T_bubble"]
        P_b_eos = eos_envelope["P_bubble"]
        T_b_pgae = pgae_envelope["T_bubble"]
        P_b_pgae = pgae_envelope["P_bubble"]

        # Interpolate PGAE to EOS T points (within overlapping T range)
        T_overlap_min = max(T_b_eos.min(), T_b_pgae.min())
        T_overlap_max = min(T_b_eos.max(), T_b_pgae.max())
        if T_overlap_max > T_overlap_min:
            mask = (T_b_eos >= T_overlap_min) & (T_b_eos <= T_overlap_max)
            P_b_pgae_interp = np.interp(T_b_eos[mask], T_b_pgae, P_b_pgae)
            err = np.abs(P_b_pgae_interp - P_b_eos[mask])
            metrics["bubble_P_mae"] = float(np.mean(err))
            metrics["bubble_P_rmse"] = float(np.sqrt(np.mean(err ** 2)))
            metrics["bubble_P_max_err"] = float(np.max(err))
            metrics["bubble_n_points"] = int(mask.sum())

    # Dew curve MAE (same approach)
    if len(pgae_envelope["T_dew"]) > 2 and len(eos_envelope["T_dew"]) > 2:
        T_d_eos = eos_envelope["T_dew"]
        P_d_eos = eos_envelope["P_dew"]
        T_d_pgae = pgae_envelope["T_dew"]
        P_d_pgae = pgae_envelope["P_dew"]

        T_overlap_min = max(T_d_eos.min(), T_d_pgae.min())
        T_overlap_max = min(T_d_eos.max(), T_d_pgae.max())
        if T_overlap_max > T_overlap_min:
            mask = (T_d_eos >= T_overlap_min) & (T_d_eos <= T_overlap_max)
            P_d_pgae_interp = np.interp(T_d_eos[mask], T_d_pgae, P_d_pgae)
            err = np.abs(P_d_pgae_interp - P_d_eos[mask])
            metrics["dew_P_mae"] = float(np.mean(err))
            metrics["dew_P_rmse"] = float(np.sqrt(np.mean(err ** 2)))
            metrics["dew_P_max_err"] = float(np.max(err))
            metrics["dew_n_points"] = int(mask.sum())

    return metrics


# =============================================================================
# 5. Publication-quality plots
# =============================================================================

def plot_phase_envelope_comparison(
    pgae_envelope: Dict[str, np.ndarray],
    eos_envelope: Optional[Dict[str, np.ndarray]],
    z: np.ndarray,
    z_label: str,
    output_dir: Path,
) -> None:
    """Plot PGAE vs EOS phase envelope in P-T space."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # PGAE envelope
    if len(pgae_envelope["T_bubble"]) > 0:
        ax.plot(pgae_envelope["T_bubble"], pgae_envelope["P_bubble"],
                "b-", linewidth=2.0, label="PGAE Bubble", alpha=0.85)
    if len(pgae_envelope["T_dew"]) > 0:
        ax.plot(pgae_envelope["T_dew"], pgae_envelope["P_dew"],
                "b--", linewidth=2.0, label="PGAE Dew", alpha=0.85)
    if pgae_envelope.get("T_crit"):
        T_c, P_c = pgae_envelope["T_crit"], pgae_envelope["P_crit"]
        if T_c is not None and P_c is not None:
            ax.plot(T_c, P_c, "bo", markersize=10, markeredgewidth=2,
                   markeredgecolor="darkblue", label=f"PGAE Crit ({T_c:.0f}°C, {P_c:.0f} kPa)")

    # EOS envelope
    if eos_envelope is not None and len(eos_envelope.get("T_all", [])) > 0:
        ax.plot(eos_envelope["T_all"], eos_envelope["P_all"],
                "r-", linewidth=1.5, label="WinProp EOS", alpha=0.7, zorder=2)
    if eos_envelope and eos_envelope.get("T_crit"):
        T_c_eos, P_c_eos = eos_envelope["T_crit"], eos_envelope["P_crit"]
        if T_c_eos is not None and P_c_eos is not None:
            ax.plot(T_c_eos, P_c_eos, "r*", markersize=14, markeredgewidth=1.5,
                   markeredgecolor="darkred", label=f"EOS Crit ({T_c_eos:.0f}°C, {P_c_eos:.0f} kPa)")

    # Fill the two-phase region (between bubble and dew)
    if len(pgae_envelope["T_bubble"]) > 2 and len(pgae_envelope["T_dew"]) > 2:
        T_b, P_b = pgae_envelope["T_bubble"], pgae_envelope["P_bubble"]
        T_d, P_d = pgae_envelope["T_dew"], pgae_envelope["P_dew"]
        T_common = np.linspace(
            max(T_b.min(), T_d.min()),
            min(T_b.max(), T_d.max()),
            200,
        )
        P_b_interp = np.interp(T_common, T_b, P_b)
        P_d_interp = np.interp(T_common, T_d, P_d)
        ax.fill_between(T_common, P_b_interp, P_d_interp, alpha=0.08, color="blue")

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Pressure (kPa)", fontsize=12)
    ax.set_title(f"P-T Phase Envelope — {z_label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=max(-60, ax.get_xlim()[0]))
    ax.set_ylim(bottom=0)

    # Add composition annotation
    ch4_pct = z[3] * 100
    c2_pct = z[4] * 100
    c10_pct = z[0] * 100
    textstr = f"CH₄={ch4_pct:.1f}%, C₂H₆={c2_pct:.1f}%, C₁₀₊={c10_pct:.1f}%"
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    safe_label = z_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("₄", "4").replace("₂", "2").replace("₁", "1").replace("₀", "0").replace("%", "pct")
    fig.savefig(output_dir / f"phase_envelope_PT_{safe_label}.png", dpi=220)
    plt.close(fig)


def plot_beta_heatmap(
    pgae_envelope: Dict[str, np.ndarray],
    z_label: str,
    output_dir: Path,
) -> None:
    """Plot β(P, T) heatmap showing the full two-phase region from PGAE."""
    beta_curves = pgae_envelope.get("beta_curves")
    P_scan = pgae_envelope.get("P_scan")
    if beta_curves is None or P_scan is None:
        return

    T_range = np.linspace(40, 180, beta_curves.shape[0])

    fig, ax = plt.subplots(figsize=(9, 7))
    X, Y = np.meshgrid(T_range, P_scan)
    im = ax.pcolormesh(X, Y, beta_curves.T, shading="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    # Overlay extracted bubble/dew curves
    if len(pgae_envelope["T_bubble"]) > 0:
        ax.plot(pgae_envelope["T_bubble"], pgae_envelope["P_bubble"],
                "k-", linewidth=2.5, label="PGAE Bubble (β→0)")
    if len(pgae_envelope["T_dew"]) > 0:
        ax.plot(pgae_envelope["T_dew"], pgae_envelope["P_dew"],
                "k--", linewidth=2.5, label="PGAE Dew/Vapour Sat.")

    if pgae_envelope.get("T_crit"):
        T_c, P_c = pgae_envelope["T_crit"], pgae_envelope["P_crit"]
        if T_c is not None:
            ax.plot(T_c, P_c, "ko", markersize=10, label=f"Est. Crit ({T_c:.0f}°C)")

    cbar = fig.colorbar(im, ax=ax, label="β (Vapour Mole Fraction)")
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Pressure (kPa)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"PGAE β(P,T) Heatmap — {z_label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    safe_label = z_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("₄", "4").replace("₂", "2").replace("₁", "1").replace("₀", "0").replace("%", "pct")
    fig.savefig(output_dir / f"phase_envelope_heatmap_{safe_label}.png", dpi=220)
    plt.close(fig)


def plot_envelope_latent_trajectory(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    pgae_envelope: Dict[str, np.ndarray],
    z_label: str,
    output_dir: Path,
) -> None:
    """Plot the latent-space trajectory of the phase envelope."""
    latent_bubble = []
    latent_dew = []

    for T, P in zip(pgae_envelope["T_bubble"], pgae_envelope["P_bubble"]):
        pred = surrogate.predict_flash(float(P), float(T), z)
        latent_bubble.append(pred["latent"])

    for T, P in zip(pgae_envelope["T_dew"], pgae_envelope["P_dew"]):
        pred = surrogate.predict_flash(float(P), float(T), z)
        latent_dew.append(pred["latent"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if latent_bubble:
        lb = np.array(latent_bubble)
        ax.plot(lb[:, 0], lb[:, 1], lb[:, 2], "b-", linewidth=2, label="Bubble curve", alpha=0.9)
    if latent_dew:
        ld = np.array(latent_dew)
        ax.plot(ld[:, 0], ld[:, 1], ld[:, 2], "r--", linewidth=2, label="Dew curve", alpha=0.9)

    ax.set_xlabel("Latent-1")
    ax.set_ylabel("Latent-2")
    ax.set_zlabel("Latent-3")
    ax.set_title("Latent Manifold: Phase Envelope Trajectory")
    ax.legend(fontsize=9)
    fig.tight_layout()
    safe_label = z_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("₄", "4").replace("₂", "2").replace("₁", "1").replace("₀", "0").replace("%", "pct")
    fig.savefig(output_dir / f"phase_envelope_latent_{safe_label}.png", dpi=220)
    plt.close(fig)


# =============================================================================
# 6. Main Experiment
# =============================================================================

def run_phase_envelope_experiment(
    surrogate: PGAEFlashSurrogate,
    z: np.ndarray,
    z_label: str,
    T_min_C: float = 40.0,
    T_max_C: float = 180.0,
    n_T_points: int = 50,
    output_dir: Optional[Path] = None,
    run_winprop: bool = True,
) -> Dict:
    """Run full phase envelope comparison for one composition.

    Args:
        surrogate: loaded PGAE model
        z: feed composition (15,)
        z_label: label for plots
        T_min_C, T_max_C: temperature range (°C)
        n_T_points: number of T points for PGAE envelope
        output_dir: directory for figures/metrics
        run_winprop: whether to run WinProp (can be slow)

    Returns:
        dict with all results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Phase Envelope Reconstruction: {z_label}")
    print(f"  T range: [{T_min_C:.0f}, {T_max_C:.0f}] °C  |  n_T={n_T_points}")
    print(f"{'='*60}")

    # --- PGAE Envelope ---
    print("\n[1/3] Generating PGAE phase envelope (P-scan method)...")
    T_range = np.linspace(T_min_C, T_max_C, n_T_points)
    pgae_env = pgae_phase_envelope(surrogate, z, T_range, P_min=10.0, P_max=50000.0, n_P=80)
    n_bubble = len(pgae_env["T_bubble"])
    n_dew = len(pgae_env["T_dew"])
    print(f"  PGAE: {n_bubble} bubble points, {n_dew} dew points")
    if pgae_env["T_crit"]:
        print(f"  PGAE critical: T={pgae_env['T_crit']:.1f}°C, P={pgae_env['P_crit']:.0f} kPa")

    # --- WinProp Envelope ---
    eos_env = None
    if run_winprop:
        print("\n[2/3] Running WinProp EOS envelope...")
        P_max = 50000.0
        if n_bubble > 0:
            P_max = max(P_max, pgae_env["P_bubble"].max() * 1.3)
        eos_env = run_winprop_envelope(z, T_min_C, T_max_C, P_max)

    # --- Compare ---
    print("\n[3/3] Computing overlap metrics...")
    overlap = {}
    if eos_env is not None and len(eos_env.get("T_all", [])) > 0:
        overlap = compute_envelope_overlap(pgae_env, eos_env)
        for k, v in overlap.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4e}")
    else:
        print("  (no EOS envelope for comparison)")

    # --- Plot ---
    if output_dir is not None:
        plot_phase_envelope_comparison(pgae_env, eos_env, z, z_label, output_dir)
        plot_beta_heatmap(pgae_env, z_label, output_dir)
        plot_envelope_latent_trajectory(surrogate, z, pgae_env, z_label, output_dir)

    summary = {
        "z_label": z_label,
        "n_bubble_points": n_bubble,
        "n_dew_points": n_dew,
        "T_crit_pgae": pgae_env.get("T_crit"),
        "P_crit_pgae": pgae_env.get("P_crit"),
        "T_crit_eos": eos_env.get("T_crit") if eos_env else None,
        "P_crit_eos": eos_env.get("P_crit") if eos_env else None,
        **overlap,
    }

    return {
        "pgae_envelope": pgae_env,
        "eos_envelope": eos_env,
        "summary": summary,
    }


def main() -> None:
    config = PGAEConfig()
    if not config.best_checkpoint_path.exists():
        print(f"Checkpoint not found: {config.best_checkpoint_path}")
        print("Please run train.py first.")
        return

    print("Loading PGAE surrogate model...")
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)

    # Pick representative compositions
    df = pd.read_csv(config.dataset_path)
    z_cols = [f"z{i}" for i in range(1, NC + 1)]
    all_z = df[z_cols].to_numpy(dtype=np.float64)

    ch4_frac = all_z[:, 3]

    # 1. Typical composition (median CH4)
    med_idx = np.argmin(np.abs(ch4_frac - np.median(ch4_frac)))
    z_typical = all_z[med_idx].copy()

    # 2. Oil-rich (low CH4)
    low_ch4_idx = np.argmin(ch4_frac)
    z_oil_rich = all_z[low_ch4_idx].copy()

    # 3. Gas-rich (high CH4)
    high_ch4_idx = np.argmax(ch4_frac)
    z_gas_rich = all_z[high_ch4_idx].copy()

    experiments = [
        (z_typical, f"Typical (CH₄={z_typical[3]*100:.1f}%)"),
        (z_oil_rich, f"Oil-rich (CH₄={z_oil_rich[3]*100:.1f}%)"),
        (z_gas_rich, f"Gas-rich (CH₄={z_gas_rich[3]*100:.1f}%)"),
    ]

    all_summaries = []
    for z, label in experiments:
        result = run_phase_envelope_experiment(
            surrogate, z, label,
            T_min_C=40, T_max_C=180, n_T_points=50,
            output_dir=config.fig_phase_envelope_dir,
            run_winprop=True,
        )
        all_summaries.append(result["summary"])

    # Save summary
    summary_path = config.metric_dir / "phase_envelope_summary.json"
    # Convert to serializable format
    serializable = []
    for s in all_summaries:
        item = {}
        for k, v in s.items():
            if isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating, np.ndarray)):
                item[k] = float(v) if v is not None else None
            else:
                item[k] = v
        serializable.append(item)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPhase envelope summary saved to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Phase Envelope Summary")
    print("=" * 70)
    for s in serializable:
        print(f"\n{s['z_label']}:")
        print(f"  PGAE: {s['n_bubble_points']} bubble, {s['n_dew_points']} dew points")
        if s.get('T_crit_pgae'):
            print(f"  PGAE critical: T={s['T_crit_pgae']:.1f}°C, P={s['P_crit_pgae']:.0f} kPa")
        if s.get('T_crit_eos'):
            print(f"  EOS  critical: T={s['T_crit_eos']:.1f}°C, P={s['P_crit_eos']:.0f} kPa")
            if s.get('T_crit_deviation'):
                print(f"  T_crit deviation: {s['T_crit_deviation']:.1f}°C")
                print(f"  P_crit deviation: {s['P_crit_deviation']:.0f} kPa")
        if s.get('bubble_P_mae'):
            print(f"  Bubble P MAE: {s['bubble_P_mae']:.1f} kPa, RMSE: {s['bubble_P_rmse']:.1f} kPa")
        if s.get('dew_P_mae'):
            print(f"  Dew P MAE: {s['dew_P_mae']:.1f} kPa, RMSE: {s['dew_P_rmse']:.1f} kPa")


if __name__ == "__main__":
    main()
