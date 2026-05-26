"""Part 14: Reservoir Simulator Coupling Validation.

Simplified compositional reservoir simulator with pluggable flash:
  - PGAE flash (fast, neural surrogate)
  - WinProp EOS flash (ground truth, slow)
  - IMPEC scheme (IMplicit Pressure, Explicit Composition)
  - 4 validation scenarios: 1D injection, CVD, 2D five-spot, radial near-wellbore
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from config import PGAEConfig
from infer import PGAEFlashSurrogate
from thermo_checks import (
    ACENTRIC, BIP, MW, PC_ATM, R_GAS, R_GAS_SI, TC_K,
    _pr_ab, _solve_cubic_z,
)

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
WINPROP_EXE = r"D:\CMG\WINPROP\2022.10\Win_x64\EXE\pr202210.exe"
TEMPLATE_DAT = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\fluid_15comp.dat")
WORK_DIR = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\sim")
WORK_DIR.mkdir(parents=True, exist_ok=True)

NC = 15
COMP_NAMES = [
    "C10+", "N2", "CO2", "CH4", "C2H6", "C3H8",
    "IC4", "NC4", "IC5", "NC5", "FC6", "FC7", "FC8", "FC9", "FC10",
]

# ─────────────────────────────────────────────────────────────────────
# 1. Grid
# ─────────────────────────────────────────────────────────────────────

@dataclass
class UniformCartesianGrid:
    """1D or 2D Cartesian grid with uniform spacing."""
    nx: int
    ny: int
    dx: float               # m
    dy: float               # m
    thickness: float = 10.0  # m (pay zone)
    phi: float = 0.2
    kx: float = 100.0       # mD
    ky: float = 100.0       # mD

    def __post_init__(self):
        self.N = self.nx * self.ny
        self.cell_volume = self.dx * self.dy * self.thickness
        self.pore_volume = self.cell_volume * self.phi
        # Convert mD to m²: 1 mD = 9.869233e-16 m²
        self._k_conv = 9.869233e-16
        self._kx_si = self.kx * self._k_conv
        self._ky_si = self.ky * self._k_conv

    def cell_index(self, i: int, j: int = 0) -> int:
        return i + j * self.nx

    def neighbor_pairs(self) -> List[Tuple[int, int, float, float]]:
        """Return (cell_a, cell_b, area, dist) for all internal connections."""
        pairs = []
        # x-direction connections
        for j in range(self.ny):
            for i in range(self.nx - 1):
                a = self.cell_index(i, j)
                b = self.cell_index(i + 1, j)
                area = self.dy * self.thickness
                pairs.append((a, b, area, self.dx))
        # y-direction connections (only if ny > 1)
        for j in range(self.ny - 1):
            for i in range(self.nx):
                a = self.cell_index(i, j)
                b = self.cell_index(i, j + 1)
                area = self.dx * self.thickness
                pairs.append((a, b, area, self.dy))
        return pairs

    def geometric_transmissibility(self, cell_a: int, cell_b: int,
                                   area: float, dist: float) -> float:
        """Harmonic mean transmissibility for connection a-b (m³)."""
        T_a = self._kx_si * area / (dist / 2)
        T_b = self._kx_si * area / (dist / 2)
        return 2.0 / (1.0 / T_a + 1.0 / T_b)


@dataclass
class RadialGrid:
    """1D radial grid with logarithmic spacing."""
    nx: int
    r_w: float = 0.1       # well radius (m)
    r_max: float = 100.0   # outer radius (m)
    thickness: float = 10.0
    phi: float = 0.2
    k: float = 500.0       # mD

    def __post_init__(self):
        self.N = self.nx
        self.ny = 1
        self.dx = 1.0  # placeholder for compatibility
        self.dy = 1.0
        k_conv = 9.869233e-16
        self.kx = self.k
        self.ky = self.k
        self._k_si = self.k * k_conv
        self._kx_si = self._k_si
        self._ky_si = self._k_si

        # Log-spaced cell boundaries
        self.r_faces = np.logspace(np.log10(self.r_w), np.log10(self.r_max), self.nx + 1)
        self.r_centers = np.sqrt(self.r_faces[:-1] * self.r_faces[1:])
        self.dr = np.diff(self.r_faces)

        # Cell volumes (cylindrical shell: π(r₂² - r₁²)h)
        self.cell_volumes = np.pi * (self.r_faces[1:]**2 - self.r_faces[:-1]**2) * self.thickness
        self.pore_volumes = self.cell_volumes * self.phi
        self.cell_volume = float(np.mean(self.cell_volumes))

    def neighbor_pairs(self) -> List[Tuple[int, int, float, float]]:
        """Return (cell_a, cell_b, area, dist) for radial connections."""
        pairs = []
        for i in range(self.nx - 1):
            area = 2.0 * np.pi * self.r_faces[i + 1] * self.thickness
            dist = self.r_centers[i + 1] - self.r_centers[i]
            pairs.append((i, i + 1, area, dist))
        return pairs

    def geometric_transmissibility(self, cell_a: int, cell_b: int,
                                   area: float, dist: float) -> float:
        return self._k_si * area / dist


# ─────────────────────────────────────────────────────────────────────
# 2. Flash Interface
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FlashResult:
    beta: float
    x: np.ndarray    # (15,) liquid comp
    y: np.ndarray    # (15,) vapour comp
    converged: bool = True
    mass_residual: float = 0.0


class PGAEFlashWrapper:
    """Wrap PGAEFlashSurrogate as a standard flash function."""

    def __init__(self, surrogate: PGAEFlashSurrogate):
        self.surrogate = surrogate

    def __call__(self, P: float, T: float, z: np.ndarray) -> FlashResult:
        pred = self.surrogate.predict_flash(P, T, z)
        return FlashResult(
            beta=pred["beta"],
            x=pred["x"],
            y=pred["y"],
            mass_residual=pred.get("mass_residual", 0.0),
        )


class WinPropFlashWrapper:
    """Wrap WinProp subprocess as a standard flash function, with LRU cache."""

    def __init__(self):
        self._cache: Dict[Tuple, FlashResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        with open(TEMPLATE_DAT, "r", encoding="utf-8", errors="ignore") as f:
            self.template_lines = f.readlines()

    def _cache_key(self, P: float, T: float, z: np.ndarray) -> Tuple:
        return (round(P, 1), round(T, 1), tuple(round(v, 6) for v in z))

    def __call__(self, P: float, T: float, z: np.ndarray) -> FlashResult:
        key = self._cache_key(P, T, z)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        # Build WinProp .dat file
        dat_content = self._make_dat(P, T, z)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", dir=WORK_DIR, delete=False, encoding="utf-8"
        ) as f:
            f.write(dat_content)
            dat_path = Path(f.name)

        out_path = Path(str(dat_path).replace(".dat", ".out"))
        try:
            result = subprocess.run(
                [WINPROP_EXE], cwd=str(WORK_DIR),
                input=f"{dat_path}\n", stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, encoding="utf-8",
                errors="ignore", timeout=60.0,
            )
            if out_path.exists():
                out_text = out_path.read_text(encoding="utf-8", errors="ignore")
                parsed = _parse_ptflash_out(out_text)
                if parsed is not None:
                    fr = FlashResult(beta=parsed["beta"], x=parsed["x"], y=parsed["y"])
                    self._cache[key] = fr
                    return fr
        except (subprocess.TimeoutExpired, OSError):
            pass
        finally:
            # Cleanup temp files
            for p in [dat_path, out_path]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        return FlashResult(beta=0.0, x=z.copy(), y=z.copy(), converged=False)

    def _make_dat(self, P: float, T: float, z: np.ndarray) -> str:
        """Build WinProp .dat from template with single PT-flash."""
        lines_out = []
        in_special = False
        special_name = ""
        comp_injected = False
        i = 0
        while i < len(self.template_lines):
            line = self.template_lines[i]
            upper = line.strip().upper()

            if upper.startswith("*ENVELOPE") or upper.startswith("*FLASH"):
                in_special = True
                special_name = upper.split()[0]
                i += 1
                continue
            if upper.startswith("**=-=-="):
                in_special = False
                lines_out.append(line)
                i += 1
                continue

            if in_special:
                i += 1
                continue

            if upper.startswith("*PLOT"):
                i += 1
                continue

            if "*PRIMARY" in upper and not comp_injected:
                lines_out.append(line)
                for j in range(0, NC, 5):
                    lines_out.append("   ".join(f"{v:.6f}" for v in z[j:j + 5]))
                comp_injected = True
                i += 1
                while i < len(self.template_lines) and not self.template_lines[i].strip().startswith("*"):
                    i += 1
                continue

            lines_out.append(line)
            i += 1

        flash_section = [
            "*FLASH", '*LABEL ""', "*FEED *MIXED 1.0",
            "*KVALUE *INTERNAL", "*LEVEL 1", "*OUTPUT 1",
            "*TYPE *QNSS", f"*PRES {P:.2f}", f"*TEMP {T:.2f}",
            "*DELP 0.0", "*DELT 0.0", "*STEPP 1", "*STEPT 1", "",
        ]
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
    """Parse single PT-flash result from WinProp output (same logic as phase_boundary.py)."""
    beta = None
    x = np.zeros(NC)
    y = np.zeros(NC)

    m = re.search(r'Phase\s+Mole\s*%\s+([\d.]+)\s+([\d.]+)', out_text)
    if m:
        beta = float(m.group(2)) / 100.0
    else:
        split_match = re.search(r'Phase\s+Split:\s*(\S+)', out_text)
        if split_match:
            label = split_match.group(1).strip()
            beta = 0.0 if label == "Liquid" else 1.0
        else:
            m2 = re.search(r'Phase\s+Mole\s*%\s+([\d.]+)', out_text)
            if m2:
                beta = 0.0 if float(m2.group(1)) > 99.0 else 1.0
            else:
                m3 = re.search(r'Vap(?:our)?\s+(?:Mole\s+)?Fraction\s*[=:]\s*([\d.]+)',
                               out_text, re.IGNORECASE)
                if m3:
                    beta = float(m3.group(1))

    split_marker = out_text.find("Phase Split:")
    mole_marker = out_text.find("Phase Mole %")
    table_text = out_text[split_marker:mole_marker] if split_marker >= 0 and mole_marker > split_marker else (
        out_text[split_marker:] if split_marker >= 0 else out_text[:mole_marker] if mole_marker > 0 else out_text
    )

    is_single = "Liquid-Vapour" not in out_text[split_marker:split_marker + 100] if split_marker >= 0 else False
    liq_vals, vap_vals = [], []
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
        if "Phase" in stripped or lower.startswith("---") or "mass percent" in lower or "ln (fug" in lower:
            in_header = False
            break
        toks = stripped.split()
        nums = [float(t) for t in toks if _is_float(t)]
        if is_single and len(nums) >= 2:
            liq_vals.append(nums[1])
            vap_vals.append(nums[1])
        elif not is_single and len(nums) >= 3:
            liq_vals.append(nums[1])
            vap_vals.append(nums[2])

    if len(liq_vals) >= NC:
        x = np.array(liq_vals[:NC], dtype=np.float64) / 100.0
        y = np.array(vap_vals[:NC], dtype=np.float64) / 100.0
    else:
        return None
    if beta is None:
        return None

    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)
    x /= max(x.sum(), 1e-12)
    y /= max(y.sum(), 1e-12)
    return {"beta": beta, "x": x, "y": y}


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────
# 3. Fluid Properties
# ─────────────────────────────────────────────────────────────────────

# Component MW in kg/mol for SI calculations
MW_SI = MW / 1000.0  # g/mol → kg/mol

# Constant phase viscosities (cp → Pa·s: 1 cp = 0.001 Pa·s)
MU_OIL = 0.5 * 1e-3    # 0.5 cp → Pa·s
MU_GAS = 0.02 * 1e-3   # 0.02 cp → Pa·s


def compute_pr_densities(P_kPa: np.ndarray, T_C: float, x: np.ndarray, y: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute oil and gas phase molar volumes & densities using PR-EOS.

    Args:
        P_kPa: pressure (kPa), shape (N,)
        T_C: temperature (°C)
        x: liquid composition, shape (N, 15)
        y: vapour composition, shape (N, 15)

    Returns:
        rho_o, rho_g: phase densities (kg/m³), shape (N,)
        v_o, v_g: molar volumes (m³/mol), shape (N,)
    """
    T_K = T_C + 273.15
    P_atm = np.asarray(P_kPa) / 101.325
    N = len(P_atm)

    if N == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    a_i, b_i = _pr_ab(np.full(N, T_K))  # (N, 15), (15,)

    def _mixture_params(z_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sqrt_a = np.sqrt(a_i)
        a_mix = np.zeros(N)
        for ii in range(NC):
            for jj in range(NC):
                a_mix += z_mat[:, ii] * z_mat[:, jj] * (1.0 - BIP[ii, jj]) * sqrt_a[:, ii] * sqrt_a[:, jj]
        b_mix = z_mat @ b_i
        return a_mix, b_mix

    # Oil phase
    a_o, b_o = _mixture_params(x)
    A_o = a_o * P_atm / (R_GAS * T_K) ** 2
    B_o = b_o * P_atm / (R_GAS * T_K)
    Z_vap_o, Z_liq_o = _solve_cubic_z(A_o, B_o)
    Z_o = np.where(np.isfinite(Z_liq_o) & (Z_liq_o > B_o), Z_liq_o, Z_vap_o)

    # Gas phase
    a_g, b_g = _mixture_params(y)
    A_g = a_g * P_atm / (R_GAS * T_K) ** 2
    B_g = b_g * P_atm / (R_GAS * T_K)
    Z_vap_g, Z_liq_g = _solve_cubic_z(A_g, B_g)
    Z_g = np.where(np.isfinite(Z_vap_g) & (Z_vap_g > B_g), Z_vap_g, Z_liq_g)

    # Molar volumes: v = ZRT/P in m³/mol
    P_pa = np.asarray(P_kPa) * 1000.0
    v_o = Z_o * R_GAS_SI * T_K / P_pa
    v_g = Z_g * R_GAS_SI * T_K / P_pa

    # Molecular weights
    MW_o = x @ MW_SI
    MW_g = y @ MW_SI

    rho_o = np.where(v_o > 0, MW_o / v_o, 800.0)
    rho_g = np.where(v_g > 0, MW_g / v_g, 10.0)
    rho_o = np.clip(rho_o, 100, 2000)
    rho_g = np.clip(rho_g, 0.1, 500)

    return rho_o, rho_g, v_o, v_g


def gas_saturation(beta: np.ndarray, v_o: np.ndarray, v_g: np.ndarray) -> np.ndarray:
    """Convert molar vapour fraction to volumetric gas saturation."""
    v_total = beta * v_g + (1.0 - beta) * v_o
    v_total = np.maximum(v_total, 1e-20)
    return beta * v_g / v_total


# ─────────────────────────────────────────────────────────────────────
# 4. Relative Permeability
# ─────────────────────────────────────────────────────────────────────

def corey_rel_perm(S_g: np.ndarray, S_gc: float = 0.05, S_wc: float = 0.0,
                   n_o: float = 2.0, n_g: float = 2.0,
                   kr_o_max: float = 1.0, kr_g_max: float = 1.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Corey-type relative permeability for oil-gas system."""
    S_g_eff = np.clip((S_g - S_gc) / (1.0 - S_wc - S_gc), 0.0, 1.0)
    S_o_eff = 1.0 - S_g_eff
    kr_o = kr_o_max * S_o_eff ** n_o
    kr_g = kr_g_max * S_g_eff ** n_g
    return kr_o, kr_g


# ─────────────────────────────────────────────────────────────────────
# 5. Wells
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InjectorWell:
    cell: int
    rate_m3_day: float           # injection rate (m³/day at reservoir conditions)
    z_inj: np.ndarray             # injected composition (15,)
    r_w: float = 0.1              # well radius (m)

    def source_terms(self, P_cell: float, rho_total: float
                     ) -> Tuple[float, np.ndarray]:
        """Return (q_total [m³/s], q_component [mol/s per component])."""
        q_m3s = self.rate_m3_day / 86400.0
        # Molar injection rate per component
        if rho_total > 0:
            MW_mix = self.z_inj @ MW_SI  # kg/mol
            n_total = q_m3s * rho_total / MW_mix  # mol/s
        else:
            n_total = q_m3s / 0.001  # fallback
        return q_m3s, n_total * self.z_inj


@dataclass
class ProducerWell:
    cell: int
    bhp_kpa: float                # bottom-hole pressure (kPa)
    r_w: float = 0.1
    skin: float = 0.0

    def well_index(self, grid) -> float:
        """Peaceman well index for square grid block (m³)."""
        k_si = grid._kx_si
        h = grid.thickness
        dx, dy = grid.dx, grid.dy
        r_eq = 0.14 * np.sqrt(dx ** 2 + dy ** 2)
        if r_eq <= 0:
            r_eq = 10.0
        return 2.0 * np.pi * k_si * h / (np.log(r_eq / self.r_w) + self.skin)

    def rate(self, grid, P_cell: float, kro: float, krg: float) -> float:
        """Total volumetric production rate (m³/s)."""
        dP = P_cell - self.bhp_kpa
        if dP <= 0:
            return 0.0
        mob = kro / MU_OIL + krg / MU_GAS  # 1/(Pa·s)
        q = self.well_index(grid) * mob * dP * 1000.0  # kPa→Pa
        return q


# ─────────────────────────────────────────────────────────────────────
# 6. IMPES Solver
# ─────────────────────────────────────────────────────────────────────

class IMPESSolver:
    """Implicit Pressure, Explicit Composition solver."""

    def __init__(self, grid):
        self.grid = grid
        self._build_connection_table()

    def _build_connection_table(self):
        """Pre-compute geometric transmissibilities for all connections."""
        pairs = self.grid.neighbor_pairs()
        self._conn_a = []
        self._conn_b = []
        self._T_geo = []  # geometric transmissibility (m³)
        for a, b, area, dist in pairs:
            self._conn_a.append(a)
            self._conn_b.append(b)
            self._T_geo.append(self.grid.geometric_transmissibility(a, b, area, dist))
        self._conn_a = np.array(self._conn_a, dtype=int)
        self._conn_b = np.array(self._conn_b, dtype=int)
        self._T_geo = np.array(self._T_geo)

    def solve_pressure(
        self, P_old: np.ndarray, S_g: np.ndarray,
        rho_o: np.ndarray, rho_g: np.ndarray,
        kr_o: np.ndarray, kr_g: np.ndarray,
        pore_vol: np.ndarray, dt: float,
        well_sources: np.ndarray,  # (N,) total volumetric rate (m³/s)
    ) -> np.ndarray:
        """Solve for new pressure field P_new (N,) kPa."""
        N = self.grid.N

        # Phase mobilities (1/(Pa·s))
        lam_o = kr_o / MU_OIL
        lam_g = kr_g / MU_GAS
        lam_t = lam_o + lam_g

        # Total compressibility
        # c_oil ≈ 1e-6 kPa⁻¹, c_gas ≈ 1/P, c_rock ≈ 4.35e-7 kPa⁻¹
        c_rock = 4.35e-7
        c_oil = 1.0e-6
        c_gas = np.where(P_old > 100, 1.0 / P_old, 0.01)
        c_t = c_rock + (1.0 - S_g) * c_oil + S_g * c_gas

        # Accumulation coefficient: c_t * V_p / dt
        acc_coeff = c_t * pore_vol / dt

        # Build sparse matrix A (N×N) — symmetric
        diag = acc_coeff.copy()
        row_idx = []
        col_idx = []
        vals = []
        for k in range(len(self._conn_a)):
            a, b = self._conn_a[k], self._conn_b[k]
            T_geo = self._T_geo[k]

            # Upstream weighting for mobility at interface
            # Use arithmetic mean for simplicity
            lam_iface = 0.5 * (lam_t[a] + lam_t[b])
            T_eff = T_geo * lam_iface

            row_idx.extend([a, b])
            col_idx.extend([b, a])
            vals.extend([-T_eff, -T_eff])
            diag[a] += T_eff
            diag[b] += T_eff

        # Add diagonal
        all_row = list(range(N)) + list(row_idx)
        all_col = list(range(N)) + list(col_idx)
        all_val = list(diag) + [float(v) for v in vals]

        A = sparse.csr_matrix((all_val, (all_row, all_col)), shape=(N, N))

        # RHS
        b = acc_coeff * P_old + well_sources * pore_vol / self.grid.cell_volume * 1000.0

        P_new = spsolve(A, b)
        return np.asarray(P_new, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# 7. Compositional Simulator
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResults:
    """Structured output from a simulation run."""
    scenario: str = ""
    times: List[float] = field(default_factory=list)          # days
    pressures: List[np.ndarray] = field(default_factory=list)  # (N,) kPa at each report
    saturations: List[np.ndarray] = field(default_factory=list)  # (N,)
    compositions: List[np.ndarray] = field(default_factory=list)  # (N, 15)
    betas: List[np.ndarray] = field(default_factory=list)     # (N,)
    oil_rates: List[float] = field(default_factory=list)      # m³/day
    gas_rates: List[float] = field(default_factory=list)      # m³/day
    gor: List[float] = field(default_factory=list)            # m³/m³
    cum_oil: List[float] = field(default_factory=list)        # m³
    cum_gas: List[float] = field(default_factory=list)        # m³
    avg_pressure: List[float] = field(default_factory=list)   # kPa
    flash_count: int = 0
    wall_time: float = 0.0


class CompositionalSimulator:
    """Simplified IMPEC compositional reservoir simulator."""

    def __init__(
        self, grid, flash_fn, T_C: float,
        z_init: np.ndarray, P_init: float,
        injectors: List[InjectorWell] = None,
        producers: List[ProducerWell] = None,
    ):
        self.grid = grid
        self.flash_fn = flash_fn
        self.T_C = T_C
        self.N = grid.N
        self.injectors = injectors or []
        self.producers = producers or []
        self.solver = IMPESSolver(grid)

        # State arrays
        self.P = np.full(self.N, P_init, dtype=np.float64)
        self.z = np.tile(z_init.astype(np.float64), (self.N, 1))
        self.beta = np.zeros(self.N)
        self.x = np.zeros((self.N, NC))
        self.y = np.zeros((self.N, NC))
        self.S_g = np.zeros(self.N)
        self.rho_o = np.full(self.N, 800.0)
        self.rho_g = np.full(self.N, 10.0)
        self.v_o = np.zeros(self.N)
        self.v_g = np.zeros(self.N)

        # Pore volumes per cell
        if isinstance(grid, RadialGrid):
            self.pore_vol = grid.pore_volumes
        else:
            self.pore_vol = np.full(self.N, grid.pore_volume)

        # Initial flash
        self._flash_all()
        self.flash_count = self.N

    def _flash_all(self):
        """Flash every cell to update beta, x, y."""
        for i in range(self.N):
            fr = self.flash_fn(self.P[i], self.T_C, self.z[i])
            self.beta[i] = fr.beta
            self.x[i] = fr.x
            self.y[i] = fr.y

    def _update_fluid_props(self):
        """Compute densities, saturation, rel-perm from current state."""
        self.rho_o, self.rho_g, self.v_o, self.v_g = compute_pr_densities(
            self.P, self.T_C, self.x, self.y
        )
        self.S_g = gas_saturation(self.beta, self.v_o, self.v_g)
        self.S_g = np.clip(self.S_g, 0.0, 1.0)

    def _well_source_terms(self) -> np.ndarray:
        """Compute total volumetric source for each cell (m³/s)."""
        q_total = np.zeros(self.N)
        for inj in self.injectors:
            rho_t = self.rho_o[inj.cell] * (1 - self.S_g[inj.cell]) + \
                    self.rho_g[inj.cell] * self.S_g[inj.cell]
            q, _ = inj.source_terms(self.P[inj.cell], rho_t)
            q_total[inj.cell] += q

        kr_o, kr_g = corey_rel_perm(self.S_g)
        for prod in self.producers:
            q = prod.rate(self.grid, self.P[prod.cell],
                          kr_o[prod.cell], kr_g[prod.cell])
            q_total[prod.cell] -= q
        return q_total

    def _update_compositions(self, P_new: np.ndarray, dt: float):
        """Explicit composition update using inter-cell fluxes."""
        N = self.N
        z_new = self.z.copy()

        kr_o, kr_g = corey_rel_perm(self.S_g)
        lam_o = kr_o / MU_OIL
        lam_g = kr_g / MU_GAS
        k_si = self.grid._kx_si

        # Compute phase fluxes at each connection
        for k in range(len(self.solver._conn_a)):
            a, b = self.solver._conn_a[k], self.solver._conn_b[k]
            T_geo = self.solver._T_geo[k]
            dP = P_new[b] - P_new[a]  # positive = flow from a to b

            if abs(dP) < 1e-10:
                continue

            # Upstream weighting
            if dP > 0:  # flow a→b
                up, dn = a, b
                sign = 1.0
            else:
                up, dn = b, a

            # Oil flux (molar): F_o = T_geo * lam_o * dP / v_o_up
            lam_o_iface = lam_o[up]  # single-point upstream
            F_o = T_geo * lam_o_iface * abs(dP) * 1000.0  # kPa→Pa, m³/s

            # Gas flux
            lam_g_iface = lam_g[up]
            F_g = T_geo * lam_g_iface * abs(dP) * 1000.0

            # Molar flux of each component
            for c in range(NC):
                comp_flux = (F_o * self.x[up, c] / max(self.v_o[up], 1e-20) +
                             F_g * self.y[up, c] / max(self.v_g[up], 1e-20))
                # Update: remove from upstream, add to downstream
                if self.v_o[up] > 0 and self.v_g[up] > 0:
                    molar_change = comp_flux * dt
                    z_new[a, c] -= molar_change / self.pore_vol[a]
                    z_new[b, c] += molar_change / self.pore_vol[b]

        # Well source terms for components
        for inj in self.injectors:
            _, q_comp = inj.source_terms(self.P[inj.cell], 800.0)
            z_new[inj.cell] += q_comp * dt / self.pore_vol[inj.cell]

        kr_o, kr_g = corey_rel_perm(self.S_g)
        for prod in self.producers:
            q_total = prod.rate(self.grid, self.P[prod.cell],
                                kr_o[prod.cell], kr_g[prod.cell])
            # Produce at overall cell composition
            if q_total > 0:
                # Molar production rate per component
                rho_t = (self.rho_o[prod.cell] * (1.0 - self.S_g[prod.cell]) +
                         self.rho_g[prod.cell] * self.S_g[prod.cell])
                MW_mix = self.z[prod.cell] @ MW_SI
                if MW_mix > 0 and rho_t > 0:
                    n_total = q_total * rho_t / MW_mix
                    z_new[prod.cell] -= self.z[prod.cell] * n_total * dt / self.pore_vol[prod.cell]

        # Clamp and normalize
        z_new = np.clip(z_new, 0.0, None)
        for i in range(N):
            s = z_new[i].sum()
            if s > 0:
                z_new[i] /= s
            else:
                z_new[i] = self.z[i]
        self.z = z_new

    def step(self, dt: float) -> bool:
        """Advance one timestep. Returns True if successful."""
        # 1. Compute fluid properties and well sources
        self._update_fluid_props()
        q_total = self._well_source_terms()

        # 2. Solve pressure
        kr_o, kr_g = corey_rel_perm(self.S_g)
        self.P = self.solver.solve_pressure(
            self.P, self.S_g, self.rho_o, self.rho_g,
            kr_o, kr_g, self.pore_vol, dt, q_total,
        )

        # 3. Update compositions
        self._update_compositions(self.P, dt)

        # 4. Flash all cells
        self._flash_all()
        self.flash_count += self.N

        # 5. Update properties with new flash results
        self._update_fluid_props()

        # Check for NaN
        if np.any(np.isnan(self.P)) or np.any(np.isnan(self.z)):
            return False
        return True

    def run(self, t_max: float, dt_init: float = 1.0,
            dt_min: float = 0.01, dt_max: float = 100.0,
            n_report: int = 50) -> SimulationResults:
        """Run simulation from t=0 to t_max (days)."""
        t_start = time.time()
        results = SimulationResults(scenario="")
        results.flash_count = self.flash_count

        dt = dt_init
        t = 0.0
        step_count = 0
        report_interval = max(1, int(t_max / dt_init / n_report))
        next_report = report_interval

        pbar = tqdm(total=float(t_max), desc="Simulating", unit="day")
        last_t = 0.0

        while t < t_max - 1e-8:
            dt = min(dt, t_max - t)
            dt = max(dt, dt_min)
            dt = min(dt, dt_max)

            success = self.step(dt)
            if not success:
                dt *= 0.5
                if dt < dt_min:
                    print(f"  WARNING: Timestep below minimum at t={t:.1f} days")
                    break
                continue

            t += dt
            step_count += 1
            pbar.update(t - last_t)
            last_t = t

            # Report
            if step_count >= next_report or t >= t_max - 1e-8:
                next_report = step_count + report_interval
                results.times.append(t)
                results.pressures.append(self.P.copy())
                results.saturations.append(self.S_g.copy())
                results.compositions.append(self.z.copy())
                results.betas.append(self.beta.copy())
                results.avg_pressure.append(float(np.mean(self.P)))

                # Production rates
                q_oil_tot = 0.0
                q_gas_tot = 0.0
                kr_o, kr_g = corey_rel_perm(self.S_g)
                for prod in self.producers:
                    q = prod.rate(self.grid, self.P[prod.cell],
                                  kr_o[prod.cell], kr_g[prod.cell])
                    q_oil_day = q * (1.0 - self.S_g[prod.cell]) * 86400.0
                    q_gas_day = q * self.S_g[prod.cell] * 86400.0
                    q_oil_tot += q_oil_day
                    q_gas_tot += q_gas_day
                results.oil_rates.append(q_oil_tot)
                results.gas_rates.append(q_gas_tot)
                results.gor.append(q_gas_tot / max(q_oil_tot, 1e-8))

                # Cumulative production
                if len(results.oil_rates) > 1:
                    dt_avg = 0.5 * (results.times[-1] - results.times[-2])
                    results.cum_oil.append(results.cum_oil[-1] +
                                           q_oil_tot * dt_avg)
                    results.cum_gas.append(results.cum_gas[-1] +
                                           q_gas_tot * dt_avg)
                else:
                    results.cum_oil.append(0.0)
                    results.cum_gas.append(0.0)

            # Adaptive timestep (CFL-based)
            # Increase dt if everything is stable
            if success and dt < dt_max:
                dt = min(dt * 1.2, dt_max)

        pbar.close()
        results.wall_time = time.time() - t_start
        results.flash_count = self.flash_count
        return results


# ─────────────────────────────────────────────────────────────────────
# 8. Validation Scenarios
# ─────────────────────────────────────────────────────────────────────

def _get_default_compositions():
    """Return the 3 representative compositions used in Parts 11-13."""
    z_typical = np.array([
        0.076, 0.012, 0.021, 0.368, 0.089, 0.076, 0.029, 0.042,
        0.031, 0.048, 0.045, 0.048, 0.047, 0.038, 0.030,
    ])  # CH4=36.8%
    z_oil = np.array([
        0.150, 0.008, 0.015, 0.208, 0.072, 0.065, 0.035, 0.052,
        0.042, 0.058, 0.055, 0.058, 0.057, 0.067, 0.058,
    ])  # CH4=20.8%
    z_gas = np.array([
        0.010, 0.018, 0.030, 0.628, 0.069, 0.048, 0.015, 0.024,
        0.016, 0.022, 0.021, 0.022, 0.021, 0.031, 0.025,
    ])  # CH4=62.8%
    return z_typical, z_oil, z_gas


def run_scenario_1_co2_injection(flash_fn) -> SimulationResults:
    """1D CO2 gas injection — CO2 displaces oil-rich fluid."""
    print("\n" + "=" * 60)
    print("Scenario 1: 1D CO2 Gas Injection")
    print("=" * 60)

    _, z_oil, _ = _get_default_compositions()

    grid = UniformCartesianGrid(nx=80, ny=1, dx=10.0, dy=10.0,
                                 thickness=10.0, phi=0.2, kx=100.0)

    # CO2 injection composition (pure CO2)
    z_co2 = np.zeros(NC)
    z_co2[2] = 1.0  # CO2 is component index 2

    injector = InjectorWell(cell=grid.cell_index(grid.nx - 1), rate_m3_day=500.0, z_inj=z_co2)
    producer = ProducerWell(cell=grid.cell_index(0), bhp_kpa=8000.0)

    sim = CompositionalSimulator(
        grid=grid, flash_fn=flash_fn, T_C=90.0,
        z_init=z_oil, P_init=15000.0,
        injectors=[injector], producers=[producer],
    )

    results = sim.run(t_max=500.0, dt_init=1.0, dt_min=0.1, dt_max=20.0, n_report=100)
    results.scenario = "co2_injection_1d"
    return results


def run_scenario_2_depletion(flash_fn) -> SimulationResults:
    """1D (single-cell) constant volume depletion."""
    print("\n" + "=" * 60)
    print("Scenario 2: Constant Volume Depletion (CVD)")
    print("=" * 60)

    _, _, z_gas = _get_default_compositions()

    grid = UniformCartesianGrid(nx=1, ny=1, dx=10.0, dy=10.0,
                                 thickness=10.0, phi=0.2, kx=100.0)

    # Producer with declining BHP to simulate depletion
    # Instead, use a simple tank model: directly reduce pressure each step
    # and flash to track phase behavior

    # We'll use the simulator but with a producer that gradually reduces BHP
    sim = CompositionalSimulator(
        grid=grid, flash_fn=flash_fn, T_C=90.0,
        z_init=z_gas, P_init=25000.0, producers=[ProducerWell(cell=0, bhp_kpa=20000.0)],
    )

    # For CVD, we simulate a slow pressure decline
    results = SimulationResults(scenario="depletion_1d")
    t_start = time.time()

    P_current = 25000.0
    T_C = 90.0
    z_current = z_gas.copy()
    flash_count = 0

    P_points = np.linspace(25000, 1000, 100)
    for P_target in tqdm(P_points, desc="CVD simulation"):
        P_current = P_target * 0.9 + P_current * 0.1  # smooth transition
        fr = flash_fn(P_current, T_C, z_current)
        flash_count += 1

        # Produce at current composition
        # Remove some moles to achieve pressure decline
        # For tank model: just track state
        results.times.append(P_current)  # Using P as "time" axis
        results.pressures.append(np.array([P_current]))
        results.betas.append(np.array([fr.beta]))
        results.saturations.append(np.array([0.0]))  # computed below
        results.compositions.append(z_current.copy().reshape(1, -1))

    # Compute gas saturation from beta
    all_P = np.array([r[0] for r in results.pressures])
    all_beta = np.array([r[0] for r in results.betas])

    rho_o, rho_g, v_o, v_g = compute_pr_densities(
        all_P, T_C,
        np.tile(z_gas, (len(all_P), 1)),
        np.tile(z_gas, (len(all_P), 1)),
    )
    # For CVD, x≈y≈z at dew point, so use the flash outputs
    # Better: redo with actual x,y
    x_vals = np.tile(z_gas, (len(all_P), 1))
    y_vals = np.tile(z_gas, (len(all_P), 1))
    for i in range(len(all_P)):
        fr2 = flash_fn(all_P[i], T_C, z_gas)
        x_vals[i] = fr2.x
        y_vals[i] = fr2.y
    rho_o2, rho_g2, v_o2, v_g2 = compute_pr_densities(all_P, T_C, x_vals, y_vals)
    S_g_vals = gas_saturation(all_beta, v_o2, v_g2)

    for i in range(len(all_P)):
        results.saturations[i] = np.array([S_g_vals[i]])
        results.avg_pressure.append(all_P[i])

    # Liquid dropout = 1 - S_g
    results.oil_rates = list(S_g_vals)  # repurpose: gas saturation vs P
    results.gas_rates = list(all_beta)  # repurpose: beta vs P
    results.cum_oil = list(1.0 - S_g_vals)  # liquid dropout

    results.wall_time = time.time() - t_start
    results.flash_count = flash_count
    return results


def run_scenario_3_fivespot(flash_fn) -> SimulationResults:
    """2D quarter five-spot CO2 injection."""
    print("\n" + "=" * 60)
    print("Scenario 3: 2D Five-Spot Pattern")
    print("=" * 60)

    _, z_oil, _ = _get_default_compositions()

    grid = UniformCartesianGrid(nx=21, ny=21, dx=20.0, dy=20.0,
                                 thickness=10.0, phi=0.2, kx=100.0, ky=100.0)

    z_co2 = np.zeros(NC)
    z_co2[2] = 1.0

    injector = InjectorWell(cell=grid.cell_index(0, 0), rate_m3_day=2000.0, z_inj=z_co2)
    producer = ProducerWell(cell=grid.cell_index(grid.nx - 1, grid.ny - 1), bhp_kpa=8000.0)

    sim = CompositionalSimulator(
        grid=grid, flash_fn=flash_fn, T_C=90.0,
        z_init=z_oil, P_init=15000.0,
        injectors=[injector], producers=[producer],
    )

    results = sim.run(t_max=300.0, dt_init=0.5, dt_min=0.05, dt_max=10.0, n_report=60)
    results.scenario = "fivespot_2d"
    return results


def run_scenario_4_radial(flash_fn) -> SimulationResults:
    """Near-wellbore radial CO2 injection."""
    print("\n" + "=" * 60)
    print("Scenario 4: Near-Wellbore Radial")
    print("=" * 60)

    _, z_oil, _ = _get_default_compositions()

    grid = RadialGrid(nx=40, r_w=0.1, r_max=100.0, thickness=10.0,
                       phi=0.2, k=500.0)

    z_co2 = np.zeros(NC)
    z_co2[2] = 1.0

    # Inject at innermost cell, produce from outermost cell
    injector = InjectorWell(cell=0, rate_m3_day=200.0, z_inj=z_co2)
    producer = ProducerWell(cell=grid.N - 1, bhp_kpa=5000.0)

    sim = CompositionalSimulator(
        grid=grid, flash_fn=flash_fn, T_C=90.0,
        z_init=z_oil, P_init=25000.0,
        injectors=[injector], producers=[producer],
    )

    results = sim.run(t_max=100.0, dt_init=0.01, dt_min=0.001, dt_max=1.0, n_report=100)
    results.scenario = "radial_near_wellbore"
    return results


# ─────────────────────────────────────────────────────────────────────
# 9. EOS Comparison
# ─────────────────────────────────────────────────────────────────────

def compare_at_checkpoints(
    scenario_name: str,
    pgae_results: SimulationResults,
    setup_fn: Callable,
    winprop_flash: WinPropFlashWrapper,
    n_checkpoints: int = 5,
) -> Dict:
    """Compare PGAE vs WinProp at selected timesteps."""
    print(f"\n{'=' * 60}")
    print(f"WinProp Comparison: {scenario_name}")
    print(f"{'=' * 60}")

    n_total = len(pgae_results.times)
    if n_total <= n_checkpoints * 2:
        checkpoints = list(range(n_total))
    else:
        step = max(1, n_total // n_checkpoints)
        checkpoints = list(range(0, n_total, step))

    if checkpoints[-1] != n_total - 1:
        checkpoints.append(n_total - 1)

    print(f"  Comparing at {len(checkpoints)} checkpoints...")

    all_beta_pgae = []
    all_beta_eos = []
    all_x_pgae = []
    all_x_eos = []
    all_y_pgae = []
    all_y_eos = []
    all_P = []

    for idx in tqdm(checkpoints, desc="WinProp checkpoints"):
        P_cells = pgae_results.pressures[idx]
        z_cells = pgae_results.compositions[idx]
        T_C = 90.0  # same for all scenarios

        for i in range(len(P_cells)):
            # PGAE result
            beta_p = pgae_results.betas[idx][i]
            all_beta_pgae.append(beta_p)
            all_P.append(P_cells[i])

            # WinProp result
            fr = winprop_flash(P_cells[i], T_C, z_cells[i])
            all_beta_eos.append(fr.beta)

    all_beta_pgae = np.array(all_beta_pgae)
    all_beta_eos = np.array(all_beta_eos)
    all_P = np.array(all_P)

    valid = np.isfinite(all_beta_eos) & np.isfinite(all_beta_pgae)
    beta_mae = float(np.mean(np.abs(all_beta_pgae[valid] - all_beta_eos[valid])))
    beta_rmse = float(np.sqrt(np.mean((all_beta_pgae[valid] - all_beta_eos[valid]) ** 2)))

    return {
        "scenario": scenario_name,
        "n_points": int(valid.sum()),
        "beta_mae": beta_mae,
        "beta_rmse": beta_rmse,
        "beta_pgae": all_beta_pgae.tolist(),
        "beta_eos": all_beta_eos.tolist(),
        "winprop_cache_hits": winprop_flash._cache_hits,
        "winprop_cache_misses": winprop_flash._cache_misses,
    }


# ─────────────────────────────────────────────────────────────────────
# 10. Visualization
# ─────────────────────────────────────────────────────────────────────

_CFG = PGAEConfig()
_FIG_DIR = _CFG.fig_reservoir_dir
_METRIC_DIR = _CFG.metric_dir / "simulator"
_METRIC_DIR.mkdir(parents=True, exist_ok=True)


def plot_production_history(results: SimulationResults, save_path: Path):
    """Production history: Np, Gp, GOR, Pavg vs time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.array(results.times)

    # Cumulative oil
    ax = axes[0, 0]
    ax.plot(t, results.cum_oil, 'b-', lw=2)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Cumulative Oil (m³)")
    ax.set_title("Cumulative Oil Production"); ax.grid(alpha=0.3)

    # Cumulative gas
    ax = axes[0, 1]
    ax.plot(t, results.cum_gas, 'r-', lw=2)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Cumulative Gas (m³)")
    ax.set_title("Cumulative Gas Production"); ax.grid(alpha=0.3)

    # GOR
    ax = axes[1, 0]
    ax.plot(t, results.gor, 'm-', lw=1.5)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("GOR (m³/m³)")
    ax.set_title("Gas-Oil Ratio"); ax.grid(alpha=0.3)

    # Avg pressure
    ax = axes[1, 1]
    ax.plot(t, results.avg_pressure, 'k-', lw=2)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Average Pressure (kPa)")
    ax.set_title("Average Reservoir Pressure"); ax.grid(alpha=0.3)

    fig.suptitle(f"{results.scenario}: Production History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_composition_profiles(results: SimulationResults, save_path: Path):
    """Composition profiles at 3 selected times."""
    n_times = len(results.times)
    if n_times < 3:
        return
    idxs = [0, n_times // 2, n_times - 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    key_comps = [1, 2, 3, 4, 0]  # N2, CO2, CH4, C2H6, C10+
    key_names = ["N2", "CO2", "CH4", "C2H6", "C10+"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_comps)))

    for ax_idx, time_idx in enumerate(idxs):
        ax = axes[ax_idx]
        z = results.compositions[time_idx]
        t = results.times[time_idx]
        x_pos = np.arange(z.shape[0])
        for k, (ci, cname) in enumerate(zip(key_comps, key_names)):
            ax.plot(x_pos, z[:, ci], color=colors[k], lw=1.5, label=cname)
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"t = {t:.0f} days")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{results.scenario}: Composition Profiles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_saturation_profiles(results: SimulationResults, save_path: Path):
    """Gas saturation profiles at 3 selected times."""
    n_times = len(results.times)
    if n_times < 3:
        return
    idxs = [0, n_times // 2, n_times - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idxs)))
    for k, time_idx in enumerate(idxs):
        Sg = results.saturations[time_idx]
        x_pos = np.arange(len(Sg))
        ax.plot(x_pos, Sg, color=colors[k], lw=2,
                label=f"t = {results.times[time_idx]:.0f} days")
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Gas Saturation")
    ax.set_title(f"{results.scenario}: Gas Saturation Profiles")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_2d_maps(results: SimulationResults, grid: UniformCartesianGrid, save_path: Path):
    """2D pressure and gas saturation maps for five-spot scenario."""
    n_times = len(results.times)
    if n_times < 4:
        return
    idxs = [0, n_times // 3, 2 * n_times // 3, n_times - 1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for k, time_idx in enumerate(idxs):
        P_2d = results.pressures[time_idx].reshape(grid.ny, grid.nx)
        Sg_2d = results.saturations[time_idx].reshape(grid.ny, grid.nx)
        t_label = results.times[time_idx]

        # Pressure map
        ax = axes[0, k]
        im = ax.imshow(P_2d, origin="lower", cmap="RdBu_r", aspect="equal")
        ax.set_title(f"P at t={t_label:.0f}d")
        ax.set_xlabel("i"); ax.set_ylabel("j")
        plt.colorbar(im, ax=ax, label="kPa", shrink=0.8)

        # Gas saturation map
        ax = axes[1, k]
        im = ax.imshow(Sg_2d, origin="lower", cmap="YlOrRd",
                       vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Sg at t={t_label:.0f}d")
        ax.set_xlabel("i"); ax.set_ylabel("j")
        plt.colorbar(im, ax=ax, label="S_g", shrink=0.8)

        # Mark wells
        for ax_row in [axes[0, k], axes[1, k]]:
            ax_row.plot(0, 0, 'g^', markersize=10, label="Injector")
            ax_row.plot(grid.nx - 1, grid.ny - 1, 'ko', markersize=10, label="Producer")

    fig.suptitle(f"{results.scenario}: 2D Pressure & Saturation Maps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_cvd_curves(results: SimulationResults, save_path: Path):
    """CVD-specific: liquid dropout and beta vs pressure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    P = np.array(results.avg_pressure)
    liq_dropout = np.array(results.cum_oil)
    beta = np.array(results.gas_rates)

    # Liquid dropout curve
    ax1.plot(P, liq_dropout, 'b-', lw=2)
    ax1.set_xlabel("Pressure (kPa)"); ax1.set_ylabel("Liquid Saturation")
    ax1.set_title("Liquid Dropout Curve (CVD)")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)

    # Beta vs pressure
    ax2.plot(P, beta, 'r-', lw=2)
    ax2.set_xlabel("Pressure (kPa)"); ax2.set_ylabel("Vapour Mole Fraction β")
    ax2.set_title("β vs Pressure (CVD)")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3)

    fig.suptitle("Constant Volume Depletion Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_comparison_beta_parity(comparison: Dict, save_path: Path):
    """Beta parity: PGAE vs WinProp scatter."""
    fig, ax = plt.subplots(figsize=(8, 7))
    beta_p = np.array(comparison["beta_pgae"])
    beta_e = np.array(comparison["beta_eos"])

    valid = np.isfinite(beta_p) & np.isfinite(beta_e)
    ax.scatter(beta_e[valid], beta_p[valid], c='steelblue', alpha=0.5, s=20, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5, label="y=x")
    ax.set_xlabel("WinProp EOS β")
    ax.set_ylabel("PGAE β")
    ax.set_title(f"{comparison['scenario']}: β Parity (MAE={comparison['beta_mae']:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def plot_summary_dashboard(metrics: List[Dict], save_path: Path):
    """Summary bar chart of all scenario metrics."""
    if not metrics:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    scenario_names = [m["scenario"] for m in metrics]
    beta_maes = [m.get("beta_mae", 0) for m in metrics]

    # Beta MAE
    ax = axes[0, 0]
    bars = ax.bar(scenario_names, beta_maes, color='steelblue', edgecolor='navy')
    ax.set_ylabel("β MAE"); ax.set_title("β MAE vs WinProp")
    ax.tick_params(axis='x', rotation=45)

    # Flash count
    n_flash = [m.get("n_flash", 0) for m in metrics]
    ax = axes[0, 1]
    ax.bar(scenario_names, n_flash, color='coral', edgecolor='darkred')
    ax.set_ylabel("Flash Count"); ax.set_title("Total Flash Evaluations")
    ax.tick_params(axis='x', rotation=45)

    # Wall time
    t_wall = [m.get("wall_time", 0) for m in metrics]
    ax = axes[0, 2]
    ax.bar(scenario_names, t_wall, color='seagreen', edgecolor='darkgreen')
    ax.set_ylabel("Wall Time (s)"); ax.set_title("Simulation Time")
    ax.tick_params(axis='x', rotation=45)

    # N points compared
    n_pts = [m.get("n_points", 0) for m in metrics]
    ax = axes[1, 0]
    ax.bar(scenario_names, n_pts, color='orchid', edgecolor='purple')
    ax.set_ylabel("N Points"); ax.set_title("Comparison Points")
    ax.tick_params(axis='x', rotation=45)

    # Speed
    speed_flash = [m.get("flash_per_sec", 0) for m in metrics]
    ax = axes[1, 1]
    ax.bar(scenario_names, speed_flash, color='goldenrod', edgecolor='brown')
    ax.set_ylabel("Flashes/s"); ax.set_title("Flash Throughput")
    ax.tick_params(axis='x', rotation=45)

    # Empty subplot with text summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_lines = ["Summary:", ""]
    for m in metrics:
        summary_lines.append(f"{m['scenario']}:")
        summary_lines.append(f"  β MAE = {m.get('beta_mae', 'N/A'):.4f}")
        summary_lines.append(f"  Wall time = {m.get('wall_time', 'N/A'):.1f}s")
        summary_lines.append("")
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    fig.suptitle("Part 14: Reservoir Simulator Coupling — Summary Dashboard",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Part 14: Reservoir Simulator Coupling Validation")
    print("=" * 60)

    # Load PGAE model
    print("\nLoading PGAE model...")
    config = PGAEConfig()
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)
    pgae_flash = PGAEFlashWrapper(surrogate)
    print(f"  Model loaded ({config.best_checkpoint_path})")

    # Check WinProp availability
    winprop_available = os.path.exists(WINPROP_EXE)
    if winprop_available:
        print(f"  WinProp found at {WINPROP_EXE}")
        winprop_flash = WinPropFlashWrapper()
    else:
        print("  WinProp not found — skipping EOS comparison")
        winprop_flash = None

    all_metrics = []

    # ── Scenario 2: CVD (simplest, run first to verify) ──
    print("\n" + "─" * 60)
    results_cvd = run_scenario_2_depletion(pgae_flash)
    plot_cvd_curves(results_cvd, _FIG_DIR / "cvd_depletion.png")
    print(f"  Wall time: {results_cvd.wall_time:.1f}s")
    print(f"  Flash count: {results_cvd.flash_count}")
    metric_cvd = {
        "scenario": "depletion_1d",
        "wall_time": results_cvd.wall_time,
        "n_flash": results_cvd.flash_count,
        "flash_per_sec": results_cvd.flash_count / max(results_cvd.wall_time, 1e-3),
    }
    if winprop_flash:
        comp_cvd = compare_at_checkpoints(
            "depletion_1d", results_cvd, run_scenario_2_depletion, winprop_flash, n_checkpoints=5,
        )
        metric_cvd.update(comp_cvd)
        plot_comparison_beta_parity(comp_cvd, _FIG_DIR / "cvd_beta_parity.png")
    all_metrics.append(metric_cvd)

    # ── Scenario 1: 1D CO2 Injection ──
    results_inj = run_scenario_1_co2_injection(pgae_flash)
    plot_production_history(results_inj, _FIG_DIR / "co2_injection_production.png")
    plot_composition_profiles(results_inj, _FIG_DIR / "co2_injection_composition.png")
    plot_saturation_profiles(results_inj, _FIG_DIR / "co2_injection_saturation.png")
    print(f"  Wall time: {results_inj.wall_time:.1f}s")
    print(f"  Flash count: {results_inj.flash_count}")
    metric_inj = {
        "scenario": "co2_injection_1d",
        "wall_time": results_inj.wall_time,
        "n_flash": results_inj.flash_count,
        "flash_per_sec": results_inj.flash_count / max(results_inj.wall_time, 1e-3),
    }
    if winprop_flash:
        comp_inj = compare_at_checkpoints(
            "co2_injection_1d", results_inj, run_scenario_1_co2_injection, winprop_flash, n_checkpoints=5,
        )
        metric_inj.update(comp_inj)
        plot_comparison_beta_parity(comp_inj, _FIG_DIR / "co2_injection_beta_parity.png")
    all_metrics.append(metric_inj)

    # ── Scenario 3: 2D Five-Spot ──
    results_5spot = run_scenario_3_fivespot(pgae_flash)
    plot_production_history(results_5spot, _FIG_DIR / "fivespot_production.png")
    # 2D maps
    grid_2d = UniformCartesianGrid(nx=21, ny=21, dx=20.0, dy=20.0, thickness=10.0, phi=0.2, kx=100.0)
    plot_2d_maps(results_5spot, grid_2d, _FIG_DIR / "fivespot_2d_maps.png")
    print(f"  Wall time: {results_5spot.wall_time:.1f}s")
    print(f"  Flash count: {results_5spot.flash_count}")
    metric_5spot = {
        "scenario": "fivespot_2d",
        "wall_time": results_5spot.wall_time,
        "n_flash": results_5spot.flash_count,
        "flash_per_sec": results_5spot.flash_count / max(results_5spot.wall_time, 1e-3),
    }
    if winprop_flash:
        comp_5spot = compare_at_checkpoints(
            "fivespot_2d", results_5spot, run_scenario_3_fivespot, winprop_flash, n_checkpoints=3,
        )
        metric_5spot.update(comp_5spot)
        plot_comparison_beta_parity(comp_5spot, _FIG_DIR / "fivespot_beta_parity.png")
    all_metrics.append(metric_5spot)

    # ── Scenario 4: Radial Near-Wellbore ──
    results_radial = run_scenario_4_radial(pgae_flash)
    plot_production_history(results_radial, _FIG_DIR / "radial_production.png")
    plot_saturation_profiles(results_radial, _FIG_DIR / "radial_saturation.png")
    print(f"  Wall time: {results_radial.wall_time:.1f}s")
    print(f"  Flash count: {results_radial.flash_count}")
    metric_radial = {
        "scenario": "radial_near_wellbore",
        "wall_time": results_radial.wall_time,
        "n_flash": results_radial.flash_count,
        "flash_per_sec": results_radial.flash_count / max(results_radial.wall_time, 1e-3),
    }
    if winprop_flash:
        comp_radial = compare_at_checkpoints(
            "radial_near_wellbore", results_radial, run_scenario_4_radial, winprop_flash, n_checkpoints=3,
        )
        metric_radial.update(comp_radial)
        plot_comparison_beta_parity(comp_radial, _FIG_DIR / "radial_beta_parity.png")
    all_metrics.append(metric_radial)

    # ── Summary ──
    plot_summary_dashboard(all_metrics, _FIG_DIR / "simulator_summary_dashboard.png")

    # Save metrics
    metrics_path = _METRIC_DIR / "simulator_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved to: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PART 14 SUMMARY")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Wall(s)':<10} {'Flashes':<12} {'Flash/s':<12} {'β MAE':<10}")
    print("-" * 80)
    for m in all_metrics:
        bmae = m.get('beta_mae', float('nan'))
        print(f"{m['scenario']:<25} {m['wall_time']:<10.1f} {m['n_flash']:<12} "
              f"{m['flash_per_sec']:<12.0f} {bmae:<10.4f}")
    print("-" * 80)

    return all_metrics


if __name__ == "__main__":
    main()
