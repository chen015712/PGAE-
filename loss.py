from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import PGAEConfig

# ===========================================================================
# PR-EOS constants (hard-coded from fluid_15comp.dat for reproducibility)
# ===========================================================================

_PC_ATM_T = torch.tensor([
    15.0, 33.5, 72.8, 45.4, 48.2,
    41.9, 36.0, 37.5, 33.4, 33.3,
    32.46, 30.97, 29.12, 26.94, 25.01,
], dtype=torch.float64)

_TC_K_T = torch.tensor([
    700.0, 126.2, 304.2, 190.6, 305.4,
    369.8, 408.1, 425.2, 460.4, 469.6,
    507.5, 543.2, 570.5, 598.5, 622.1,
], dtype=torch.float64)

_ACENTRIC_T = torch.tensor([
    0.6, 0.04, 0.225, 0.008, 0.098,
    0.152, 0.176, 0.193, 0.227, 0.251,
    0.27504, 0.308301, 0.351327, 0.390781, 0.443774,
], dtype=torch.float64)

_R_GAS = 82.05746  # cm³·atm/(mol·K)
_SQRT2 = math.sqrt(2.0)

# BIP matrix (15×15, symmetric, diagonal=0).
# Only C10+ (col 0) and N2 (col 1) have non-zero BIP with other components.
_bip = torch.zeros(15, 15, dtype=torch.float64)
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
        _bip[i + 1, j] = val
        _bip[j, i + 1] = val
_ONE_MINUS_K = 1.0 - _bip  # (15, 15)


def _pr_alpha_torch(Tr: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Peng-Robinson alpha(T), vectorized over components.

    Args:
        Tr: reduced temperature, shape (N, 15)
        omega: acentric factor, shape (15,)

    Returns:
        alpha, shape (N, 15)
    """
    m = torch.where(
        omega <= 0.49,
        0.37464 + 1.54226 * omega - 0.26992 * omega ** 2,
        0.3796 + 1.485 * omega - 0.1644 * omega ** 2 + 0.01667 * omega ** 3,
    )  # (15,)
    return (1.0 + m[None, :] * (1.0 - torch.sqrt(Tr))) ** 2


def _pr_ab_torch(T_K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-component PR a and b parameters.

    Args:
        T_K: temperature (K), shape (N,)

    Returns:
        a_i: (N, 15)  in cm⁶·atm/mol²
        b_i: (15,)    in cm³/mol
    """
    device = T_K.device
    Tc = _TC_K_T.to(device)
    Pc = _PC_ATM_T.to(device)
    omega = _ACENTRIC_T.to(device)

    Tr = T_K[:, None] / Tc[None, :]  # (N, 15)
    alpha = _pr_alpha_torch(Tr, omega)  # (N, 15)
    a_i = 0.45724 * _R_GAS ** 2 * Tc[None, :] ** 2 / Pc[None, :] * alpha  # (N, 15)
    b_i = 0.07780 * _R_GAS * Tc / Pc  # (15,)
    return a_i, b_i


def _solve_cubic_z_torch(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve PR-EOS cubic for compressibility factor Z.

    Z³ + c2·Z² + c1·Z + c0 = 0

    Returns (Z_vap, Z_liq) — largest and smallest real roots.
    """
    c2 = B - 1.0
    c1 = A - 2.0 * B - 3.0 * B ** 2
    c0 = -(A * B - B ** 2 - B ** 3)

    p = c1 - c2 ** 2 / 3.0
    q = c0 - c2 * c1 / 3.0 + 2.0 * c2 ** 3 / 27.0

    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

    # Three real roots when disc <= 0 (trigonometric solution)
    three_real = disc <= 0

    r = torch.sqrt(torch.clamp(-p / 3.0, min=0.0))
    cos_arg = torch.clamp(-q / (2.0 * r ** 3 + 1e-15), -1.0, 1.0)
    phi = torch.acos(cos_arg)

    Z0 = 2.0 * r * torch.cos(phi / 3.0) - c2 / 3.0                    # largest
    Z2 = 2.0 * r * torch.cos((phi + 4.0 * math.pi) / 3.0) - c2 / 3.0  # smallest

    # One real root when disc > 0 (Cardano)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    term1 = -q / 2.0 + sqrt_disc
    term2 = -q / 2.0 - sqrt_disc
    Z_single = (
        torch.sign(term1) * torch.pow(torch.abs(term1), 1.0 / 3.0)
        + torch.sign(term2) * torch.pow(torch.abs(term2), 1.0 / 3.0)
        - c2 / 3.0
    )

    Z_vap = torch.where(three_real, Z0, Z_single)
    Z_liq = torch.where(three_real, Z2, Z_single)

    Z_vap = torch.clamp(Z_vap, min=B + 1e-10)
    Z_liq = torch.clamp(Z_liq, min=B + 1e-10)

    return Z_vap, Z_liq


def _fugacity_coefficients_torch(
    P_kPa: torch.Tensor,
    T_K: torch.Tensor,
    z: torch.Tensor,
    phase: str = "vapour",
) -> torch.Tensor:
    """Compute ln(φ_i) via PR-EOS.

    Args:
        P_kPa: pressure (kPa), shape (N,)
        T_K: temperature (K), shape (N,)
        z: mole fractions, shape (N, 15)
        phase: 'vapour' (largest Z) or 'liquid' (smallest Z)

    Returns:
        ln_phi: log fugacity coefficients, shape (N, 15)
    """
    device = P_kPa.device
    one_minus_k = _ONE_MINUS_K.to(device)
    b_i = 0.07780 * _R_GAS * _TC_K_T.to(device) / _PC_ATM_T.to(device)  # (15,)
    Tc = _TC_K_T.to(device)
    Pc = _PC_ATM_T.to(device)
    omega = _ACENTRIC_T.to(device)

    P_atm = P_kPa.to(torch.float64) / 101.325
    T = T_K.to(torch.float64)
    z = z.to(torch.float64).clamp_min(1e-15)
    N = P_atm.shape[0]

    # Pure-component a_i, b_i
    Tr = T[:, None] / Tc[None, :]
    alpha = _pr_alpha_torch(Tr, omega)
    a_i = 0.45724 * _R_GAS ** 2 * Tc[None, :] ** 2 / Pc[None, :] * alpha  # (N, 15)

    sqrt_a = torch.sqrt(a_i)  # (N, 15)

    # Mixture a_mix (vectorized double sum)
    M_ij = sqrt_a[:, :, None] * sqrt_a[:, None, :] * one_minus_k[None, :, :]  # (N, 15, 15)
    a_mix = (z[:, :, None] * z[:, None, :] * M_ij).sum(dim=(1, 2))  # (N,)

    # Mixture b_mix
    b_mix = z @ b_i  # (N,)

    A_mix = a_mix * P_atm / (_R_GAS * T) ** 2
    B_mix = b_mix * P_atm / (_R_GAS * T)

    Z_vap, Z_liq = _solve_cubic_z_torch(A_mix, B_mix)
    Z = Z_vap if phase == "vapour" else Z_liq

    # ln(φ_i) for each component (vectorized)
    weighted = z * sqrt_a  # (N, 15)
    S = weighted @ one_minus_k  # (N, 15): S[n,i] = Σ_j z[n,j]·sqrt_a[n,j]·(1 - k_ij)
    sum_zj_aij = sqrt_a * S  # (N, 15): Σ_j z_j·(1-k_ij)·sqrt_a_i·sqrt_a_j

    b_i_b = b_i[None, :] / b_mix[:, None]  # (N, 15)
    A_B = A_mix[:, None] / (B_mix[:, None] + 1e-15)  # (N, 1)

    Z_B = Z[:, None] + (1.0 + _SQRT2) * B_mix[:, None]
    Z_B2 = Z[:, None] + (1.0 - _SQRT2) * B_mix[:, None] + 1e-15

    ln_phi = (
        b_i_b * (Z[:, None] - 1.0)
        - torch.log(Z[:, None] - B_mix[:, None] + 1e-15)
        - A_B / (2.0 * _SQRT2) * (2.0 * sum_zj_aij / (a_mix[:, None] + 1e-15) - b_i_b)
        * torch.log(torch.clamp(Z_B / Z_B2, min=1e-15))
    )
    # Clamp to prevent extreme values from destabilizing training
    ln_phi = torch.clamp(ln_phi, min=-20.0, max=20.0)
    return ln_phi.to(torch.float32)


def _gibbs_mixing_energy_torch(
    P_kPa: torch.Tensor,
    T_K: torch.Tensor,
    z: torch.Tensor,
    phase: str = "vapour",
) -> torch.Tensor:
    """Dimensionless Gibbs free energy of mixing: ΔG_mix/(RT).

    G/(RT) = Σ_i z_i * ln(z_i * φ_i)
    """
    z = z.clamp_min(1e-15)
    ln_phi = _fugacity_coefficients_torch(P_kPa, T_K, z, phase)
    return (z * (torch.log(z) + ln_phi)).sum(dim=-1)


# ===========================================================================
# Thermodynamic consistency losses
# ===========================================================================


def rachford_rice_loss(
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Rachford-Rice residual loss (Huber for robustness).

    RR(β) = Σ_i z_i (K_i - 1) / (1 + β (K_i - 1))

    Uses smooth L1 (Huber) loss with dead-zone below 1e-4 to focus
    on eliminating large violations rather than fine-tuning small ones.
    """
    beta = pred["beta"].squeeze(-1)
    z = batch["z"]
    K = pred.get("K", pred["y"] / (pred["x"].clamp_min(eps)))

    beta_expanded = beta.unsqueeze(-1)
    denom = 1.0 + beta_expanded * (K - 1.0)
    # Clip denominator away from zero to prevent singularities
    denom = torch.where(denom.abs() < 1e-6, torch.sign(denom) * 1e-6, denom)

    rr = (z * (K - 1.0) / denom).sum(dim=-1)

    # Dead-zone: zero penalty for |RR| < 1e-4, linear beyond
    rr_abs = rr.abs()
    violation = torch.clamp(rr_abs - 1e-4, min=0.0)

    # Huber (smooth L1) on the violation
    delta = 0.1
    huber = torch.where(
        violation < delta,
        0.5 * violation ** 2,
        delta * (violation - 0.5 * delta),
    )
    return huber.mean()


def gibbs_consistency_loss(
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize two-phase Gibbs > single-phase Gibbs.

    Computes PR-EOS Gibbs free energy of mixing for:
      - Single-phase feed: G(z), using vapour root
      - Two-phase split:   (1-β)·G(x, liquid root) + β·G(y, vapour root)

    Loss = mean(max(0, G_two - G_single)).  Includes NaN guard and
    gradient clipping for numerical stability.
    """
    P_kPa = batch["pt_raw"][:, 0]
    T_C = batch["pt_raw"][:, 1]
    T_K = T_C + 273.15

    z = batch["z"]
    x = pred["x"]
    y = pred["y"]
    beta = pred["beta"].squeeze(-1)

    g_single = _gibbs_mixing_energy_torch(P_kPa, T_K, z, phase="vapour")
    g_x = _gibbs_mixing_energy_torch(P_kPa, T_K, x, phase="liquid")
    g_y = _gibbs_mixing_energy_torch(P_kPa, T_K, y, phase="vapour")

    # NaN guard: if any Gibbs energy is NaN, fall back to zero loss for this batch
    if torch.isnan(g_single).any() or torch.isnan(g_x).any() or torch.isnan(g_y).any():
        return beta.new_tensor(0.0)

    g_two = (1.0 - beta) * g_x + beta * g_y

    dg = g_two - g_single
    # Clip extreme values for gradient stability
    dg = torch.clamp(dg, min=-10.0, max=10.0)
    violation = torch.clamp(dg, min=0.0)
    return violation.mean()


# ===========================================================================
# Original loss functions
# ===========================================================================


def _log_k(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log((y + eps) / (x + eps))


def phase_consistency_loss(batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], eps: float) -> torch.Tensor:
    beta_true = batch["beta"]
    x_true = batch["x"].clamp_min(eps)
    y_true = batch["y"].clamp_min(eps)
    x_pred = pred["x"].clamp_min(eps)
    y_pred = pred["y"].clamp_min(eps)

    two_phase = ((beta_true > 1.0e-5) & (beta_true < 1.0 - 1.0e-5)).squeeze(-1)
    if two_phase.any():
        two_phase_loss = F.smooth_l1_loss(_log_k(x_pred[two_phase], y_pred[two_phase], eps), _log_k(x_true[two_phase], y_true[two_phase], eps))
    else:
        two_phase_loss = x_pred.new_tensor(0.0)

    single_phase = ~two_phase
    if single_phase.any():
        single_phase_loss = F.mse_loss(x_pred[single_phase], y_pred[single_phase])
    else:
        single_phase_loss = x_pred.new_tensor(0.0)

    return two_phase_loss + single_phase_loss


def latent_manifold_loss(batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    latent = pred["latent"]
    norm_loss = latent.pow(2).mean()
    if latent.shape[0] < 3:
        return norm_loss

    state = torch.cat([batch["input"][:, :2], batch["z"]], dim=-1)
    distance = torch.cdist(state, state, p=2)
    distance = distance + torch.eye(distance.shape[0], device=distance.device) * 1.0e6
    nearest = torch.argmin(distance, dim=1)
    latent_distance = (latent - latent[nearest]).pow(2).sum(dim=-1)
    state_distance = distance.gather(1, nearest[:, None]).squeeze(1).clamp_min(1.0e-3)
    smooth_loss = (latent_distance / state_distance).mean()
    return norm_loss + 0.1 * smooth_loss


def boundary_phase_loss(batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    beta_true = batch["beta"]
    beta_pred = pred["beta"]
    single_phase = (beta_true <= 1.0e-6) | (beta_true >= 1.0 - 1.0e-6)
    if not single_phase.any():
        return beta_pred.new_tensor(0.0)
    return F.mse_loss(beta_pred[single_phase], beta_true[single_phase]) + F.mse_loss(pred["x"][single_phase.squeeze(-1)], pred["y"][single_phase.squeeze(-1)])


def composition_smoothness_loss(
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    config: PGAEConfig,
) -> torch.Tensor:
    """Penalize large |Δβ| / ‖Δz‖ — targets adversarial sensitivity.

    For nearest-neighbor pairs in z-space with similar (P,T), penalize
    excessive β change relative to composition change. This regularises
    the β(z) mapping to be locally Lipschitz-smooth, directly addressing
    the high-P adversarial sensitivity where tiny Δz causes β 0→1 flip.
    """
    z = batch["z"]
    P = batch["pt_raw"][:, 0]
    beta_pred = pred["beta"].squeeze(-1)
    n = z.shape[0]

    if n < 3:
        return beta_pred.new_tensor(0.0)

    # Find 2 nearest neighbors per sample in z-space
    dist_z = torch.cdist(z, z, p=2)
    dist_z = dist_z + torch.eye(n, device=dist_z.device) * 1e9  # exclude self
    _, nn_idx = torch.topk(dist_z, k=min(3, n - 1), dim=1, largest=False)

    # Also require similar P (within 5000 kPa) to isolate z→β from P→β
    P_diff = (P.unsqueeze(0) - P.unsqueeze(1)).abs()

    total_penalty = beta_pred.new_tensor(0.0)
    count = 0
    for i in range(n):
        for j_idx in range(nn_idx.shape[1]):
            j = int(nn_idx[i, j_idx])
            if P_diff[i, j] < 5000.0:  # similar P
                dbeta = (beta_pred[i] - beta_pred[j]).abs()
                dz = dist_z[i, j]  # original (before +1e9)
                # Recompute proper dz
                dz_real = (z[i] - z[j]).norm(p=2)
                if dz_real > 1e-6:
                    ratio = dbeta / dz_real
                    # Penalize ratios > 20 (β change > 0.2 per 0.01 Δz)
                    excess = torch.clamp(ratio - 20.0, min=0.0)
                    total_penalty = total_penalty + excess
                    count += 1

    if count == 0:
        return beta_pred.new_tensor(0.0)
    return total_penalty / count


def monotonicity_loss(batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config: PGAEConfig) -> torch.Tensor:
    """Penalize ∂β/∂P > 0 using k-NN composition matching + continuous softplus.

    For each valid sample i, find its k nearest neighbors in (z, T/100) space.
    For each pair where ΔP exceeds mono_min_dp: if P_i > P_j then β_i should be < β_j.
    Violations are penalized with softplus(sign(ΔP) * Δβ) — smooth, continuous,
    and ~linear in violation magnitude when positive, ~0 when negative.

    Enhanced with high-P double-weighting above mono_high_p_boost.
    """
    P = batch["pt_raw"][:, 0]          # (N,)
    T = batch["pt_raw"][:, 1]          # (N,)
    z = batch["z"]                     # (N, nc)
    beta_pred = pred["beta"].squeeze(-1)  # (N,)
    beta_true = batch["beta"].squeeze(-1)  # (N,)

    # Filter: only two-phase samples below T threshold
    two_phase = (beta_true > config.eps) & (beta_true < 1.0 - config.eps)
    low_T = T < config.mono_t_max
    valid = two_phase & low_T

    if valid.sum() < 3:
        return beta_pred.new_tensor(0.0)

    # Build state vector: z + normalized T
    T_norm = T[valid].unsqueeze(-1) / 100.0  # (M, 1)
    state = torch.cat([z[valid], T_norm], dim=-1)  # (M, nc+1)

    P_valid = P[valid]        # (M,)
    beta_valid = beta_pred[valid]  # (M,)
    M = state.shape[0]
    k = min(config.mono_k_neighbors, M - 1)

    # k-NN in (z, T) space
    dist = torch.cdist(state, state, p=2)
    dist = dist + torch.eye(M, device=dist.device) * 1e9
    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)  # (M, k)

    total_penalty = beta_pred.new_tensor(0.0)
    pair_count = 0

    for i in range(M):
        Pi = P_valid[i]
        betai = beta_valid[i]
        for j_idx in range(k):
            j = int(nn_idx[i, j_idx])
            dP = Pi - P_valid[j]
            if dP.abs() < config.mono_min_dp:
                continue

            dbeta = betai - beta_valid[j]
            # softplus(sign(dP) * dbeta):
            #   dP > 0, dbeta > 0 → penalty ≈ dbeta   (violation: higher P but higher β)
            #   dP > 0, dbeta < 0 → penalty ≈ 0        (correct: higher P, lower β)
            #   dP < 0, dbeta < 0 → penalty ≈ -dbeta   (violation: lower P but lower β)
            #   dP < 0, dbeta > 0 → penalty ≈ 0        (correct: lower P, higher β)
            violation = F.softplus(torch.sign(dP) * dbeta)

            # High-P boost: double penalty above mono_high_p_boost
            if Pi > config.mono_high_p_boost or P_valid[j] > config.mono_high_p_boost:
                violation = violation * 2.0

            total_penalty = total_penalty + violation
            pair_count += 1

    if pair_count == 0:
        return beta_pred.new_tensor(0.0)
    return total_penalty / pair_count


def k_p_monotonicity_loss(
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    config: PGAEConfig,
) -> torch.Tensor:
    """Penalize ∂logK/∂P > 0 — the root cause of β-P inversions.

    Physical basis: (∂ ln K_i / ∂ P)_T ≈ -(V_i^vap - V_i^liq) / RT < 0
    for sub-critical temperatures.  When logK violates this, the Newton-RR
    solver produces β values that can jump 0→1→0 with increasing P.

    Uses the same k-NN composition-matching strategy as monotonicity_loss.
    For each valid pair where P differs by ≥ mono_min_dp: if P_i > P_j,
    then logK_i_c should be < logK_j_c for every component c.
    Penalty = mean over (pairs, components) of softplus(sign(ΔP) * ΔlogK).
    """
    P = batch["pt_raw"][:, 0]       # (N,)
    T = batch["pt_raw"][:, 1]       # (N,)
    z = batch["z"]                  # (N, nc)
    log_K = pred["log_K"]           # (N, nc)
    beta_true = batch["beta"].squeeze(-1)  # (N,)

    # Apply to ALL samples below T threshold (not just two-phase),
    # because K-value P-monotonicity is a pure-component thermodynamic constraint
    low_T = T < config.mono_t_max
    # Exclude pure single-phase endpoints where K may be unreliable
    not_endpoint = (beta_true > 1e-4) & (beta_true < 1.0 - 1e-4)
    valid = low_T & not_endpoint

    if valid.sum() < 3:
        return log_K.new_tensor(0.0)

    T_norm = T[valid].unsqueeze(-1) / 100.0
    state = torch.cat([z[valid], T_norm], dim=-1)
    P_valid = P[valid]
    logK_valid = log_K[valid]
    M = state.shape[0]
    k = min(config.mono_k_neighbors, M - 1)

    dist = torch.cdist(state, state, p=2)
    dist = dist + torch.eye(M, device=dist.device) * 1e9
    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

    total_penalty = log_K.new_tensor(0.0)
    pair_count = 0

    for i in range(M):
        Pi = P_valid[i]
        logKi = logK_valid[i]  # (nc,)
        for j_idx in range(k):
            j = int(nn_idx[i, j_idx])
            dP = Pi - P_valid[j]
            if dP.abs() < config.mono_min_dp:
                continue

            dlogK = logKi - logK_valid[j]  # (nc,)
            # Penalize per-component: softplus(sign(dP) * dlogK_c)
            violation_c = F.softplus(torch.sign(dP) * dlogK)  # (nc,)
            total_penalty = total_penalty + violation_c.mean()
            pair_count += 1

    if pair_count == 0:
        return log_K.new_tensor(0.0)
    return total_penalty / pair_count


def k_value_loss(
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Direct K-value supervision in log space.

    K_i = y_i / x_i — equilibrium ratio for each component.
    Log-K MSE ensures the model learns physically correct K-values
    rather than just producing consistent (x, y) through normalization.
    """
    K_pred = pred["K"]
    K_true = (batch["y"] + eps) / (batch["x"] + eps)
    return F.mse_loss(
        torch.log(K_pred.clamp_min(eps)),
        torch.log(K_true.clamp_min(eps)),
    )


def pgae_loss(batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config: PGAEConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    beta_true = batch["beta"]
    x_true = batch["x"]
    y_true = batch["y"]
    z = batch["z"]

    beta_pred = pred["beta"]
    x_pred = pred["x"]
    y_pred = pred["y"]
    z_reconstructed = pred["z_flash"]

    recon_loss = F.mse_loss(pred["output"], batch["target"])
    beta_loss = F.mse_loss(beta_pred, beta_true)
    x_loss = F.mse_loss(x_pred, x_true)
    y_loss = F.mse_loss(y_pred, y_true)
    mass_loss = F.mse_loss(z_reconstructed, z)
    phase_loss = phase_consistency_loss(batch, pred, config.eps)
    latent_loss = latent_manifold_loss(batch, pred)
    boundary_loss = boundary_phase_loss(batch, pred)
    mono_loss = monotonicity_loss(batch, pred, config)
    k_mono_loss = k_p_monotonicity_loss(batch, pred, config)
    rr_loss = rachford_rice_loss(batch, pred, config.eps)
    gibbs_loss = gibbs_consistency_loss(batch, pred) if config.lambda_gibbs > 0 else rr_loss.new_tensor(0.0)
    kl_loss = k_value_loss(batch, pred, config.eps)
    smooth_loss = composition_smoothness_loss(batch, pred, config) if config.lambda_smooth > 0 else rr_loss.new_tensor(0.0)

    total = (
        recon_loss
        + config.lambda_beta * beta_loss
        + config.lambda_mass * mass_loss
        + config.lambda_phase * phase_loss
        + config.lambda_latent * latent_loss
        + config.lambda_boundary * boundary_loss
        + config.lambda_mono * mono_loss
        + config.lambda_k_mono * k_mono_loss
        + config.lambda_rr * rr_loss
        + config.lambda_gibbs * gibbs_loss
        + config.lambda_k * kl_loss
        + config.lambda_smooth * smooth_loss
    )

    with torch.no_grad():
        phase_split_error = torch.mean(torch.abs((y_pred - x_pred) - (y_true - x_true)))
        thermo_residual = torch.mean(torch.abs(_log_k(x_pred, y_pred, config.eps) - _log_k(x_true, y_true, config.eps)))
        x_norm_residual = torch.mean(torch.abs(x_pred.sum(dim=-1) - 1.0))
        y_norm_residual = torch.mean(torch.abs(y_pred.sum(dim=-1) - 1.0))
        metrics = {
            "loss": total.detach(),
            "recon_loss": recon_loss.detach(),
            "beta_loss": beta_loss.detach(),
            "x_loss": x_loss.detach(),
            "y_loss": y_loss.detach(),
            "mass_loss": mass_loss.detach(),
            "phase_loss": phase_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
            "mono_loss": mono_loss.detach(),
            "k_mono_loss": k_mono_loss.detach(),
            "rr_loss": rr_loss.detach(),
            "gibbs_loss": gibbs_loss.detach(),
            "k_loss": kl_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "beta_mae": torch.mean(torch.abs(beta_pred - beta_true)),
            "x_mae": torch.mean(torch.abs(x_pred - x_true)),
            "y_mae": torch.mean(torch.abs(y_pred - y_true)),
            "mass_residual": torch.mean(torch.abs(z_reconstructed - z)),
            "x_norm_residual": x_norm_residual,
            "y_norm_residual": y_norm_residual,
            "phase_split_error": phase_split_error,
            "thermo_consistency_residual": thermo_residual,
        }
    return total, metrics
