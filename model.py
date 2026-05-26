from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """Linear layer with strictly non-negative weights for monotonic networks."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.weight), self.bias)


class ConditionalMonotonicNet(nn.Module):
    """Two-layer P-modulation network with structural ∂output/∂P ≥ 0.

    Architecture (ReLU for strong gradients, wide hidden for expressivity):
      h = relu(softplus(w_p1)·P + w_ctx1·latent + b1)       [hidden_dim]
      output = softplus(softplus(w_p2)·h + w_ctx2·latent + b2)  [out_dim]

    Monotonicity: ∂h/∂P = relu'(…) · softplus(w_p1) ≥ 0 (relu' ∈ {0,1})
                  ∂out/∂P = softplus'(…) · softplus(w_p2) · ∂h/∂P ≥ 0

    Final bias initialised to -10 so p_effect ≈ 4.5e-5 at start (near zero).
    This lets base_log_K learn K-values immediately; P-network grows from zero.
    """

    def __init__(self, p_dim: int = 1, ctx_dim: int = 3, out_dim: int = 15, hidden_dim: int = 32):
        super().__init__()
        self.w_p1 = nn.Parameter(torch.randn(hidden_dim, p_dim) * 0.02)
        self.w_ctx1 = nn.Parameter(torch.randn(hidden_dim, ctx_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.w_p2 = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.02)
        self.w_ctx2 = nn.Parameter(torch.randn(out_dim, ctx_dim) * 0.02)
        self.b2 = nn.Parameter(torch.full((out_dim,), -10.0))  # softplus(-10) ≈ 4.5e-5

    def forward(self, P_norm: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = F.relu(
            F.linear(P_norm, F.softplus(self.w_p1))
            + F.linear(latent, self.w_ctx1)
            + self.b1
        )
        return F.softplus(
            F.linear(h, F.softplus(self.w_p2))
            + F.linear(latent, self.w_ctx2)
            + self.b2
        )


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MLPStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.residual = ResidualBlock(out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.proj(x))


class PGAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        latent_dim: int = 3,
        nc: int = 15,
        encoder_dims: Iterable[int] = (128, 64, 32),
        decoder_dims: Iterable[int] = (32, 64, 128),
        dropout: float = 0.02,
        p_net_hidden: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nc = nc

        self.encoder = self._build_mlp(input_dim, encoder_dims, dropout)
        self.latent_head = nn.Linear(list(encoder_dims)[-1], latent_dim)
        self.decoder = self._build_mlp(latent_dim, decoder_dims, dropout)
        self.output_head = nn.Linear(list(decoder_dims)[-1], nc)

        # Conditional monotonic P-network: p_effect(P, latent) ≥ 0, monotonic ↑ in P.
        self.p_net = ConditionalMonotonicNet(p_dim=1, ctx_dim=latent_dim, out_dim=nc, hidden_dim=p_net_hidden)

    @staticmethod
    def _build_mlp(in_dim: int, dims: Iterable[int], dropout: float) -> nn.Sequential:
        layers = []
        current = in_dim
        for dim in dims:
            layers.append(MLPStage(current, dim, dropout=dropout))
            current = dim
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.latent_head(self.encoder(inputs))

    @staticmethod
    def _solve_rr_beta(z: torch.Tensor, K: torch.Tensor, n_iter: int = 15) -> torch.Tensor:
        """Solve Rachford-Rice equation for β via Newton's method.

        f(β) = Σ_i z_i (K_i - 1) / (1 + β (K_i - 1)) = 0
        """
        eps = 1e-12
        Kd = K - 1.0
        K_safe = K.clamp_min(eps)
        z = z.clamp_min(eps)

        f0 = (z * K).sum(dim=-1) - 1.0
        f1 = 1.0 - (z / K_safe).sum(dim=-1)
        two_phase = (f0 > 0.0) & (f1 < 0.0)

        beta = torch.where(
            two_phase,
            (f0 / (f0 - f1 + eps)).clamp(0.001, 0.999),
            torch.where(f0 <= 0.0, torch.zeros_like(f0), torch.ones_like(f0)),
        ).unsqueeze(-1)

        for _ in range(n_iter):
            denom = 1.0 + beta * Kd
            denom = denom.clamp_min(eps)
            f = (z * Kd / denom).sum(dim=-1, keepdim=True)
            df = -(z * Kd * Kd / (denom * denom)).sum(dim=-1, keepdim=True)
            delta = f / df.clamp_max(-eps)
            beta = beta - delta
            beta = beta.clamp(0.0, 1.0)

        beta = torch.where(
            two_phase.unsqueeze(-1),
            beta,
            torch.where(f0.unsqueeze(-1) <= 0.0, torch.zeros_like(beta), torch.ones_like(beta)),
        )
        return beta  # (N, 1)

    def decode(self, latent: torch.Tensor, z: torch.Tensor, P_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_log_K = self.output_head(self.decoder(latent))  # (N, nc)
        # Conditional monotonic P-modulation: p_effect ≥ 0, monotonic ↑ in P for fixed latent.
        log_K = base_log_K - self.p_net(P_norm, latent)
        log_K = torch.clamp(log_K, min=-15.0, max=15.0)
        K = torch.exp(log_K)

        beta = self._solve_rr_beta(z, K)

        denom = 1.0 + beta * (K - 1.0)
        denom = denom.clamp_min(1e-15)
        x_raw = z / denom
        y_raw = K * x_raw

        x = x_raw / x_raw.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        y = y_raw / y_raw.sum(dim=-1, keepdim=True).clamp_min(1e-15)

        z_flash = (1.0 - beta) * x + beta * y
        output = torch.cat([beta, x, y], dim=-1)
        return {
            "raw": base_log_K,  # raw = base_log_K (before P-modulation)
            "beta": beta,
            "K": K,
            "log_K": log_K,
            "x": x,
            "y": y,
            "z_flash": z_flash,
            "output": output,
        }

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = inputs[:, 2:]
        P_norm = inputs[:, 0:1]
        # Encoder sees [P, T, z] — full (P,T,z) information in latent.
        # P-network provides monotonic baseline; k_mono_loss enforces ∂logK/∂P < 0.
        latent = self.encode(inputs)
        decoded = self.decode(latent, z, P_norm)
        decoded["latent"] = latent
        return decoded
