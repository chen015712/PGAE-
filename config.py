from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch


@dataclass
class PGAEConfig:
    base_dir: Path = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file")
    dataset_path: Path = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\sim\pgae_dataset_merged.csv")
    output_dir: Path = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\outputs")

    input_dim: int = 17  # P_norm + T_norm + 15z (P in encoder for K-value accuracy)
    nc: int = 15
    latent_dim: int = 8  # increased from 3 for richer (P, T, z) encoding
    output_dim: int = 15  # log_K only (β solved via Newton from RR)

    hidden_encoder: tuple[int, ...] = (128, 64, 32)
    hidden_decoder: tuple[int, ...] = (32, 64, 128)
    residual_dropout: float = 0.02

    batch_size: int = 256
    epochs: int = 1000
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    val_fraction: float = 0.2
    seed: int = 2026
    num_workers: int = 0
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    early_stop_patience: int = 120
    lr_patience: int = 30
    p_net_hidden: int = 128  # hidden dim for conditional monotonic P-network (v3: 32→128)

    lambda_mass: float = 10.0
    lambda_phase: float = 0.2
    lambda_latent: float = 1.0e-3
    lambda_boundary: float = 0.05
    lambda_mono: float = 0.2  # β-P monotonicity (enforced harder since P in encoder)
    lambda_k_mono: float = 0.2  # ∂logK/∂P < 0 (enforced harder since P in encoder)
    lambda_beta: float = 1.0  # direct β supervision (v3: 0.5→1.0)
    mono_t_max: float = 120.0  # only apply mono constraints below this T (°C)
    mono_min_dp: float = 100.0  # min ΔP (kPa) for monotonicity pair check
    mono_high_p_boost: float = 20000.0  # P threshold (kPa) for double-penalty on β-P inversion
    mono_k_neighbors: int = 3  # k-NN for composition matching in mono losses
    lambda_rr: float = 0.1  # Newton β already ensures RR=0; keep as diagnostic
    lambda_gibbs: float = 0.0  # disabled: PR-EOS fugacity too unstable for gradient training
    lambda_k: float = 2.0  # direct K-value supervision in log space (v3: 1.0→2.0)
    lambda_smooth: float = 0.0  # disabled: conflicts with sharp phase transitions
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        # Figure subdirectories
        self.fig_training_dir.mkdir(parents=True, exist_ok=True)
        self.fig_thermo_dir.mkdir(parents=True, exist_ok=True)
        self.fig_phase_boundary_dir.mkdir(parents=True, exist_ok=True)
        self.fig_phase_envelope_dir.mkdir(parents=True, exist_ok=True)
        self.fig_latent_dir.mkdir(parents=True, exist_ok=True)
        self.fig_speed_dir.mkdir(parents=True, exist_ok=True)
        self.fig_reservoir_dir.mkdir(parents=True, exist_ok=True)
        self.fig_robustness_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def fig_training_dir(self) -> Path:
        return self.figure_dir / "training"

    @property
    def fig_thermo_dir(self) -> Path:
        return self.figure_dir / "thermo_checks"

    @property
    def fig_phase_boundary_dir(self) -> Path:
        return self.figure_dir / "phase_boundary"

    @property
    def fig_phase_envelope_dir(self) -> Path:
        return self.figure_dir / "phase_envelope"

    @property
    def fig_latent_dir(self) -> Path:
        return self.figure_dir / "latent_analysis"

    @property
    def fig_speed_dir(self) -> Path:
        return self.figure_dir / "speed_benchmark"

    @property
    def fig_reservoir_dir(self) -> Path:
        return self.figure_dir / "reservoir_simulator"

    @property
    def fig_robustness_dir(self) -> Path:
        return self.figure_dir / "robustness"

    @property
    def metric_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def inference_dir(self) -> Path:
        return self.output_dir / "inference"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_pgae.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last_pgae.pt"

    @property
    def history_path(self) -> Path:
        return self.metric_dir / "training_history.csv"

    @property
    def validation_metrics_path(self) -> Path:
        return self.metric_dir / "validation_metrics.json"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data
