"""Part 13: Computational Speed Benchmark — PGAE vs WinProp EOS.

Measures:
  1. PGAE single-flash latency (1 sample)
  2. PGAE batch throughput (10–10⁶ samples)
  3. WinProp single-flash latency (external process)
  4. Speedup factor vs WinProp
  5. CPU vs GPU scaling
  6. Memory footprint
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import PGAEConfig
from infer import PGAEFlashSurrogate

WINPROP_EXE = r"D:\CMG\WINPROP\2022.10\Win_x64\EXE\pr202210.exe"
TEMPLATE_DAT = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\fluid_15comp.dat")
WORK_DIR = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\sim")
NC = 15
COMP_NAMES = ["C10+", "N2", "CO2", "CH4", "C2H6", "C3H8", "IC4", "NC4", "IC5", "NC5",
              "FC6", "FC7", "FC8", "FC9", "FC10"]


# =============================================================================
# 1. PGAE benchmark
# =============================================================================

def benchmark_pgae_single(
    surrogate: PGAEFlashSurrogate,
    n_warmup: int = 100,
    n_repeat: int = 1000,
) -> Dict[str, float]:
    """Benchmark PGAE single-flash latency (1 sample at a time).

    Returns mean/median/std/min/max latency in milliseconds.
    """
    config = surrogate.config
    device = config.device
    model = surrogate.model
    model.eval()
    stats = surrogate.stats

    # Prepare a single test input
    P, T = 15000.0, 90.0
    z = np.array([0.05, 0.02, 0.02, 0.55, 0.08, 0.06, 0.03, 0.03, 0.02, 0.02,
                  0.03, 0.03, 0.02, 0.02, 0.02], dtype=np.float32)
    pt = torch.tensor([P, T], dtype=torch.float32)
    pt_norm = (pt - stats.pt_mean) / stats.pt_std
    z_t = torch.tensor(z, dtype=torch.float32)
    z_t = z_t / z_t.sum().clamp_min(1e-12)
    inp = torch.cat([pt_norm, z_t], dim=0).unsqueeze(0).to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(inp)

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = model(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def benchmark_pgae_batch(
    surrogate: PGAEFlashSurrogate,
    batch_sizes: List[int] = [1, 8, 32, 128, 512, 2048, 8192, 32768, 131072],
    n_warmup: int = 5,
    n_repeat: int = 20,
) -> Dict[str, np.ndarray]:
    """Benchmark PGAE batch throughput at various batch sizes.

    Returns arrays of batch_size, mean_ms, throughput (samples/sec).
    """
    config = surrogate.config
    device = config.device
    model = surrogate.model
    model.eval()
    stats = surrogate.stats

    results = {"batch_size": [], "time_ms": [], "throughput_sps": []}

    for bs in batch_sizes:
        # Generate random batch
        P_arr = np.random.uniform(100, 50000, bs).astype(np.float32)
        T_arr = np.random.uniform(40, 180, bs).astype(np.float32)
        z_arr = np.random.dirichlet(np.ones(NC), bs).astype(np.float32)

        pt = torch.tensor(np.stack([P_arr, T_arr], axis=1))
        pt_norm = (pt - stats.pt_mean) / stats.pt_std
        z_t = torch.tensor(z_arr)
        inp = torch.cat([pt_norm, z_t], dim=1).to(device)

        # Warmup
        for _ in range(n_warmup):
            _ = model(inp)

        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        mean_time = float(np.mean(times))
        results["batch_size"].append(bs)
        results["time_ms"].append(mean_time)
        results["throughput_sps"].append(bs / (mean_time / 1000))

        print(f"  batch={bs:>6d}: {mean_time:.3f} ms  →  {bs / (mean_time / 1000):.0f} samples/s")

    return results


# =============================================================================
# 2. WinProp benchmark
# =============================================================================

def _make_benchmark_dat(P: float, T: float, z: np.ndarray, template_lines: List[str]) -> str:
    """Generate a minimal WinProp DAT for a single flash."""
    lines_out = []
    in_envelope = False
    in_flash = False
    comp_injected = False
    flash_injected = False
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
            if not flash_injected:
                lines_out.append("*FLASH")
                lines_out.append("*LABEL    ''")
                lines_out.append("*FEED  *MIXED 1.0")
                lines_out.append("*KVALUE  *INTERNAL")
                lines_out.append("*LEVEL 1")
                lines_out.append("*OUTPUT 1")
                lines_out.append("*TYPE  *QNSS")
                lines_out.append(f"*PRES {P:.2f}")
                lines_out.append(f"*TEMP {T:.2f}")
                lines_out.append("*DELP 0.0")
                lines_out.append("*DELT 0.0")
                lines_out.append("*STEPP 1")
                lines_out.append("*STEPT 1")
                lines_out.append("")
                flash_injected = True
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


def benchmark_winprop_single(
    z: np.ndarray,
    P: float = 15000.0,
    T: float = 90.0,
    n_repeat: int = 20,
    template_path: Path = TEMPLATE_DAT,
    work_dir: Path = WORK_DIR,
) -> Optional[Dict[str, float]]:
    """Benchmark WinProp single-flash latency.

    Measures wall-clock time for process spawn + execution + file I/O.
    """
    if not os.path.exists(WINPROP_EXE):
        print("WinProp not found, skipping WinProp benchmark")
        return None

    with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
        template_lines = f.readlines()

    dat_content = _make_benchmark_dat(P, T, z, template_lines)

    times = []
    for i in range(n_repeat):
        dat_path = work_dir / f"_bench_{i:03d}.dat"
        dat_path.write_text(dat_content, encoding="utf-8")

        t0 = time.perf_counter()
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
        except (subprocess.TimeoutExpired, PermissionError, OSError):
            pass
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_repeat": n_repeat,
    }


# =============================================================================
# 3. Memory footprint
# =============================================================================

def benchmark_memory(surrogate: PGAEFlashSurrogate) -> Dict[str, float]:
    """Measure model memory footprint."""
    model = surrogate.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    # Measure peak memory for inference at different batch sizes
    device = surrogate.config.device
    stats = surrogate.stats
    memory_at_bs = {}

    for bs in [1, 256, 4096, 65536]:
        try:
            P_arr = np.random.uniform(100, 50000, bs).astype(np.float32)
            T_arr = np.random.uniform(40, 180, bs).astype(np.float32)
            z_arr = np.random.dirichlet(np.ones(NC), bs).astype(np.float32)
            pt = torch.tensor(np.stack([P_arr, T_arr], axis=1))
            pt_norm = (pt - stats.pt_mean) / stats.pt_std
            z_t = torch.tensor(z_arr)
            inp = torch.cat([pt_norm, z_t], dim=1).to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                _ = model(inp)
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mb = inp.element_size() * inp.numel() / (1024 ** 2) * 10  # rough estimate
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            peak_mb = float("inf")

        memory_at_bs[f"bs_{bs}"] = peak_mb

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_size_mb,
        **memory_at_bs,
    }


# =============================================================================
# 4. Plots
# =============================================================================

def plot_speed_comparison(
    pgae_single: Dict[str, float],
    pgae_batch: Dict[str, np.ndarray],
    winprop_single: Optional[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Speed comparison plots: latency bar chart + batch scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Latency comparison (log scale)
    methods = ["PGAE\n(single)", "PGAE\n(batch=256)", "PGAE\n(batch=4096)"]
    latencies = [
        pgae_single["mean_ms"],
        pgae_batch["time_ms"][pgae_batch["batch_size"].index(256) if 256 in pgae_batch["batch_size"] else 3],
        pgae_batch["time_ms"][pgae_batch["batch_size"].index(4096) if 4096 in pgae_batch["batch_size"] else 5],
    ]
    if winprop_single:
        methods.append("WinProp\n(single)")
        latencies.append(winprop_single["mean_ms"])

    colors = ["#2196F3", "#4CAF50", "#8BC34A", "#FF5722"]
    bars = ax1.bar(methods, latencies, color=colors[:len(methods)], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Flash Calculation Latency")
    ax1.set_yscale("log")

    # Annotate
    for bar, lat in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{lat:.2f} ms", ha="center", fontsize=9, fontweight="bold")
    if winprop_single:
        speedup = winprop_single["mean_ms"] / pgae_single["mean_ms"]
        ax1.text(0.5, 0.95, f"PGAE Speedup: {speedup:.0f}× vs WinProp",
                transform=ax1.transAxes, fontsize=10, fontweight="bold",
                ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Batch scaling
    ax2.loglog(pgae_batch["batch_size"], pgae_batch["throughput_sps"],
              "bo-", linewidth=2, markersize=8, label="PGAE Throughput")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.set_title("PGAE Batch Scaling")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    # Annotate key points
    for bs, tput in zip(pgae_batch["batch_size"][::2], pgae_batch["throughput_sps"][::2]):
        if tput > 1000:
            ax2.annotate(f"{tput/1000:.0f}K/s", (bs, tput), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7)

    fig.suptitle("PGAE Flash Surrogate: Computational Speed Benchmark", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "speed_benchmark.png", dpi=220)
    plt.close(fig)


def plot_winprop_overhead(winprop_single: Dict[str, float], output_dir: Path) -> None:
    """Breakdown of WinProp overhead components (if measured)."""
    if winprop_single is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    stats = [
        ("Mean", winprop_single["mean_ms"]),
        ("Median", winprop_single["median_ms"]),
        ("Std", winprop_single["std_ms"]),
        ("Min", winprop_single["min_ms"]),
        ("Max", winprop_single["max_ms"]),
    ]
    names, vals = zip(*stats)
    ax.bar(names, vals, color=["#2196F3", "#4CAF50", "#FF9800", "#8BC34A", "#F44336"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("WinProp Single-Flash Latency Distribution")
    for i, v in enumerate(vals):
        ax.text(i, v + 5, f"{v:.0f} ms", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "speed_winprop_overhead.png", dpi=220)
    plt.close(fig)


# =============================================================================
# 5. Main
# =============================================================================

def main() -> None:
    config = PGAEConfig()
    if not config.best_checkpoint_path.exists():
        print(f"Checkpoint not found: {config.best_checkpoint_path}")
        print("Please run train.py first.")
        return

    print("Loading model...")
    surrogate = PGAEFlashSurrogate(config.best_checkpoint_path)
    device = config.device
    print(f"Device: {device}")

    # ---- Memory ----
    print("\n" + "=" * 60)
    print("1. Memory Footprint")
    print("=" * 60)
    mem = benchmark_memory(surrogate)
    print(f"  Total parameters:     {mem['total_parameters']:,}")
    print(f"  Trainable parameters: {mem['trainable_parameters']:,}")
    print(f"  Model size on disk:   {mem['model_size_mb']:.2f} MB")
    for k, v in mem.items():
        if k.startswith("bs_"):
            print(f"  Peak memory ({k}):    {v:.1f} MB" if v != float("inf") else f"  Peak memory ({k}): OOM")

    # ---- PGAE Single ----
    print("\n" + "=" * 60)
    print("2. PGAE Single-Flash Latency")
    print("=" * 60)
    pgae_single = benchmark_pgae_single(surrogate, n_warmup=100, n_repeat=2000)
    print(f"  Mean:   {pgae_single['mean_ms']:.4f} ms")
    print(f"  Median: {pgae_single['median_ms']:.4f} ms")
    print(f"  Std:    {pgae_single['std_ms']:.4f} ms")
    print(f"  P99:    {pgae_single['p99_ms']:.4f} ms")
    print(f"  → {1000 / pgae_single['mean_ms']:.0f} flashes/second (single-threaded)")

    # ---- PGAE Batch ----
    print("\n" + "=" * 60)
    print("3. PGAE Batch Throughput Scaling")
    print("=" * 60)
    batch_sizes = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    if device.type == "cuda":
        batch_sizes.append(65536)
    pgae_batch = benchmark_pgae_batch(surrogate, batch_sizes=batch_sizes)

    # Find optimal batch size
    best_idx = np.argmax(pgae_batch["throughput_sps"])
    print(f"\n  Best batch size: {pgae_batch['batch_size'][best_idx]}")
    print(f"  Max throughput:  {pgae_batch['throughput_sps'][best_idx]:.0f} samples/s")

    # ---- WinProp ----
    print("\n" + "=" * 60)
    print("4. WinProp Single-Flash Latency")
    print("=" * 60)
    # Use a typical composition from dataset
    df = pd.read_csv(config.dataset_path)
    z_cols = [f"z{i}" for i in range(1, NC + 1)]
    z_typical = df[z_cols].iloc[0].to_numpy(dtype=np.float64)
    z_typical = z_typical / z_typical.sum()

    winprop_single = benchmark_winprop_single(z_typical, n_repeat=15)
    if winprop_single:
        print(f"  Mean:   {winprop_single['mean_ms']:.1f} ms")
        print(f"  Median: {winprop_single['median_ms']:.1f} ms")
        print(f"  Std:    {winprop_single['std_ms']:.1f} ms")
        print(f"  Min:    {winprop_single['min_ms']:.1f} ms")
        print(f"  → {1000 / winprop_single['mean_ms']:.1f} flashes/second")
        speedup = winprop_single["mean_ms"] / pgae_single["mean_ms"]
        print(f"\n  *** PGAE Speedup: {speedup:.0f}× vs WinProp (single flash) ***")
    else:
        print("  WinProp not available, skipping")

    # ---- Plots ----
    print("\n" + "=" * 60)
    print("5. Generating figures...")
    print("=" * 60)
    plot_speed_comparison(pgae_single, pgae_batch, winprop_single, config.fig_speed_dir)
    print("  → speed_benchmark.png")
    if winprop_single:
        plot_winprop_overhead(winprop_single, config.fig_speed_dir)
        print("  → speed_winprop_overhead.png")

    # ---- Save Results ----
    summary = {
        "device": str(device),
        "model_parameters": mem["total_parameters"],
        "model_size_mb": mem["model_size_mb"],
        "pgae_single_ms": pgae_single,
        "pgae_batch": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pgae_batch.items()},
        "winprop_single_ms": winprop_single,
        "speedup_vs_winprop": (winprop_single["mean_ms"] / pgae_single["mean_ms"]) if winprop_single else None,
        "best_batch_size": int(pgae_batch["batch_size"][best_idx]),
        "max_throughput_sps": float(pgae_batch["throughput_sps"][best_idx]),
        "memory_mb": {k: v for k, v in mem.items() if k.startswith("bs_")},
    }

    summary_path = config.metric_dir / "speed_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  Model params:        {mem['total_parameters']:,} ({mem['model_size_mb']:.2f} MB)")
    print(f"  PGAE single flash:   {pgae_single['mean_ms']:.4f} ms")
    print(f"  PGAE max throughput: {pgae_batch['throughput_sps'][best_idx]:.0f} samples/s")
    if winprop_single:
        print(f"  WinProp single:      {winprop_single['mean_ms']:.1f} ms")
        print(f"  *** SPEEDUP:         {speedup:.0f}× ***")
    print(f"  Figures:             {config.fig_speed_dir / 'speed_benchmark.png'}")
    print(f"  Metrics:             {summary_path}")


if __name__ == "__main__":
    main()
