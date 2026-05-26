"""
Master pipeline — re-run the entire PGAE project from scratch.

Run order:
  1. Data generation   (pgae_data_pipeline.py)
  2. Training           (train.py)
  3. Inference          (infer.py)
  4. Phase boundary     (phase_boundary.py)
  5. Phase envelope     (phase_envelope.py)
  6. Latent analysis    (latent_analysis.py)
  7. Speed benchmark    (speed_benchmark.py)
  8. Robustness test    (robustness_test.py)
  9. Reservoir sim      (reservoir_simulator.py)

Each step deletes its own outputs before running.
"""
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file")
OUT = BASE / "outputs"
SIM = BASE / "sim"

STEPS = [
    {
        "name": "数据生成 (pgae_data_pipeline.py)",
        "script": BASE / "pgae_data_pipeline.py",
        "clear": [SIM / "pgae_dataset.csv", SIM / "pgae_dataset_augmented.csv",
                   SIM / "pgae_dataset_merged.csv", SIM / "physics_constants.pth",
                   SIM],  # also clean .srf .out temp files
        "critical": True,
    },
    {
        "name": "模型训练 (train.py)",
        "script": BASE / "train.py",
        "clear": [OUT / "checkpoints", OUT / "metrics", OUT / "figures"],
        "critical": True,
    },
    {
        "name": "全量推断 + 热力学检验 (infer.py)",
        "script": BASE / "infer.py",
        "clear": [OUT / "inference"],
        "critical": True,
    },
    {
        "name": "相边界连续性分析 (phase_boundary.py)",
        "script": BASE / "phase_boundary.py",
        "clear": [OUT / "figures" / "phase_boundary",
                   OUT / "metrics" / "phase_boundary_summary.json"],
        "critical": False,
    },
    {
        "name": "相包络重构 (phase_envelope.py)",
        "script": BASE / "phase_envelope.py",
        "clear": [OUT / "figures" / "phase_envelope",
                   OUT / "metrics" / "phase_envelope_summary.json"],
        "critical": False,
    },
    {
        "name": "潜空间分析 (latent_analysis.py)",
        "script": BASE / "latent_analysis.py",
        "clear": [OUT / "figures" / "latent_analysis"],
        "critical": False,
    },
    {
        "name": "计算效率对比 (speed_benchmark.py)",
        "script": BASE / "speed_benchmark.py",
        "clear": [OUT / "figures" / "speed_benchmark"],
        "critical": False,
    },
    {
        "name": "鲁棒性测试 (robustness_test.py)",
        "script": BASE / "robustness_test.py",
        "clear": [OUT / "figures" / "robustness",
                   OUT / "metrics" / "robustness_summary.json"],
        "critical": False,
    },
    {
        "name": "油藏模拟验证 (reservoir_simulator.py)",
        "script": BASE / "reservoir_simulator.py",
        "clear": [OUT / "figures" / "reservoir_simulator"],
        "critical": False,
    },
]


def clear_paths(paths: list[Path]) -> None:
    """Delete files/directories before a step runs."""
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            if p.is_dir():
                # For SIM dir, only delete .srf and .out temp files
                if p.name == "sim":
                    for f in p.glob("*.srf"):
                        f.unlink(missing_ok=True)
                    for f in p.glob("tmp*.dat"):
                        f.unlink(missing_ok=True)
                    for f in p.glob("_env_test*"):
                        f.unlink(missing_ok=True)
                    # Also clean any .out files that aren't the template
                    for f in p.glob("*.out"):
                        if f.name not in ("fluid_15comp.out",):
                            f.unlink(missing_ok=True)
                    print(f"  Cleaned temp files in: {p}")
                else:
                    shutil.rmtree(p)
                    print(f"  Deleted dir: {p}")
            else:
                p.unlink()
                print(f"  Deleted file: {p}")
        except Exception as e:
            print(f"  WARNING: Could not clear {p}: {e}")


def run_step(step: dict) -> bool:
    """Run a single pipeline step. Returns True on success."""
    print("\n" + "=" * 70)
    print(f">>> STEP: {step['name']}")
    print("=" * 70)

    # 1) Clear previous outputs
    print("  Clearing previous outputs...")
    clear_paths(step["clear"])

    # 2) Run the script
    script = step["script"]
    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return not step["critical"]

    print(f"  Running: {script}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE),
            capture_output=False,
            text=True,
            timeout=7200,  # 2-hour timeout per step
        )
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            print(f"  SUCCESS — elapsed: {elapsed:.1f} s")
            return True
        else:
            print(f"  FAILED (exit code {result.returncode}) — elapsed: {elapsed:.1f} s")
            return not step["critical"]
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  TIMEOUT after {elapsed:.1f} s")
        return not step["critical"]
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  EXCEPTION: {e} — elapsed: {elapsed:.1f} s")
        return not step["critical"]


def main() -> None:
    print("=" * 70)
    print("  PGAE FULL PIPELINE — RE-RUN FROM SCRATCH")
    print(f"  Working directory: {BASE}")
    print(f"  Python: {sys.executable}")
    print("=" * 70)

    # Ensure output dir exists
    OUT.mkdir(parents=True, exist_ok=True)

    total = len(STEPS)
    passed = 0
    failed = 0
    skipped = 0

    for i, step in enumerate(STEPS, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{total}] {step['name']}")
        print(f"{'─' * 70}")
        ok = run_step(step)
        if ok:
            passed += 1
        else:
            if step["critical"]:
                print(f"\n  CRITICAL STEP FAILED — stopping pipeline.")
                failed += 1
                break
            else:
                print(f"\n  Non-critical step failed — continuing.")
                failed += 1

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Passed: {passed}/{total}  Failed: {failed}  Skipped: {skipped}")
    print(f"  Output directory: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
