"""
Reorganize outputs/ folder according to the parts defined in
"基于PGAE的多组分流体相平衡研究_新研究方向详细说明书".

Mapping:
  Part 07 (训练策略与超参数优化): checkpoints/, metrics/training_history, metrics/validation_metrics
  Part 08 (基础精度验证实验): figures/training/, inference/
  Part 09 (热力学一致性验证): figures/thermo_checks/, root thermo figures
  Part 10 (相边界连续性验证): figures/phase_boundary/, metrics/phase_boundary_summary
  Part 11 (相包络重建): figures/phase_envelope/, metrics/phase_envelope_summary
  Part 12 (潜空间流形分析): figures/latent_analysis/, metrics/latent_analysis_summary
  Part 13 (计算速度基准测试): figures/speed_benchmark/, metrics/speed_benchmark_summary
  Part 14 (油藏模拟器耦合验证): figures/reservoir_simulator/, metrics/simulator/
  Part 15 (鲁棒性测试): figures/robustness/, metrics/robustness_summary
  Part 16 (论文撰写策略): generated docx files
"""
import shutil
from pathlib import Path

BASE = Path(r"C:\Users\Lenovo\Desktop\fluid_15comp_file\outputs")

MAPPING = {
    "第07部分_训练策略与超参数优化": {
        "dirs": [
            BASE / "checkpoints",
        ],
        "files": [
            BASE / "metrics" / "training_history.csv",
            BASE / "metrics" / "validation_metrics.json",
        ],
    },
    "第08部分_基础精度验证实验": {
        "dirs": [
            BASE / "figures" / "training",
            BASE / "inference",
        ],
    },
    "第09部分_热力学一致性验证": {
        "dirs": [
            BASE / "figures" / "thermo_checks",
        ],
        "files": [
            BASE / "figures" / "kvalue_parity.png",
            BASE / "figures" / "rachford_rice_histogram.png",
            BASE / "figures" / "gibbs_energy_histogram.png",
        ],
    },
    "第10部分_相边界连续性验证": {
        "dirs": [
            BASE / "figures" / "phase_boundary",
        ],
        "files": [
            BASE / "metrics" / "phase_boundary_summary.csv",
        ],
    },
    "第11部分_相包络重建": {
        "dirs": [
            BASE / "figures" / "phase_envelope",
        ],
        "files": [
            BASE / "metrics" / "phase_envelope_summary.json",
        ],
    },
    "第12部分_潜空间流形分析": {
        "dirs": [
            BASE / "figures" / "latent_analysis",
        ],
        "files": [
            BASE / "metrics" / "latent_analysis_summary.json",
        ],
    },
    "第13部分_计算速度基准测试": {
        "dirs": [
            BASE / "figures" / "speed_benchmark",
        ],
        "files": [
            BASE / "metrics" / "speed_benchmark_summary.json",
        ],
    },
    "第14部分_油藏模拟器耦合验证": {
        "dirs": [
            BASE / "figures" / "reservoir_simulator",
        ],
        "files": [
            BASE / "metrics" / "simulator",
        ],
    },
    "第15部分_鲁棒性测试": {
        "dirs": [
            BASE / "figures" / "robustness",
        ],
        "files": [
            BASE / "metrics" / "robustness_summary.json",
        ],
    },
    "第16部分_论文撰写策略与创新点提炼": {
        "files": [
            BASE / "第十六章_结果与讨论.docx",
            BASE / "第十六章_结构化科学分析.docx",
        ],
    },
}


def main():
    src_root = BASE

    for part_name, sources in MAPPING.items():
        dst_dir = BASE / part_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  {part_name}")
        print(f"  -> {dst_dir}")

        # Move directories (move contents into target, then remove original)
        for src in sources.get("dirs", []):
            if not src.exists():
                print(f"  SKIP (not found): {src}")
                continue
            dst_sub = dst_dir / src.name
            if src.is_dir():
                if dst_sub.exists():
                    shutil.rmtree(dst_sub)
                shutil.move(str(src), str(dst_sub))
                print(f"  MOVED dir: {src.name}/ -> {dst_sub}")

        # Move files
        for src in sources.get("files", []):
            if not src.exists():
                print(f"  SKIP (not found): {src}")
                continue
            if src.is_dir():
                dst_sub = dst_dir / src.name
                if dst_sub.exists():
                    shutil.rmtree(dst_sub)
                shutil.move(str(src), str(dst_sub))
                print(f"  MOVED dir: {src.name}/ -> {dst_sub}")
            else:
                dst_file = dst_dir / src.name
                if dst_file.exists():
                    dst_file.unlink()
                shutil.move(str(src), str(dst_file))
                print(f"  MOVED file: {src.name} -> {dst_file}")

    # Clean up empty remaining directories
    remaining_dirs = [
        BASE / "figures",
        BASE / "metrics",
    ]
    for d in remaining_dirs:
        if d.exists():
            try:
                shutil.rmtree(d)
                print(f"\n  CLEANED: {d}")
            except Exception as e:
                print(f"\n  WARNING: could not remove {d}: {e}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
