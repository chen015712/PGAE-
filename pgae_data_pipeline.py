import os
import re
import subprocess
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# =================================================================
# 1. 终极配置区（严格绝对路径，杜绝相对路径迷失）
# =================================================================
WINPROP_EXE = r"D:\CMG\WINPROP\2022.10\Win_x64\EXE\pr202210.exe"
BASE_DIR = r"C:\Users\Lenovo\Desktop\fluid_15comp_file"

CONFIG = {
    "template_dat": os.path.join(BASE_DIR, "fluid_15comp.dat"),
    "template_out": os.path.join(BASE_DIR, "fluid_15comp.out"),
    "nc": 15,
    "sample_count": 5000,

    # -----------------------------
    # 压力温度范围
    # -----------------------------
    "p_range": (5000.0, 45000.0),
    "t_range": (30.0, 180.0),

    "work_dir": os.path.join(BASE_DIR, "sim"),
    "output_csv": os.path.join(BASE_DIR, "sim", "pgae_dataset.csv"),
    "physics_pth": os.path.join(BASE_DIR, "sim", "physics_constants.pth")
}

if not os.path.exists(CONFIG["work_dir"]):
    os.makedirs(CONFIG["work_dir"], exist_ok=True)

# =================================================================
# 2. 物理算子提取器
# =================================================================
def extract_physics_ops():

    print(f"[*] 正在从原始 .out 文件提取 EOS 物理算子...")

    if not os.path.exists(CONFIG["template_out"]):
        print(f"⚠️ 找不到 {CONFIG['template_out']}，跳过提取。")
        return

    with open(CONFIG["template_out"], 'r', encoding='gbk', errors='ignore') as f:
        content = f.read()

    def get_val(key):

        match = re.findall(
            rf'\*{key}\s+([\d\.\s\-Ee]+)',
            content,
            re.S
        )

        return np.array(match[0].split(), dtype=float) if match else None

    try:

        pc = get_val('PCRIT')
        tc = get_val('TCRIT')
        omega = get_val('AC')

        bip = np.zeros((CONFIG['nc'], CONFIG['nc']))

        bin_block = re.findall(
            r'\*BIN\s+(.*?)\*PVC3',
            content,
            re.S
        )

        if bin_block:

            lines = bin_block[0].strip().split('\n')

            for i, line in enumerate(lines):

                vals = list(map(float, line.split()))

                for j, val in enumerate(vals):

                    bip[i + 1, j] = val
                    bip[j, i + 1] = val

        ops = {
            'Pc': torch.tensor(pc, dtype=torch.float32),
            'Tc': torch.tensor(tc, dtype=torch.float32),
            'omega': torch.tensor(omega, dtype=torch.float32),
            'BIP': torch.tensor(bip, dtype=torch.float32),
            'R': 8.314472
        }

        torch.save(ops, CONFIG["physics_pth"])

        print(f"✅ 物理算子张量已保存至: {CONFIG['physics_pth']}")

    except Exception as e:

        print(f"⚠️ 物理参数提取异常: {e}")

# =================================================================
# 3. Phase-aware 物理约束采样器
# =================================================================

def generate_physical_z():
    """
    生成具有物理意义的15组分组成
    """

    z = np.zeros(CONFIG["nc"])

    # ==========================================================
    # 轻组分
    # ==========================================================

    z[1] = np.random.uniform(0.01, 0.08)   # N2
    z[2] = np.random.uniform(0.05, 0.35)   # CO2
    z[3] = np.random.uniform(0.25, 0.60)   # CH4

    # ==========================================================
    # 中间组分
    # ==========================================================

    z[4] = np.random.uniform(0.03, 0.12)   # C2
    z[5] = np.random.uniform(0.02, 0.10)   # C3

    z[6] = np.random.uniform(0.01, 0.06)   # IC4
    z[7] = np.random.uniform(0.01, 0.06)   # NC4

    z[8] = np.random.uniform(0.005, 0.05)  # IC5
    z[9] = np.random.uniform(0.005, 0.05)  # NC5

    # ==========================================================
    # 重组分
    # ==========================================================

    heavy_total = np.random.uniform(0.05, 0.25)

    fc = np.random.dirichlet(np.ones(5)) * heavy_total

    z[10:] = fc

    # ==========================================================
    # C10+
    # ==========================================================

    z[0] = np.random.uniform(0.02, 0.12)

    # ==========================================================
    # 归一化
    # ==========================================================

    z = z / z.sum()

    return z

# =================================================================
# 4. 相图感知 PT 采样器
# =================================================================

def generate_phase_aware_PT():

    """
    基于 phase envelope 的采样器
    """

    # ==========================================================
    # 温度优先采样在相变敏感区
    # ==========================================================

    T = np.random.uniform(40.0, 180.0)

    # ==========================================================
    # 根据你 phase envelope 的经验拟合
    # ==========================================================

    P_center = (
        22500
        - 0.55 * (T - 120.0) ** 2
    )

    rand = np.random.rand()

    # ==========================================================
    # 70%：相边界附近
    # ==========================================================

    if rand < 0.7:

        P = P_center + np.random.normal(0, 1800)

    # ==========================================================
    # 20%：两相区内部
    # ==========================================================

    elif rand < 0.9:

        P = P_center + np.random.uniform(-2500, 2500)

    # ==========================================================
    # 10%：单相区
    # ==========================================================

    else:

        P = np.random.uniform(*CONFIG["p_range"])

    P = np.clip(
        P,
        CONFIG["p_range"][0],
        CONFIG["p_range"][1]
    )

    return P, T

# =================================================================
# 5. 逐行状态机重构器
# =================================================================

def create_safe_dat_content(template_text, p, t, z_array):

    lines = template_text.split('\n')

    new_lines = []

    i = 0

    while i < len(lines):

        line = lines[i]

        upper_line = line.strip().upper()

        # ======================================================
        # 删除 plot 指令
        # ======================================================

        if upper_line.startswith('*PLOT'):

            i += 1
            continue

        # ======================================================
        # 替换压力
        # ======================================================

        if upper_line.startswith('*PRES '):

            new_lines.append(f"*PRES {p:.2f}")

            i += 1
            continue

        # ======================================================
        # 替换温度
        # ======================================================

        if upper_line.startswith('*TEMP '):

            new_lines.append(f"*TEMP {t:.2f}")

            i += 1
            continue

        # ======================================================
        # Flash 搜索参数
        # ======================================================

        if upper_line.startswith('*DELP '):

            new_lines.append("*DELP 500.0")

            i += 1
            continue

        if upper_line.startswith('*STEPP '):

            new_lines.append("*STEPP 20")

            i += 1
            continue

        # ======================================================
        # 组分替换
        # ======================================================

        if '*PRIMARY' in upper_line:

            new_lines.append(line)

            for j in range(0, 15, 5):

                z_row = "   ".join(
                    [f"{v:.6f}" for v in z_array[j:j+5]]
                )

                new_lines.append(z_row)

            i += 1

            while i < len(lines):

                next_line_upper = lines[i].strip().upper()

                if next_line_upper.startswith('*') or next_line_upper.startswith('**'):
                    break

                i += 1

            continue

        new_lines.append(line)

        i += 1

    return "\n".join(new_lines)

# =================================================================
# 6. WinProp 输出解析器
# =================================================================

def parse_winprop_result(out_file):

    if not os.path.exists(out_file):
        return None

    with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:

        text = f.read()

    try:

        # ======================================================
        # 两相
        # ======================================================

        if "Phase Split: Liquid-Vapour" in text:

            beta = float(
                re.findall(
                    r'Phase Mole %\s+[\d\.]+\s+([\d\.]+)',
                    text
                )[0]
            ) / 100.0

            table_match = re.search(
                r'component\s+Feed\s+Phase01\s+Phase02(.*?)\.\.\.\.\.\.',
                text,
                re.S
            )

            if not table_match:
                return None

            lines = [
                line
                for line in table_match.group(1).strip().split('\n')
                if line.strip()
            ][:CONFIG['nc']]

            x = []
            y = []

            for line in lines:

                parts = line.split()

                x.append(float(parts[2]) / 100.0)
                y.append(float(parts[3]) / 100.0)

            phase_label = 2

            return {
                'beta': beta,
                'x': x,
                'y': y,
                'phase_label': phase_label
            }

        # ======================================================
        # 单气相
        # ======================================================

        elif "Phase Split: Vapour" in text:

            beta = 1.0

            table_match = re.search(
                r'component\s+Feed\s+Phase01(.*?)\.\.\.\.\.\.',
                text,
                re.S
            )

            if not table_match:
                return None

            lines = [
                line
                for line in table_match.group(1).strip().split('\n')
                if line.strip()
            ][:CONFIG['nc']]

            x = []
            y = []

            for line in lines:

                parts = line.split()

                val = float(parts[2]) / 100.0

                x.append(val)
                y.append(val)

            phase_label = 1

            return {
                'beta': beta,
                'x': x,
                'y': y,
                'phase_label': phase_label
            }

        # ======================================================
        # 单液相
        # ======================================================

        elif "Phase Split: Liquid" in text:

            beta = 0.0

            table_match = re.search(
                r'component\s+Feed\s+Phase01(.*?)\.\.\.\.\.\.',
                text,
                re.S
            )

            if not table_match:
                return None

            lines = [
                line
                for line in table_match.group(1).strip().split('\n')
                if line.strip()
            ][:CONFIG['nc']]

            x = []
            y = []

            for line in lines:

                parts = line.split()

                val = float(parts[2]) / 100.0

                x.append(val)
                y.append(val)

            phase_label = 0

            return {
                'beta': beta,
                'x': x,
                'y': y,
                'phase_label': phase_label
            }

        else:

            return None

    except Exception:

        return None

# =================================================================
# 7. Fixed-composition P-sweep data augmentation (fixes β-P inversion)
# =================================================================

def _make_single_flash_dat(P: float, T: float, z: np.ndarray, template_lines: list) -> str:
    """Generate a WinProp DAT file with a single fixed-PT flash at given (P, T, z).

    Uses *TYPE *QNSS with *DELP 0.0 *STEPP 1 for a single fixed-PT flash
    (no pressure/temperature sweep).
    """
    lines_out = []
    in_envelope = False
    in_flash = False
    composition_injected = False
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
        if upper.startswith("**=-=-="):
            in_envelope = False
            in_flash = False
            lines_out.append(line)
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

        if "*PRIMARY" in upper and not composition_injected:
            lines_out.append(line)
            for j in range(0, CONFIG["nc"], 5):
                lines_out.append("   ".join(f"{v:.6f}" for v in z[j:j+5]))
            composition_injected = True
            i += 1
            while i < len(template_lines) and not template_lines[i].strip().startswith("*"):
                i += 1
            continue

        lines_out.append(line)
        i += 1

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


def _parse_single_flash_out(out_text: str):
    """Parse a single fixed-PT flash result from WinProp .out file.

    Returns dict with beta, x, y, phase_label or None on failure.
    phase_label: 0=Liquid, 1=Vapour, 2=Two-Phase (matching existing pipeline).
    """
    beta = None
    is_two_phase = "Phase Split: Liquid-Vapour" in out_text
    is_vapour = "Phase Split: Vapour" in out_text and not is_two_phase
    is_liquid = "Phase Split: Liquid" in out_text and not is_two_phase

    if is_two_phase:
        m = re.search(r'Phase\s+Mole\s*%\s+([\d.]+)\s+([\d.]+)', out_text)
        if m:
            beta = float(m.group(2)) / 100.0
        phase_label = 2
    elif is_vapour:
        beta = 1.0
        phase_label = 1
    elif is_liquid:
        beta = 0.0
        phase_label = 0
    else:
        return None

    # Parse composition table
    split_marker = out_text.find("Phase Split:")
    mole_marker = out_text.find("Phase Mole %")
    if split_marker >= 0:
        table_text = out_text[split_marker:mole_marker] if mole_marker > split_marker else out_text[split_marker:]
    else:
        table_text = out_text[:mole_marker] if mole_marker > 0 else out_text

    liq_vals = []
    vap_vals = []
    in_table = False
    for line in table_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if "component" in lower or "feed" in lower:
            in_table = True
            continue
        if not in_table:
            continue
        if "Phase" in stripped or lower.startswith("---"):
            in_table = False
            break
        if "mass percent" in lower or "ln (fug" in lower:
            in_table = False
            break
        nums = []
        for t in stripped.split():
            try:
                nums.append(float(t))
            except ValueError:
                pass
        if is_two_phase:
            if len(nums) >= 3:
                liq_vals.append(nums[1])
                vap_vals.append(nums[2])
        else:
            if len(nums) >= 2:
                liq_vals.append(nums[1])
                vap_vals.append(nums[1])

    if len(liq_vals) < CONFIG["nc"]:
        return None

    x = np.array(liq_vals[:CONFIG["nc"]], dtype=np.float64) / 100.0
    y = np.array(vap_vals[:CONFIG["nc"]], dtype=np.float64) / 100.0
    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)
    x_sum = max(x.sum(), 1e-12)
    y_sum = max(y.sum(), 1e-12)
    x = x / x_sum
    y = y / y_sum

    return {"beta": beta, "x": x, "y": y, "phase_label": phase_label}


def generate_p_sweep_data():
    """Generate fixed-composition P-sweep data to fix the P-β inversion.

    Selects 5 representative compositions from the existing dataset and runs
    WinProp on a dense P grid (20 points) at 3 temperatures each, producing
    ~300 new samples with correct ∂β/∂P < 0 (β decreases as P increases).
    """
    import pandas as pd

    output_csv = CONFIG["output_csv"]
    aug_csv = os.path.join(CONFIG["work_dir"], "pgae_dataset_augmented.csv")

    if not os.path.exists(output_csv):
        print(f"⚠️  Existing dataset not found at {output_csv}, skipping augmentation.")
        return

    df = pd.read_csv(output_csv)
    z_cols = [f"z{i}" for i in range(1, CONFIG["nc"] + 1)]
    all_z = df[z_cols].to_numpy(dtype=np.float64)
    ch4 = all_z[:, 3]

    # Select 5 compositions spanning the CH4 range
    percentiles = [5, 25, 50, 75, 95]
    selected_indices = []
    compositions = []
    comp_labels = []
    for pct in percentiles:
        target = np.percentile(ch4, pct)
        idx = np.argmin(np.abs(ch4 - target))
        # Avoid duplicates
        while idx in selected_indices:
            idx = (idx + 1) % len(ch4)
        selected_indices.append(idx)
        z_i = all_z[idx].copy()
        compositions.append(z_i)
        comp_labels.append(f"CH4_{pct}pct={z_i[3]:.3f}")

    print("\n" + "=" * 60)
    print("=== Data Augmentation: Fixed-Composition P-Sweeps ===")
    print("=" * 60)
    print(f"Selected {len(compositions)} compositions:")
    for i, label in enumerate(comp_labels):
        print(f"  [{i}] {label}  C10+={compositions[i][0]:.4f}  CO2={compositions[i][2]:.4f}")

    temperatures = [60.0, 100.0, 150.0]
    P_points = np.linspace(100.0, 50000.0, 20)

    total_runs = len(compositions) * len(temperatures) * len(P_points)
    print(f"\nTotal runs: {len(compositions)} comps × {len(temperatures)} T × {len(P_points)} P = {total_runs}")

    if not os.path.exists(WINPROP_EXE):
        print(f"❌ WinProp not found: {WINPROP_EXE}")
        return

    with open(CONFIG["template_dat"], "r", encoding="utf-8", errors="ignore") as f:
        template_lines = [line.rstrip("\n") for line in f.readlines()]

    new_rows = []
    n_conv = 0
    n_fail = 0

    print("\nRunning WinProp P-sweeps...")
    pbar = tqdm(total=total_runs, desc="P-sweep Progress")

    for z_i, z_label in zip(compositions, comp_labels):
        for T_i in temperatures:
            for P_i in P_points:
                dat_content = _make_single_flash_dat(float(P_i), float(T_i), z_i, template_lines)
                dat_path = os.path.join(CONFIG["work_dir"], f"_aug_{n_conv:04d}.dat")
                with open(dat_path, "w", encoding="utf-8") as f:
                    f.write(dat_content)

                try:
                    result = subprocess.run(
                        [WINPROP_EXE],
                        cwd=CONFIG["work_dir"],
                        input=f"{dat_path}\n",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=120.0,
                    )
                    out_path = dat_path.replace(".dat", ".out")
                    if os.path.exists(out_path):
                        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                            out_text = f.read()
                        parsed = _parse_single_flash_out(out_text)
                        if parsed is not None:
                            row = (
                                [float(P_i), float(T_i)]
                                + list(z_i)
                                + [parsed["beta"]]
                                + [parsed["phase_label"]]
                                + list(parsed["x"])
                                + list(parsed["y"])
                            )
                            new_rows.append(row)
                            n_conv += 1
                        else:
                            n_fail += 1
                except (subprocess.TimeoutExpired, OSError):
                    n_fail += 1

                pbar.update(1)

    pbar.close()

    print(f"\n  Converged: {n_conv}/{total_runs}  Failed: {n_fail}/{total_runs}")

    if not new_rows:
        print("❌ No augmented data generated.")
        return

    cols = (
        ["P", "T"]
        + [f"z{j}" for j in range(1, CONFIG["nc"] + 1)]
        + ["beta_V"]
        + ["phase_label"]
        + [f"x{j}" for j in range(1, CONFIG["nc"] + 1)]
        + [f"y{j}" for j in range(1, CONFIG["nc"] + 1)]
    )
    df_aug = pd.DataFrame(new_rows, columns=cols)
    df_aug.to_csv(aug_csv, index=False)
    print(f"✅ Augmented data saved: {aug_csv}  ({len(df_aug)} samples)")

    # Merge with existing dataset
    df_merged = pd.concat([df, df_aug], ignore_index=True)
    merged_csv = os.path.join(CONFIG["work_dir"], "pgae_dataset_merged.csv")
    df_merged.to_csv(merged_csv, index=False)
    print(f"✅ Merged dataset saved: {merged_csv}  ({len(df_merged)} samples)")

    # Verify P-β correlation at fixed T
    print("\n=== P-β correlation in augmented data (fixed T windows) ===")
    for T_center in [60, 100, 150]:
        mask = (df_aug["T"] >= T_center - 5) & (df_aug["T"] <= T_center + 5)
        s = df_aug[mask]
        if len(s) > 10:
            corr = s["P"].corr(s["beta_V"])
            status = "✅ CORRECT (P↑→β↓)" if corr < 0 else "❌ WRONG (P↑→β↑)"
            print(f"  T~{T_center:3.0f}°C: N={len(s):3d}  corr(P,β)={corr:+.4f}  {status}")


# =================================================================
# 8. 主流水线
# =================================================================

def main():

    print("================================================")
    print("=== PGAE 数据集全自动生产流水线（Phase-aware）===")
    print("================================================")

    extract_physics_ops()

    if not os.path.exists(CONFIG["template_dat"]):

        print(f"❌ 未找到模板 DAT 文件: {CONFIG['template_dat']}")
        return

    with open(CONFIG["template_dat"], 'r', encoding='utf-8', errors='ignore') as f:

        template_str = f.read()

    # ==========================================================
    # 生成样本
    # ==========================================================

    z_samples = []
    p_samples = []
    t_samples = []

    print("\n[*] 正在生成 phase-aware 样本点...")

    for _ in range(CONFIG["sample_count"]):

        z = generate_physical_z()

        P, T = generate_phase_aware_PT()

        z_samples.append(z)
        p_samples.append(P)
        t_samples.append(T)

    z_samples = np.array(z_samples)
    p_samples = np.array(p_samples)
    t_samples = np.array(t_samples)

    dataset = []

    if not os.path.exists(WINPROP_EXE):

        print(f"❌ 找不到 WinProp: {WINPROP_EXE}")
        return

    print(f"\n🚀 开始执行 WinProp 仿真...")
    print(f"📁 工作目录: {CONFIG['work_dir']}")

    # ==========================================================
    # 批量计算
    # ==========================================================

    for i in tqdm(range(CONFIG["sample_count"]), desc="Simulation Progress"):

        sim_dat = create_safe_dat_content(
            template_str,
            p_samples[i],
            t_samples[i],
            z_samples[i]
        )

        dat_path = os.path.abspath(
            os.path.join(CONFIG["work_dir"], f"case_{i}.dat")
        )

        out_path = dat_path.replace(".dat", ".out")

        with open(dat_path, 'w', encoding='utf-8') as f:

            f.write(sim_dat)

        try:

            result = subprocess.run(
                [WINPROP_EXE],
                cwd=CONFIG["work_dir"],
                input=f"{dat_path}\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='ignore',
                timeout=120.0
            )

            if (
                "error" in result.stderr.lower()
                or
                "error" in result.stdout.lower()
            ):

                for line in result.stdout.splitlines():

                    if 'error' in line.lower():

                        print(f"\n[!] 算例 {i}: {line.strip()[:200]}")
                        break

        except subprocess.TimeoutExpired:

            print(f"\n[!] 算例 {i} 超时。")
            continue

        except FileNotFoundError:

            print(f"\n❌ 找不到 WinProp EXE")
            return

        res = parse_winprop_result(out_path)

        if res:

            row = (
                [p_samples[i], t_samples[i]]
                +
                list(z_samples[i])
                +
                [res['beta']]
                +
                [res['phase_label']]
                +
                res['x']
                +
                res['y']
            )

            dataset.append(row)

        else:

            print(f"\n[!] 算例 {i} 解析失败。")

    # ==========================================================
    # 保存数据
    # ==========================================================

    if dataset:

        cols = (
            ['P', 'T']
            +
            [f'z{j}' for j in range(1, 16)]
            +
            ['beta_V']
            +
            ['phase_label']
            +
            [f'x{j}' for j in range(1, 16)]
            +
            [f'y{j}' for j in range(1, 16)]
        )

        df = pd.DataFrame(dataset, columns=cols)

        try:

            df.to_csv(CONFIG["output_csv"], index=False)

            print("\n🎉 数据集生成成功！")
            print(f"📄 数据条数: {len(df)}")
            print(f"📁 保存路径: {CONFIG['output_csv']}")

            # ==================================================
            # 统计信息
            # ==================================================

            two_phase_ratio = np.mean(
                (df['beta_V'] > 0.0) &
                (df['beta_V'] < 1.0)
            )

            print("\n==============================")
            print("数据统计")
            print("==============================")

            print(f"两相数据占比: {two_phase_ratio*100:.2f}%")

            print(f"单液相占比: {np.mean(df['beta_V']==0)*100:.2f}%")

            print(f"单气相占比: {np.mean(df['beta_V']==1)*100:.2f}%")

        except PermissionError:

            import time

            fallback = CONFIG["output_csv"].replace(
                ".csv",
                f"_{int(time.time())}.csv"
            )

            df.to_csv(fallback, index=False)

            print(f"\n⚠️ 原 CSV 被占用。")
            print(f"已另存为: {fallback}")

    else:

        print("\n❌ 未生成有效数据。")

# =================================================================
# 8. 程序入口
# =================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--augment":
        generate_p_sweep_data()
    else:
        main()