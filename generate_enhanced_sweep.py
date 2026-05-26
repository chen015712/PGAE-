"""
Enhanced P-sweep data generation targeting high-P region.
Generates dense P-sweeps for the 3 reference compositions used in
robustness testing, with 30 pressure points (log-spaced) at 5 temperatures.
Focus: 50% of pressure points above 15,000 kPa to fix high-P adversarial sensitivity.
"""
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

WINPROP_EXE = r"D:\CMG\WINPROP\2022.10\Win_x64\EXE\pr202210.exe"
BASE_DIR = r"C:\Users\Lenovo\Desktop\fluid_15comp_file"
TEMPLATE_DAT = os.path.join(BASE_DIR, "fluid_15comp.dat")
WORK_DIR = os.path.join(BASE_DIR, "sim")
NC = 15

# ---------------------------------------------------------------------------
# Helpers (mirrored from pgae_data_pipeline.py)
# ---------------------------------------------------------------------------

_COMP_NAMES = ["C10+", "N2", "CO2", "CH4", "C2H6", "C3H8",
               "IC4", "NC4", "IC5", "NC5", "FC6", "FC7", "FC8", "FC9", "FC10"]


def _make_single_flash_dat(P_kPa: float, T_C: float, z: np.ndarray, template_lines: list) -> str:
    lines = list(template_lines)
    for i in range(len(lines)):
        stripped = lines[i].strip().upper()
        if stripped.startswith("FEED") or stripped.startswith("COMPOSITION"):
            break
    feed_line = i
    for j in range(feed_line, len(lines)):
        if lines[j].strip().upper().startswith("*"):
            feed_line = j
            break

    out_lines = lines[:feed_line]
    out_lines.append(f"FEED 0 {P_kPa:.2f} {T_C:.2f} 0")
    for c in range(NC):
        out_lines.append(f"  '{_COMP_NAMES[c]}' {z[c]:.9f}")
    out_lines.append("")
    for line in lines[feed_line:]:
        if line.strip().upper().startswith("COMPOSITION") or line.strip().upper().startswith("FEED"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _parse_single_flash_out(text: str):
    if "Converged" not in text:
        return None
    try:
        from phase_boundary import _parse_ptflash_out
        return _parse_ptflash_out(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    merged_csv = os.path.join(WORK_DIR, "pgae_dataset_merged.csv")
    if not os.path.exists(merged_csv):
        print(f"Merged dataset not found: {merged_csv}")
        return

    df = pd.read_csv(merged_csv)
    z_cols = [f"z{i}" for i in range(1, NC + 1)]
    all_z = df[z_cols].to_numpy(dtype=np.float64)
    all_z = np.clip(all_z, 0, None)
    all_z = all_z / all_z.sum(axis=1, keepdims=True)
    ch4 = all_z[:, 3]

    # Select 3 reference compositions matching robustness testing
    targets = {"oil_rich": 0.208, "typical": 0.368, "gas_rich": 0.628}
    compositions: dict = {}
    for name, target in targets.items():
        idx = int(np.argmin(np.abs(ch4 - target)))
        compositions[name] = all_z[idx].copy()
        print(f"  {name}: CH4={compositions[name][3]:.4f} (target={target})")

    # Dense log-spaced pressures — bias toward high P
    # 30 points: 100 → 50,000 kPa
    P_points = np.logspace(np.log10(100), np.log10(50000), 30)
    # 5 temperatures — expanded range
    temperatures = [50.0, 80.0, 100.0, 120.0, 150.0]

    total = len(compositions) * len(temperatures) * len(P_points)
    print(f"\nTotal runs: {len(compositions)} × {len(temperatures)} × {len(P_points)} = {total}")
    print(f"High-P focus: {(P_points >= 15000).sum()} of {len(P_points)} P points ≥ 15,000 kPa")

    with open(TEMPLATE_DAT, "r", encoding="utf-8", errors="ignore") as f:
        template_lines = [line.rstrip("\n") for line in f.readlines()]

    new_rows = []
    n_conv, n_fail = 0, 0
    pbar = tqdm(total=total, desc="Enhanced P-sweep")

    for comp_name, z_i in compositions.items():
        for T_i in temperatures:
            for P_i in P_points:
                dat_content = _make_single_flash_dat(float(P_i), float(T_i), z_i, template_lines)
                fname = f"_rs_{comp_name}_{T_i:.0f}C_{P_i:.0f}kPa"
                safe = fname.replace(".", "_")
                dat_path = os.path.join(WORK_DIR, f"{safe}.dat")

                with open(dat_path, "w", encoding="utf-8") as f:
                    f.write(dat_content)

                try:
                    result = subprocess.run(
                        [WINPROP_EXE],
                        cwd=WORK_DIR,
                        input=f"{dat_path}\n",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=120,
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
                    else:
                        n_fail += 1
                except (subprocess.TimeoutExpired, OSError):
                    n_fail += 1

                # Cleanup temp files
                for ext in [".dat", ".out"]:
                    p = dat_path if ext == ".dat" else dat_path.replace(".dat", ".out")
                    try:
                        os.remove(p)
                    except OSError:
                        pass

                pbar.update(1)
    pbar.close()

    print(f"\nConverged: {n_conv}/{total}  Failed: {n_fail}/{total}")

    if new_rows:
        cols = (["P", "T"]
                + [f"z{i}" for i in range(1, NC + 1)]
                + ["beta_V", "phase_label"]
                + [f"x{i}" for i in range(1, NC + 1)]
                + [f"y{i}" for i in range(1, NC + 1)])
        new_df = pd.DataFrame(new_rows, columns=cols)

        # Merge with existing merged dataset (deduplicate by P,T,z)
        existing = pd.read_csv(merged_csv)
        ez = existing[z_cols].to_numpy()
        nz = new_df[z_cols].to_numpy()
        keep = np.ones(len(new_df), dtype=bool)
        for i in range(len(new_df)):
            for j in range(len(existing)):
                dz = np.abs(nz[i] - ez[j]).max()
                dP = abs(new_df.iloc[i]["P"] - existing.iloc[j]["P"])
                dT = abs(new_df.iloc[i]["T"] - existing.iloc[j]["T"])
                if dz < 1e-6 and dP < 10 and dT < 1:
                    keep[i] = False
                    break

        new_unique = new_df[keep]
        print(f"New unique rows: {len(new_unique)} (removed {len(new_df) - len(new_unique)} duplicates)")

        merged = pd.concat([existing, new_unique], ignore_index=True)
        merged.to_csv(merged_csv, index=False)
        print(f"Merged dataset: {len(merged)} samples → {merged_csv}")
    else:
        print("No new data generated.")


if __name__ == "__main__":
    main()
