import pandas as pd
import numpy as np

df = pd.read_csv('c:/Users/Lenovo/Desktop/fluid_15comp_file/sim/pgae_dataset.csv')

# Check dataset overall P-beta relationship
print("=== Overall P-beta relationship ===")
print(f"corr(P, beta) = {df['P'].corr(df['beta_V']):.4f}")
print(f"corr(T, beta) = {df['T'].corr(df['beta_V']):.4f}")

# Check T≈120°C
mask_t = (df['T'] > 115) & (df['T'] < 125)
s = df[mask_t]
print(f"\n=== T≈120°C: {len(s)} samples ===")
print(f"P range: [{s['P'].min():.0f}, {s['P'].max():.0f}] kPa")
print(f"beta range: [{s['beta_V'].min():.4f}, {s['beta_V'].max():.4f}]")
print(f"P-beta corr at T≈120: {s['P'].corr(s['beta_V']):.4f}")

# Bin by P
bins = [0, 5000, 10000, 20000, 30000, 40000, 50000]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (s['P'] >= lo) & (s['P'] < hi)
    n = m.sum()
    if n > 0:
        print(f"  P [{lo:5d},{hi:5d}) kPa: n={n:4d}, mean_beta={s.loc[m,'beta_V'].mean():.4f}")

# Phase distribution
print(f"\nPhase distribution at T≈120:")
print(f"  beta<=1e-6 (liquid): {(s['beta_V']<=1e-6).sum()}")
print(f"  beta>=1-1e-6 (vapor): {(s['beta_V']>=1-1e-6).sum()}")
print(f"  0<beta<1 (two-phase): {((s['beta_V']>1e-6)&(s['beta_V']<1-1e-6)).sum()}")

# Also check: for a specific composition, how does beta vary with P?
print("\n=== Check specific composition ===")
z_cols = [f'z{i}' for i in range(1, 16)]
all_z = df[z_cols].to_numpy()
# Find composition closest to median CH4
ch4 = all_z[:, 3]
med = np.median(ch4)
med_idx = np.argmin(np.abs(ch4 - med))
z_med = all_z[med_idx]
print(f"Median CH4 index: {med_idx}, CH4={z_med[3]:.4f}")

# Find all rows with similar composition
z_dist = np.abs(all_z - z_med).sum(axis=1)
close_idx = np.argsort(z_dist)[:100]
close_set = df.iloc[close_idx]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (close_set['P'] >= lo) & (close_set['P'] < hi)
    n = m.sum()
    if n > 0:
        print(f"  P [{lo:5d},{hi:5d}) kPa: n={n:4d}, mean_beta={close_set.loc[m,'beta_V'].mean():.4f}, mean_T={close_set.loc[m,'T'].mean():.0f}")
