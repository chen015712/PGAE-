import pandas as pd, numpy as np
df = pd.read_csv('c:/Users/Lenovo/Desktop/fluid_15comp_file/sim/pgae_dataset.csv')
s = df[(df['T'] >= 115) & (df['T'] <= 125)]
print(f"N={len(s)} P_corr={s['P'].corr(s['beta_V']):.4f}")
print(f"beta: [{s['beta_V'].min():.3f}, {s['beta_V'].max():.3f}]")
for lo, hi in [(0,5000),(5000,15000),(15000,25000),(25000,35000),(35000,50000)]:
    m = (s['P'] >= lo) & (s['P'] < hi)
    n = m.sum()
    if n > 0:
        print(f"  P[{lo:5d},{hi:5d}): n={n:4d}  beta={s.loc[m,'beta_V'].mean():.4f}")
print(f"L={(s['beta_V']<=1e-6).sum()} V={(s['beta_V']>=1-1e-6).sum()} 2P={((s['beta_V']>1e-6)&(s['beta_V']<1-1e-6)).sum()}")
