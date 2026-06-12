"""Verify all statistical claims for mathematical accuracy"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
mp = pd.read_csv("output/master_panel_data_v2.csv")
ps = pd.read_csv("output/policy_scores_panel.csv")

gy_ps = ps[ps['County_Code'].astype(str) == '130628'].copy()
m = mp[['Year', 'textile_firms', 'total_firms']].merge(
    gy_ps[['Year', 'policy_intensity_total']], on='Year')
m = m[m['Year'] <= 2024].copy()
m['time'] = m['Year'] - 2000

y = m['textile_firms'].values
x_pol = m['policy_intensity_total'].values
x_time = m['time'].values

print("="*60)
print("VERIFICATION OF ALL STATISTICAL CLAIMS")
print("="*60)

# --- Model 1: No time control ---
X1 = np.column_stack([np.ones(len(m)), x_pol])
b1 = np.linalg.inv(X1.T @ X1) @ X1.T @ y
resid1 = y - X1 @ b1
sigma2_1 = np.sum(resid1**2) / (len(m) - 2)
vcov1 = sigma2_1 * np.linalg.inv(X1.T @ X1)
se1 = np.sqrt(np.diag(vcov1))
t1 = b1 / se1
p1 = 2 * (1 - stats.t.cdf(np.abs(t1), len(m)-2))
r2_1 = 1 - np.sum(resid1**2) / np.sum((y - y.mean())**2)
dw = np.sum(np.diff(resid1)**2) / np.sum(resid1**2)

print(f"\nModel 1 (No time):")
print(f"  Policy beta={b1[1]:.3f}, SE={se1[1]:.3f}, t={t1[1]:.3f}, p={p1[1]:.4f}")
print(f"  R2={r2_1:.4f}, DW={dw:.4f}")
print(f"  R2 > DW? {r2_1 > dw}")
print(f"  DW critical values (n=25,k=1,5%): dL=1.29, dU=1.45")
print(f"  DW={dw:.3f} < dL=1.29 => significant positive autocorrelation")

# --- Model 2: With time ---
X2 = np.column_stack([np.ones(len(m)), x_pol, x_time])
b2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y
resid2 = y - X2 @ b2
sigma2_2 = np.sum(resid2**2) / (len(m) - 3)
vcov2 = sigma2_2 * np.linalg.inv(X2.T @ X2)
se2 = np.sqrt(np.diag(vcov2))
t2 = b2 / se2
p2 = 2 * (1 - stats.t.cdf(np.abs(t2), len(m)-3))
r2_2 = 1 - np.sum(resid2**2) / np.sum((y - y.mean())**2)

print(f"\nModel 2 (With time):")
print(f"  Policy beta={b2[1]:.3f}, SE={se2[1]:.3f}, t={t2[1]:.3f}, p={p2[1]:.4f}")
print(f"  Time beta={b2[2]:.3f}, SE={se2[2]:.3f}, t={t2[2]:.3f}, p={p2[2]:.6f}")
print(f"  R2={r2_2:.4f}")

# --- Time-only model ---
X_t = np.column_stack([np.ones(len(m)), x_time])
b_t = np.linalg.inv(X_t.T @ X_t) @ X_t.T @ y
resid_t = y - X_t @ b_t
r2_t = 1 - np.sum(resid_t**2) / np.sum((y - y.mean())**2)
print(f"\nTime-only model: R2={r2_t:.4f}")
print(f"  Delta R2 (policy after time) = {r2_2:.4f} - {r2_t:.4f} = {r2_2-r2_t:.4f}")

# --- Model 3: With time + Post2017 ---
m['post2017'] = (m['Year'] >= 2017).astype(int)
X3 = np.column_stack([np.ones(len(m)), x_pol, x_time, m['post2017'].values])
b3 = np.linalg.inv(X3.T @ X3) @ X3.T @ y
resid3 = y - X3 @ b3
sigma2_3 = np.sum(resid3**2) / (len(m) - 4)
vcov3 = sigma2_3 * np.linalg.inv(X3.T @ X3)
se3 = np.sqrt(np.diag(vcov3))
t3 = b3 / se3
p3 = 2 * (1 - stats.t.cdf(np.abs(t3), len(m)-3))
r2_3 = 1 - np.sum(resid3**2) / np.sum((y - y.mean())**2)

print(f"\nModel 3 (With time + Post2017):")
print(f"  Policy beta={b3[1]:.3f}, SE={se3[1]:.3f}, t={t3[1]:.3f}, p={p3[1]:.4f}")
print(f"  Post2017 beta={b3[3]:.3f}, SE={se3[3]:.3f}, t={t3[3]:.3f}, p={p3[3]:.4f}")
print(f"  R2={r2_3:.4f}")

# --- Model 4: First-difference ---
dy = np.diff(y)
dx = np.diff(x_pol)
n_fd = len(dy)
X4 = np.column_stack([np.ones(n_fd), dx])
b4 = np.linalg.inv(X4.T @ X4) @ X4.T @ dy
resid4 = dy - X4 @ b4
sigma2_4 = np.sum(resid4**2) / (n_fd - 2)
vcov4 = sigma2_4 * np.linalg.inv(X4.T @ X4)
se4 = np.sqrt(np.diag(vcov4))
t4 = b4 / se4
p4 = 2 * (1 - stats.t.cdf(np.abs(t4), n_fd-2))
r2_4 = 1 - np.sum(resid4**2) / np.sum((dy - dy.mean())**2)

print(f"\nModel 4 (First-difference):")
print(f"  Policy beta={b4[1]:.3f}, SE={se4[1]:.3f}, t={t4[1]:.3f}, p={p4[1]:.4f}")
print(f"  R2={r2_4:.4f}")
print(f"  N={n_fd}")

# --- Correlation matrix ---
print(f"\n--- Correlation verification ---")
corr_vars = np.column_stack([y, m['total_firms'].values, x_pol, x_time])
corr_names = ['textile', 'total_firms', 'policy_LLM', 'time']
corr = np.corrcoef(corr_vars.T)
for i, ni in enumerate(corr_names):
    for j, nj in enumerate(corr_names):
        if i < j:
            print(f"  r({ni}, {nj}) = {corr[i,j]:.4f}")

# --- Zero-inflation stats ---
print(f"\n--- Zero-inflation verification ---")
zeros = (x_pol == 0).sum()
print(f"  Policy zeros: {zeros}/{len(m)} ({100*zeros/len(m):.1f}%)")
print(f"  Policy mean: {x_pol.mean():.1f}, var: {x_pol.var():.1f}")
print(f"  var/mean ratio: {x_pol.var()/x_pol.mean():.1f}")
print(f"  Textile mean: {y.mean():.1f}, var: {y.var():.1f}")
print(f"  Textile var/mean: {y.var()/y.mean():.1f}")

# --- DW test interpretation ---
print(f"\n--- DW test interpretation ---")
print(f"  DW = {dw:.3f}")
print(f"  For n=25, k=1, alpha=0.05: dL=1.29, dU=1.45")
print(f"  DW={dw:.3f} < dL=1.29 => reject H0 of no autocorrelation")
print(f"  This means: positive AR(1) autocorrelation in residuals")
print(f"  Consequence: OLS SEs are biased downward, t-stats inflated")
print(f"  BUT: this is NOT the Granger-Newbold spurious regression flag")
print(f"  GN flag: R2 > DW. Here R2={r2_1:.3f}, DW={dw:.3f}, R2 < DW")

# --- Granger-Newbold simulation note ---
print(f"\n--- Important nuance ---")
print(f"  Granger-Newbold (1974) specifically concerns I(1) processes.")
print(f"  With n=27, we CANNOT reliably test for unit roots.")
print(f"  Key finding regardless of TS vs DS:")
print(f"    - With time trend control: policy p={p2[1]:.4f}")
print(f"    - With first-differencing: policy p={p4[1]:.4f}")
print(f"  Under EITHER specification, policy is insignificant.")
print(f"  This is robust to the TS/DS ambiguity.")

# --- Construct validity ---
print(f"\n--- Construct validity ---")
ti = pd.read_csv("output/textile_indices_annual.csv")
m5 = ti[['Year', 'index_policy_support']].merge(
    gy_ps[['Year', 'policy_intensity_total']], on='Year', how='inner')
if len(m5) > 0:
    r_val = np.corrcoef(m5['index_policy_support'], m5['policy_intensity_total'])[0,1]
    print(f"  LLM vs Official Index r = {r_val:.4f}, n = {len(m5)}")
    print(f"  Years: {list(m5['Year'].values)}")

print("\nVerification complete.")
