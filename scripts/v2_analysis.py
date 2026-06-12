"""V2 comprehensive analysis pipeline.
Uses policy_scores_panel_v2.csv (all 9 bugs fixed).
Compares v1 vs v2 results systematically.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis"

# Load v2 data
ps2 = pd.read_csv("output/policy_scores_panel_v2.csv")
mp = pd.read_csv("output/master_panel_data_v2.csv")

gy2 = ps2[ps2['County_Code'].astype(str) == '130628'].copy()
# Keep only core columns from mp to avoid merge conflicts
mp_cols = ['County_Code', 'Year', 'textile_firms', 'total_firms', 'textile_ratio', 'textile_firms_ma3']
m2 = mp[mp['County_Code'].astype(str) == '130628'][mp_cols].copy()
m2 = m2[m2['Year'] <= 2024].copy()

# Merge v2 policy scores
m2 = m2.merge(
    gy2[['Year', 'policy_intensity_total', 'policy_intensity_concentration',
         'equipment_sum', 'environment_sum', 'ecommerce_sum', 'brandquality_sum',
         'cluster_sum', 'finance_sum', 'education_sum',
         'equipment_index', 'environment_index', 'ecommerce_index',
         'brandquality_index', 'cluster_index', 'finance_index', 'education_index',
         'total_chunks', 'scored', 'keyword_passed']],
    on='Year', how='left'
)
m2['time'] = m2['Year'] - 2000
m2['post2017'] = (m2['Year'] >= 2017).astype(int)
m2['ln_textile'] = np.log(m2['textile_firms'])

# Also load v1 for comparison
ps1 = pd.read_csv("output/policy_scores_panel.csv")
gy1 = ps1[ps1['County_Code'].astype(str) == '130628'].copy()
m1 = mp[mp['County_Code'].astype(str) == '130628'][mp_cols].copy()
m1 = m1[m1['Year'] <= 2024].copy()
m1 = m1.merge(gy1[['Year', 'policy_intensity_total', 'total_chunks']], on='Year', how='left')
m1['time'] = m1['Year'] - 2000

print("=" * 70)
print("V2 ANALYSIS — Gaoyang County (130628), 2000-2024")
print("=" * 70)

# ============================================================
# 1. Data quality comparison
# ============================================================
print("\n--- 1. Data Quality: V1 vs V2 ---")
print(f"\n  {'':<30} {'V1 (Buggy)':>15} {'V2 (Fixed)':>15}")
print(f"  {'Non-zero years':<30} {(m1['policy_intensity_total'] > 0).sum():>15} {(m2['policy_intensity_total'] > 0).sum():>15}")
print(f"  {'Zero rate':<30} {(m1['policy_intensity_total'] == 0).sum()/25*100:>14.1f}% {(m2['policy_intensity_total'] == 0).sum()/25*100:>14.1f}%")
print(f"  {'Mean policy_total':<30} {m1['policy_intensity_total'].mean():>15.1f} {m2['policy_intensity_total'].mean():>15.1f}")
print(f"  {'Var policy_total':<30} {m1['policy_intensity_total'].var():>15.1f} {m2['policy_intensity_total'].var():>15.1f}")
print(f"  {'Var/Mean':<30} {m1['policy_intensity_total'].var()/max(m1['policy_intensity_total'].mean(),0.1):>15.1f} {m2['policy_intensity_total'].var()/max(m2['policy_intensity_total'].mean(),0.1):>15.1f}")

# ============================================================
# 2. Year-by-year comparison
# ============================================================
print(f"\n--- 2. Year-by-Year: V1 vs V2 policy_intensity_total ---")
print(f"  {'Year':<8} {'V1':>10} {'V2_abs':>10} {'V2_conc':>10} {'Chunks':>8}")
for _, row in m2.iterrows():
    yr = int(row['Year'])
    v1_val = m1[m1['Year'] == yr]['policy_intensity_total'].values
    v1_str = f"{v1_val[0]:.0f}" if len(v1_val) > 0 else "N/A"
    print(f"  {yr:<8} {v1_str:>10} {row['policy_intensity_total']:>10.1f} {row['policy_intensity_concentration']:>10.1f} {row['total_chunks']:>8.0f}")

# ============================================================
# 3. Correlations with textile firms
# ============================================================
print(f"\n--- 3. Variable Rankings ---")
vars_to_check = {
    'time': m2['time'],
    'total_chunks': m2['total_chunks'],
    'policy_total_v2': m2['policy_intensity_total'],
    'policy_conc_v2': m2['policy_intensity_concentration'],
    'scored_chunks': m2['scored'],
}
for name, var in vars_to_check.items():
    r = np.corrcoef(m2['textile_firms'].values, var.values)[0, 1]
    print(f"  r(textile, {name:<25s}) = {r:.4f}")

# ============================================================
# 4. Main regressions — 5 specifications
# ============================================================
print(f"\n--- 4. OLS Regressions (V2 policy_intensity_total) ---")
y = m2['textile_firms'].values
x_pol = m2['policy_intensity_total'].values
x_time = m2['time'].values
x_chunks = m2['total_chunks'].values
x_post = m2['post2017'].values

specs = [
    ("No time", np.column_stack([np.ones(25), x_pol])),
    ("+Time", np.column_stack([np.ones(25), x_pol, x_time])),
    ("+Time+Post", np.column_stack([np.ones(25), x_pol, x_time, x_post])),
    ("+Time+Chunks", np.column_stack([np.ones(25), x_pol, x_time, x_chunks])),
]

print(f"  {'Spec':<20} {'Policy_beta':>12} {'Policy_SE':>10} {'Policy_p':>10} {'R2':>8}")
for name, X in specs:
    k = X.shape[1]
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    r = y - X @ b
    se = np.sqrt(np.sum(r**2) / (25 - k) * np.diag(np.linalg.inv(X.T @ X)))
    p = 2 * (1 - stats.t.cdf(np.abs(b[1] / se[1]), 25 - k))
    r2 = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    print(f"  {name:<20} {b[1]:>12.3f} {se[1]:>10.3f} {p:>10.4f} {r2:>8.4f}")

# First-difference
dy = np.diff(y)
dx = np.diff(x_pol)
X_fd = np.column_stack([np.ones(24), dx])
b_fd = np.linalg.inv(X_fd.T @ X_fd) @ X_fd.T @ dy
r_fd = dy - X_fd @ b_fd
se_fd = np.sqrt(np.sum(r_fd**2) / 22 * np.diag(np.linalg.inv(X_fd.T @ X_fd)))
p_fd = 2 * (1 - stats.t.cdf(np.abs(b_fd[1] / se_fd[1]), 22))
r2_fd = 1 - np.sum(r_fd**2) / np.sum((dy - dy.mean())**2)
print(f"  {'1st-Diff':<20} {b_fd[1]:>12.3f} {se_fd[1]:>10.3f} {p_fd:>10.4f} {r2_fd:>8.4f}")

# ============================================================
# 5. Construct validity
# ============================================================
print(f"\n--- 5. Construct Validity (LLM vs Official Index) ---")
ti = pd.read_csv("output/textile_indices_annual.csv")
m5 = ti[['Year', 'index_policy_support']].merge(
    gy2[['Year', 'policy_intensity_total', 'policy_intensity_concentration']],
    on='Year', how='inner')
if len(m5) > 0:
    r_val_abs = np.corrcoef(m5['index_policy_support'], m5['policy_intensity_total'])[0, 1]
    r_val_conc = np.corrcoef(m5['index_policy_support'], m5['policy_intensity_concentration'])[0, 1]
    print(f"  r(official, LLM_absolute) = {r_val_abs:.4f}, n = {len(m5)}")
    print(f"  r(official, LLM_concentration) = {r_val_conc:.4f}, n = {len(m5)}")
    print(f"  Years: {list(m5['Year'].values)}")

# ============================================================
# 6. Dimension-level analysis
# ============================================================
print(f"\n--- 6. Dimension-Level Regressions (with time control) ---")
dims = [
    ('equipment_sum', 'Equipment'),
    ('environment_sum', 'Environment'),
    ('ecommerce_sum', 'Ecommerce'),
    ('brandquality_sum', 'BrandQuality'),
    ('cluster_sum', 'Cluster'),
    ('finance_sum', 'Finance'),
    ('education_sum', 'Education'),
]
print(f"  {'Dimension':<20} {'Beta':>10} {'SE':>8} {'p':>8} {'Significant':>12}")
for col, label in dims:
    X = np.column_stack([np.ones(25), m2[col].fillna(0).values, x_time])
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    r = y - X @ b
    se = np.sqrt(np.sum(r**2) / 22 * np.diag(np.linalg.inv(X.T @ X)))
    p = 2 * (1 - stats.t.cdf(np.abs(b[1] / se[1]), 22))
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    print(f"  {label:<20} {b[1]:>10.3f} {se[1]:>8.3f} {p:>8.4f} {sig:>12}")

# ============================================================
# 7. Summary
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  V1 buggy data: r(textile, policy) = {np.corrcoef(m1['textile_firms'].values, m1['policy_intensity_total'].values)[0,1]:.4f}")
print(f"  V2 fixed data: r(textile, policy_abs) = {np.corrcoef(y, x_pol)[0,1]:.4f}")
print(f"  V2 fixed data: r(textile, policy_conc) = {np.corrcoef(y, m2['policy_intensity_concentration'].values)[0,1]:.4f}")
print(f"\nKey question: After fixing all 9 bugs, does the LLM score have ANY explanatory power?")
print("Done.")
