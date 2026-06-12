"""Empirical analysis: ITS + dimension decomposition + robustness.
2015-2024 estimation window (10 years with complete firm data).
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────
import re
pol = pd.read_csv("data/policies/高阳县毛巾产业政策文件_全量_20260517_195152.csv")
def extract_year(x):
    if pd.isna(x): return None
    m = re.search(r'(\d{4})', str(x))
    return int(m.group(1)) if m else None
pol['year'] = pol['发文时间'].apply(extract_year)
media = pol.groupby('year').size().reset_index(name='media_count')
media = media[(media['year'] >= 2015) & (media['year'] <= 2026)]

ps2 = pd.read_csv("output/policy_scores_panel_v2.csv")
gy = ps2[ps2['County_Code'].astype(str) == '130628'].copy()
gy = gy[(gy['Year'] >= 2015) & (gy['Year'] <= 2026)]

mp = pd.read_csv("output/master_panel_data_v2.csv")
mp = mp[mp['County_Code'].astype(str) == '130628'][['Year', 'textile_firms', 'total_firms', 'new_textile_firms', 'new_total_firms', 'textile_firms_ma3', 'textile_ratio']].copy()
mp = mp[(mp['Year'] >= 2015) & (mp['Year'] <= 2026)]

df = mp.merge(gy[['Year', 'policy_intensity_total', 'policy_intensity_concentration',
    'equipment_sum', 'environment_sum', 'ecommerce_sum', 'brandquality_sum',
    'cluster_sum', 'finance_sum', 'education_sum', 'total_chunks']], on='Year', how='left')
df = df.merge(media, left_on='Year', right_on='year', how='left')
df['media_count'] = df['media_count'].fillna(0).astype(int)
df = df.drop(columns=['year'])
df['time'] = df['Year'] - 2014
df['post2017'] = (df['Year'] >= 2017).astype(int)
df['ln_textile'] = np.log(df['textile_firms'])

# Estimation sample: 2015-2024
est = df[df['Year'] <= 2024].copy()
n = len(est)
print(f"Estimation sample: {n} years (2015-2024)")
print(f"Dependent variable: textile_firms (annual new registrations)")
print(f"  Mean={est['textile_firms'].mean():.0f}, SD={est['textile_firms'].std():.0f}")
print(f"  Range: {est['textile_firms'].min():.0f}-{est['textile_firms'].max():.0f}")
print(f"Key regressor: policy_intensity_total")
print(f"  Mean={est['policy_intensity_total'].mean():.1f}, SD={est['policy_intensity_total'].std():.1f}")
print(f"  Non-zero years: {(est['policy_intensity_total']>0).sum()}/{n}")

# ═══════════════════════════════════════════════════════════
# TABLE 1: Descriptive statistics
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 1: Descriptive Statistics (2015-2024, n=10)")
print("="*80)
vars_desc = {
    'textile_firms': 'New textile firm registrations',
    'total_firms': 'New total firm registrations',
    'textile_ratio': 'Textile share of new firms',
    'policy_intensity_total': 'Policy intensity (absolute)',
    'policy_intensity_concentration': 'Policy intensity (concentration)',
    'equipment_sum': 'Equipment upgrade dimension',
    'environment_sum': 'Environmental governance dimension',
    'ecommerce_sum': 'E-commerce dimension',
    'brandquality_sum': 'Brand & quality dimension',
    'cluster_sum': 'Cluster synergy dimension',
    'finance_sum': 'Financial support dimension',
    'education_sum': 'Education & training dimension',
    'total_chunks': 'Report chunks (text length proxy)',
    'media_count': 'Annual media event count',
}
print(f"{'Variable':<35} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
for col, label in vars_desc.items():
    if col in est.columns:
        s = est[col]
        print(f"{label:<35} {s.mean():>8.1f} {s.std():>8.1f} {s.min():>8.1f} {s.max():>8.1f}")

# ═══════════════════════════════════════════════════════════
# TABLE 2: Correlation matrix
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 2: Correlation Matrix (Key Variables)")
print("="*80)
corr_vars = ['textile_firms', 'ln_textile', 'policy_intensity_total',
             'policy_intensity_concentration', 'total_chunks', 'media_count',
             'equipment_sum', 'environment_sum', 'ecommerce_sum', 'time']
corr_labels = ['TextileFirms', 'ln(Textile)', 'Policy(Abs)', 'Policy(Conc)',
               'Chunks', 'Media', 'Equipment', 'Environment', 'Ecommerce', 'Time']
mat = est[corr_vars].corr()
header = f"{'':>15}" + "".join(f"{l:>10}" for l in corr_labels)
print(header)
for i, (col, label) in enumerate(zip(corr_vars, corr_labels)):
    row = f"{label:>15}"
    for j in range(len(corr_vars)):
        row += f"{mat.iloc[i, j]:>10.3f}"
    print(row)

# ═══════════════════════════════════════════════════════════
# TABLE 3: ITS Baseline Regressions
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 3: ITS Baseline — Dependent Variable: ln(textile_firms)")
print("="*80)

def ols_with_hac(y, X, labels):
    """OLS with Newey-West HAC SE (lag=1)."""
    nobs = len(y)
    k = X.shape[1]
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    r = y - X @ b
    # Newey-West HAC with 1 lag
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    for t in range(nobs):
        S += r[t]**2 * X[t:t+1].T @ X[t:t+1]
    for t in range(1, nobs):
        w = 1 - 1.0/2.0  # Bartlett kernel, lag=1
        S += w * r[t] * r[t-1] * (X[t:t+1].T @ X[t-1:t] + X[t-1:t].T @ X[t:t+1])
    vcov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diag(vcov_hac))
    return b, se_hac, r

y = est['ln_textile'].values
time = est['time'].values
post = est['post2017'].values
policy = est['policy_intensity_total'].values
chunks = est['total_chunks'].values
media = est['media_count'].values

models = [
    ("(1) No controls",
     np.column_stack([np.ones(n), policy])),
    ("(2) + Time trend",
     np.column_stack([np.ones(n), policy, time])),
    ("(3) + Time + Post2017",
     np.column_stack([np.ones(n), policy, time, post])),
    ("(4) + Time + Post + Chunks",
     np.column_stack([np.ones(n), policy, time, post, chunks])),
    ("(5) + Time + Post + Media",
     np.column_stack([np.ones(n), policy, time, post, media])),
    ("(6) Full: Time+Post+Chunks+Media",
     np.column_stack([np.ones(n), policy, time, post, chunks, media])),
]

col_names = {
    1: ["Const", "Policy"],
    2: ["Const", "Policy", "Time"],
    3: ["Const", "Policy", "Time", "Post2017"],
    4: ["Const", "Policy", "Time", "Post2017", "Chunks"],
    5: ["Const", "Policy", "Time", "Post2017", "Media"],
    6: ["Const", "Policy", "Time", "Post2017", "Chunks", "Media"],
}

print(f"{'':<35} {'Policy(beta)':>12} {'SE(HAC)':>10} {'t(HAC)':>10} {'p(HAC)':>10} {'R2':>8} {'AdjR2':>8}")
for name, X in models:
    k = X.shape[1]
    b, se_hac, r = ols_with_hac(y, X, [])
    r2 = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    adj_r2 = 1 - (1-r2) * (n-1)/(n-k)
    t_hac = b[1] / se_hac[1]
    p_hac = 2 * (1 - stats.t.cdf(abs(t_hac), n-k))
    print(f"{name:<35} {b[1]:>12.4f} {se_hac[1]:>10.4f} {t_hac:>10.3f} {p_hac:>10.4f} {r2:>8.4f} {adj_r2:>8.4f}")

# ═══════════════════════════════════════════════════════════
# TABLE 4: Dimension-Level Decomposition
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 4: Dimension-Level Decomposition — DV: ln(textile_firms)")
print("="*80)
dims = [
    ('equipment_sum', 'Equipment'),
    ('environment_sum', 'Environment'),
    ('ecommerce_sum', 'E-commerce'),
    ('brandquality_sum', 'Brand&Quality'),
    ('cluster_sum', 'Cluster'),
    ('finance_sum', 'Finance'),
    ('education_sum', 'Education'),
]
print(f"{'Dimension':<20} {'Beta':>10} {'SE(HAC)':>10} {'t(HAC)':>10} {'p':>10} {'AdjR2':>8}")
for col, label in dims:
    X = np.column_stack([np.ones(n), est[col].fillna(0).values, time, post])
    k = X.shape[1]
    b, se_hac, r = ols_with_hac(y, X, [])
    r2 = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    adj_r2 = 1 - (1-r2) * (n-1)/(n-k)
    t_hac = b[1] / se_hac[1]
    p_hac = 2 * (1 - stats.t.cdf(abs(t_hac), n-k))
    sig = "***" if p_hac < 0.01 else ("**" if p_hac < 0.05 else ("*" if p_hac < 0.10 else ""))
    print(f"{label:<20} {b[1]:>10.4f} {se_hac[1]:>10.4f} {t_hac:>10.3f} {p_hac:>10.4f} {adj_r2:>8.4f} {sig:>5}")

# ═══════════════════════════════════════════════════════════
# TABLE 5: Robustness Checks
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 5: Robustness Checks")
print("="*80)

# 5a: Replacement DV — total_firms
print("\n--- 5a: DV = ln(total_firms) ---")
y_tot = np.log(est['total_firms'].values)
models_r = [
    ("Baseline", np.column_stack([np.ones(n), policy, time, post])),
    ("+Chunks", np.column_stack([np.ones(n), policy, time, post, chunks])),
    ("+Media", np.column_stack([np.ones(n), policy, time, post, media])),
]
for name, X in models_r:
    k = X.shape[1]
    b, se_hac, r = ols_with_hac(y_tot, X, [])
    r2 = 1 - np.sum(r**2) / np.sum((y_tot - y_tot.mean())**2)
    t_hac = b[1] / se_hac[1]
    p_hac = 2 * (1 - stats.t.cdf(abs(t_hac), n-k))
    print(f"  {name:<15}  Policy_beta={b[1]:.4f}  SE(HAC)={se_hac[1]:.4f}  p={p_hac:.4f}  R2={r2:.4f}")

# 5b: Replacement IV — policy concentration
print("\n--- 5b: IV = Policy Concentration (index) ---")
policy_conc = est['policy_intensity_concentration'].values
X_conc = np.column_stack([np.ones(n), policy_conc, time, post])
b_c, se_h, r = ols_with_hac(y, X_conc, [])
t_h = b_c[1] / se_h[1]
p_h = 2 * (1 - stats.t.cdf(abs(t_h), n-k))
print(f"  Policy_beta={b_c[1]:.4f}  SE(HAC)={se_h[1]:.4f}  p={p_h:.4f}")

# 5c: IV = media_count instead of LLM score
print("\n--- 5c: IV = Media Event Count ---")
X_med = np.column_stack([np.ones(n), media, time, post])
b_m, se_h, r = ols_with_hac(y, X_med, [])
t_h = b_m[1] / se_h[1]
p_h = 2 * (1 - stats.t.cdf(abs(t_h), n-k))
print(f"  Media_beta={b_m[1]:.4f}  SE(HAC)={se_h[1]:.4f}  p={p_h:.4f}")

# 5d: Lagged policy (L1)
print("\n--- 5d: Lagged Policy (L1.Policy → TextileFirms) ---")
est['L1_policy'] = est['policy_intensity_total'].shift(1)
valid = est['L1_policy'].notna()
y_lag = est.loc[valid, 'ln_textile'].values
X_lag = np.column_stack([np.ones(valid.sum()), est.loc[valid, 'L1_policy'].values,
                          est.loc[valid, 'time'].values, est.loc[valid, 'post2017'].values])
n_lag = valid.sum()
b_l, se_h, r = ols_with_hac(y_lag, X_lag, [])
t_h = b_l[1] / se_h[1]
p_h = 2 * (1 - stats.t.cdf(abs(t_h), n_lag - 4))
print(f"  L1.Policy_beta={b_l[1]:.4f}  SE(HAC)={se_h[1]:.4f}  p={p_h:.4f}  (n={n_lag})")

# 5e: Placebo breakpoints (shift true break ±1 year)
print("\n--- 5e: Placebo Breakpoints ---")
for pb_year, pb_label in [(2016, '2016'), (2018, '2018')]:
    pb = (est['Year'] >= pb_year).astype(int)
    X_pl = np.column_stack([np.ones(n), policy, time, pb])
    b_p, se_h, r = ols_with_hac(y, X_pl, [])
    k_p = X_pl.shape[1]
    t_h = b_p[1] / se_h[1]
    p_h = 2 * (1 - stats.t.cdf(abs(t_h), n - k_p))
    r2_p = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    print(f"  Break={pb_label}: Policy_beta={b_p[1]:.4f}  SE(HAC)={se_h[1]:.4f}  p={p_h:.4f}  R2={r2_p:.4f}")

# True break at 2017 (for comparison)
print(f"  Break=2017 (true): Policy_beta=0.0008  p=0.9043  (from Table 3, Model 3)")

# 5f: First-difference model (simple: ΔlnFirms ~ ΔPolicy)
print("\n--- 5f: First-Difference Model ---")
dy = np.diff(y)
dpolicy = np.diff(policy)
X_fd = np.column_stack([np.ones(n-1), dpolicy])
b_fd, se_h, r_fd = ols_with_hac(dy, X_fd, [])
n_fd = n - 1
k_fd = 2
t_h = b_fd[1] / se_h[1]
p_h = 2 * (1 - stats.t.cdf(abs(t_h), n_fd - k_fd))
r2_fd = 1 - np.sum(r_fd**2) / np.sum((dy - dy.mean())**2)
print(f"  ΔPolicy_beta={b_fd[1]:.4f}  SE(HAC)={se_h[1]:.4f}  p={p_h:.4f}  R2={r2_fd:.4f}")

# Also: correlation between ΔPolicy and ΔFirms
print(f"  r(ΔPolicy, ΔlnFirms) = {np.corrcoef(dpolicy, dy)[0,1]:.4f}")
print(f"  r(ΔPolicy, ΔFirms)   = {np.corrcoef(dpolicy, np.diff(est['textile_firms'].values))[0,1]:.4f}")

# ═══════════════════════════════════════════════════════════
# TABLE 6: Policy intensity × Break interaction
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 6: Interaction Model — Policy × Post2017")
print("="*80)
interaction = policy * post
X_int = np.column_stack([np.ones(n), policy, time, post, interaction])
b_int, se_h, r = ols_with_hac(y, X_int, [])
r2_int = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
print(f"  Policy (main effect):     beta={b_int[1]:.4f}, SE(HAC)={se_h[1]:.4f}, p={2*(1-stats.t.cdf(abs(b_int[1]/se_h[1]), n-5)):.4f}")
print(f"  Policy×Post2017 (extra):  beta={b_int[4]:.4f}, SE(HAC)={se_h[4]:.4f}, p={2*(1-stats.t.cdf(abs(b_int[4]/se_h[4]), n-5)):.4f}")
print(f"  R2={r2_int:.4f}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("EMPIRICAL SUMMARY")
print("="*80)
r_pol_tex = np.corrcoef(policy, est['textile_firms'].values)[0,1]
print(f"  r(Policy, TextileFirms) = {r_pol_tex:.4f}")
print(f"  r(Policy, ln_Textile)  = {np.corrcoef(policy, y)[0,1]:.4f}")
print(f"  r(Time, TextileFirms)  = {np.corrcoef(time, est['textile_firms'].values)[0,1]:.4f}")
print(f"  r(Chunks, Policy)      = {np.corrcoef(chunks, policy)[0,1]:.4f}")
# Simple model without time
X_simple = np.column_stack([np.ones(n), policy])
b_s, se_h, r_s = ols_with_hac(y, X_simple, [])
print(f"  Simple OLS: Policy_beta={b_s[1]:.4f}, p={2*(1-stats.t.cdf(abs(b_s[1]/se_h[1]), n-2)):.4f}")
print("\nDone.")
