"""Rigorous empirical improvements addressing all reviewer concerns.
Improvements:
  I.   Residualization: strip text-length effect from policy scores
  II.  PCA: reduce 7 dimensions to 1-2 components, avoid multiple-comparison trap
  III. Randomization inference: permutation-based p-values replace asymptotic HAC
  IV.  Quarterly-frequency analysis: leverage 25-quarter textile indices panel
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import warnings, re, os
warnings.filterwarnings('ignore')

np.random.seed(42)
N_PERMUTATIONS = 5000

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

# Annual policy + firm data (same as before)
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
mp = mp[mp['County_Code'].astype(str) == '130628'][['Year', 'textile_firms', 'total_firms', 'textile_firms_ma3', 'textile_ratio']].copy()
mp = mp[(mp['Year'] >= 2015) & (mp['Year'] <= 2026)]

df = mp.merge(gy[['Year', 'policy_intensity_total', 'policy_intensity_concentration',
    'equipment_sum', 'environment_sum', 'ecommerce_sum', 'brandquality_sum',
    'cluster_sum', 'finance_sum', 'education_sum', 'total_chunks']], on='Year', how='left')
df = df.merge(media, left_on='Year', right_on='year', how='left')
df['media_count'] = df['media_count'].fillna(0).astype(int)
df = df.drop(columns=['year'])
df['time'] = df['Year'] - 2014
df['post2017'] = (df['Year'] >= 2017).astype(int)

est = df[df['Year'] <= 2024].copy()
n = len(est)
y_raw = est['textile_firms'].values
y_ln = np.log(y_raw)
time = est['time'].values
post = est['post2017'].values
chunks = est['total_chunks'].values
media_arr = est['media_count'].values
policy_total = est['policy_intensity_total'].values

dim_names = ['equipment_sum', 'environment_sum', 'ecommerce_sum', 'brandquality_sum',
             'cluster_sum', 'finance_sum', 'education_sum']
dim_labels = ['Equipment', 'Environment', 'Ecommerce', 'BrandQuality',
              'Cluster', 'Finance', 'Education']

# ═══════════════════════════════════════════════════════════
# IMPROVEMENT I: RESIDUALIZATION
# Strip text-length effect from policy scores
# ═══════════════════════════════════════════════════════════
print("="*80)
print("IMPROVEMENT I: RESIDUALIZATION — Removing Text-Length Confound")
print("="*80)

# Regress policy_total on chunks, extract residuals
X_resid = np.column_stack([np.ones(n), chunks])
b_resid = np.linalg.inv(X_resid.T @ X_resid) @ X_resid.T @ policy_total
policy_predicted = X_resid @ b_resid
policy_residual = policy_total - policy_predicted

print(f"Policy ~ Chunks: intercept={b_resid[0]:.2f}, slope={b_resid[1]:.4f}")
print(f"Corr(Policy_raw, Chunks) = {np.corrcoef(policy_total, chunks)[0,1]:.4f}")
print(f"Corr(Policy_residual, Chunks) = {np.corrcoef(policy_residual, chunks)[0,1]:.4f}")
print(f"Policy_residual stats: mean={policy_residual.mean():.1f}, SD={policy_residual.std():.1f}")
print(f"  Range: [{policy_residual.min():.1f}, {policy_residual.max():.1f}]")
print(f"\nInterpretation: Residual > 0 means policy intensity EXCEEDS what text length alone predicts.")
print(f"  Positive residuals (abnormal policy effort):")
for i in range(n):
    yr = est.iloc[i]['Year']
    raw = policy_total[i]
    pred = policy_predicted[i]
    res = policy_residual[i]
    flag = " <<< ABNORMAL EFFORT" if res > 5 else ""
    print(f"  {int(yr)}: raw={raw:5.1f}  predicted={pred:5.1f}  residual={res:+6.1f}{flag}")

# Residualize each dimension too
dim_residuals = {}
for col in dim_names:
    vals = est[col].fillna(0).values
    b_d = np.linalg.inv(X_resid.T @ X_resid) @ X_resid.T @ vals
    dim_residuals[col] = vals - X_resid @ b_d

# ═══════════════════════════════════════════════════════════
# IMPROVEMENT II: PCA DIMENSION REDUCTION
# Replace 7 separate tests with 1-2 principal components
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("IMPROVEMENT II: PCA — Dimensionality Reduction on 7 Policy Dimensions")
print("="*80)

# PCA on raw dimensions
dim_matrix_raw = np.column_stack([est[c].fillna(0).values for c in dim_names])
pca_raw = PCA()
pca_raw.fit(dim_matrix_raw)
print("PCA on RAW dimension scores:")
print(f"  PC1 explains {pca_raw.explained_variance_ratio_[0]:.1%} of variance")
print(f"  PC2 explains {pca_raw.explained_variance_ratio_[1]:.1%} of variance")
print(f"  PC1+PC2 cumulative: {sum(pca_raw.explained_variance_ratio_[:2]):.1%}")
print(f"  PC1 loadings:")
for i, (name, label) in enumerate(zip(dim_names, dim_labels)):
    print(f"    {label:<15}: {pca_raw.components_[0][i]:+7.4f}")
print(f"  PC2 loadings:")
for i, (name, label) in enumerate(zip(dim_names, dim_labels)):
    print(f"    {label:<15}: {pca_raw.components_[1][i]:+7.4f}")

# PCA on residualized dimensions
dim_matrix_resid = np.column_stack([dim_residuals[c] for c in dim_names])
pca_resid = PCA()
pca_resid.fit(dim_matrix_resid)
pc1_resid = pca_resid.transform(dim_matrix_resid)[:, 0]
pc2_resid = pca_resid.transform(dim_matrix_resid)[:, 1]

print(f"\nPCA on RESIDUALIZED dimension scores:")
print(f"  PC1 explains {pca_resid.explained_variance_ratio_[0]:.1%} of variance")
print(f"  PC2 explains {pca_resid.explained_variance_ratio_[1]:.1%} of variance")
print(f"  PC1+PC2 cumulative: {sum(pca_resid.explained_variance_ratio_[:2]):.1%}")
print(f"  PC1 loadings:")
for i, (name, label) in enumerate(zip(dim_names, dim_labels)):
    print(f"    {label:<15}: {pca_resid.components_[0][i]:+7.4f}")
print(f"  PC2 loadings:")
for i, (name, label) in enumerate(zip(dim_names, dim_labels)):
    print(f"    {label:<15}: {pca_resid.components_[1][i]:+7.4f}")

# ═══════════════════════════════════════════════════════════
# IMPROVEMENT III: RANDOMIZATION INFERENCE
# Permutation tests replace asymptotic HAC p-values
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("IMPROVEMENT III: RANDOMIZATION INFERENCE — Permutation Tests")
print("="*80)

def permutation_test(y, X, policy_idx=1, n_perm=N_PERMUTATIONS):
    """Permutation test: shuffle Y, recompute beta, get empirical p-value.
    H0: policy coefficient = 0 (no association).
    """
    b_obs = np.linalg.inv(X.T @ X) @ X.T @ y
    t_obs = b_obs[policy_idx]
    # Permute Y
    greater = 0
    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = np.random.permutation(y)
        b_perm = np.linalg.inv(X.T @ X) @ X.T @ y_perm
        null_dist[i] = b_perm[policy_idx]
        if abs(b_perm[policy_idx]) >= abs(t_obs):
            greater += 1
    p_two_sided = greater / n_perm
    return t_obs, null_dist, p_two_sided

def permutation_test_block(y, X, policy_idx=1, n_perm=N_PERMUTATIONS):
    """Permutation test preserving the post2017 block structure.
    Only permute WITHIN pre-2017 and post-2017 blocks to respect the break.
    """
    b_obs = np.linalg.inv(X.T @ X) @ X.T @ y
    t_obs = b_obs[policy_idx]
    # Identify pre and post blocks
    # post2017 is in X[:, 3] for our standard specification
    pre_idx = np.where(X[:, 3] == 0)[0]  # post2017 column
    post_idx = np.where(X[:, 3] == 1)[0]
    greater = 0
    for i in range(n_perm):
        y_perm = y.copy()
        y_perm[pre_idx] = np.random.permutation(y[pre_idx])
        y_perm[post_idx] = np.random.permutation(y[post_idx])
        b_perm = np.linalg.inv(X.T @ X) @ X.T @ y_perm
        if abs(b_perm[policy_idx]) >= abs(t_obs):
            greater += 1
    p_two_sided = greater / n_perm
    return t_obs, p_two_sided

# Test 1: Raw policy (original specification)
X_base = np.column_stack([np.ones(n), policy_total, time, post])
t1, null1, p1 = permutation_test(y_ln, X_base)
t1b, p1b = permutation_test_block(y_ln, X_base)
print(f"\nTest 1: Raw Policy Intensity (Model 3 specification)")
print(f"  Observed beta = {t1:.6f}")
print(f"  Permutation p (unrestricted) = {p1:.4f}")
print(f"  Permutation p (block-respecting) = {p1b:.4f}")
print(f"  Null distribution: mean={null1.mean():.6f}, SD={null1.std():.6f}")
print(f"  95% CI of null: [{np.percentile(null1, 2.5):.6f}, {np.percentile(null1, 97.5):.6f}]")

# Test 2: Residualized policy
X_res = np.column_stack([np.ones(n), policy_residual, time, post])
t2, null2, p2 = permutation_test(y_ln, X_res)
t2b, p2b = permutation_test_block(y_ln, X_res)
print(f"\nTest 2: Residualized Policy (text-length orthogonalized)")
print(f"  Observed beta = {t2:.6f}")
print(f"  Permutation p (unrestricted) = {p2:.4f}")
print(f"  Permutation p (block-respecting) = {p2b:.4f}")
print(f"  Null distribution: mean={null2.mean():.6f}, SD={null2.std():.6f}")

# Test 3: PC1 from residualized dimensions
X_pc = np.column_stack([np.ones(n), pc1_resid, time, post])
t3, null3, p3 = permutation_test(y_ln, X_pc)
t3b, p3b = permutation_test_block(y_ln, X_pc)
print(f"\nTest 3: PC1 of Residualized Dimensions (1 test instead of 7)")
print(f"  Observed beta = {t3:.6f}")
print(f"  Permutation p (unrestricted) = {p3:.4f}")
print(f"  Permutation p (block-respecting) = {p3b:.4f}")

# Test 4: Media count (simpler proxy)
X_med = np.column_stack([np.ones(n), media_arr, time, post])
t4, null4, p4 = permutation_test(y_ln, X_med)
t4b, p4b = permutation_test_block(y_ln, X_med)
print(f"\nTest 4: Media Event Count (simple proxy)")
print(f"  Observed beta = {t4:.6f}")
print(f"  Permutation p (unrestricted) = {p4:.4f}")
print(f"  Permutation p (block-respecting) = {p4b:.4f}")

# Test 5: Environment dimension (residualized)
env_resid = dim_residuals['environment_sum']
X_env = np.column_stack([np.ones(n), env_resid, time, post])
t5, null5, p5 = permutation_test(y_ln, X_env)
t5b, p5b = permutation_test_block(y_ln, X_env)
print(f"\nTest 5: Environment Dimension (residualized)")
print(f"  Observed beta = {t5:.6f}")
print(f"  Permutation p (unrestricted) = {p5:.4f}")
print(f"  Permutation p (block-respecting) = {p5b:.4f}")

# ═══════════════════════════════════════════════════════════
# SUMMARY: Permutation-based inference table
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUMMARY: Permutation-Based Inference (all tests, block-respecting)")
print("="*80)
print(f"{'Variable':<30} {'Observed Beta':>14} {'Permutation p':>14} {'Asymptotic p':>14} {'Conclusion':>20}")
# Recompute asymptotic p for comparison
for label, beta_obs, X_mat in [
    ("Raw Policy (Model 3)", t1, X_base),
    ("Residualized Policy", t2, X_res),
    ("PC1 (Resid Dimensions)", t3, X_pc),
    ("Media Events", t4, X_med),
    ("Environment (Resid)", t5, X_env),
]:
    k = X_mat.shape[1]
    b = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_ln
    r = y_ln - X_mat @ b
    s2 = np.sum(r**2) / (n - k)
    se_ols = np.sqrt(s2 * np.diag(np.linalg.inv(X_mat.T @ X_mat)))
    t_ols = b[1] / se_ols[1]
    p_ols = 2 * (1 - stats.t.cdf(abs(t_ols), n - k))
    # Get permutation p
    if "Raw" in label: p_perm = p1b
    elif "Residualized" in label: p_perm = p2b
    elif "PC1" in label: p_perm = p3b
    elif "Media" in label: p_perm = p4b
    elif "Environment" in label: p_perm = p5b
    else: p_perm = 1.0
    conclusion = "Significant at 10%" if p_perm < 0.10 else ("Significant at 5%" if p_perm < 0.05 else "Not significant")
    print(f"{label:<30} {b[1]:>14.6f} {p_perm:>14.4f} {p_ols:>14.4f} {conclusion:>20}")

# ═══════════════════════════════════════════════════════════
# IMPROVEMENT IV: QUARTERLY TEXTILE INDICES ANALYSIS
# Leverage n≈25 quarterly observations
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("IMPROVEMENT IV: Quarterly Textile Indices — Higher-Frequency Analysis")
print("="*80)

# Load all quarterly indices
idx_dir = "data/textile_indices"
quarterly_indices = {}
for fname in sorted(os.listdir(idx_dir)):
    if not fname.endswith('.xlsx'): continue
    fp = os.path.join(idx_dir, fname)
    df_idx = pd.read_excel(fp)
    label = fname.replace('河北·高阳纺织指数-', '').replace('.xlsx', '')
    # Parse time column
    time_col = df_idx.columns[0]
    val_col = df_idx.columns[1]
    df_idx['period'] = df_idx[time_col].astype(str)
    # Keep only quarterly data (Q1-Q4 format)
    df_idx = df_idx[df_idx['period'].str.contains(r'Q\d', regex=True)].copy()
    if len(df_idx) == 0:
        # Try monthly format
        df_idx = df_idx[df_idx['period'].str.match(r'\d{4}-\d{2}')].copy()
        if len(df_idx) == 0:
            continue
    df_idx['value'] = pd.to_numeric(df_idx[val_col], errors='coerce')
    quarterly_indices[label] = df_idx[['period', 'value']].copy()
    print(f"  {label}: {len(df_idx)} observations, range [{df_idx['value'].min():.2f}, {df_idx['value'].max():.2f}]")

# Build unified quarterly panel
# Key indices for our analysis
key_indices = ['产业景气指数', '产业竞争力指数', '政策支持指数', '产业发展指数']
print(f"\n===== Quarterly Index Panel =====")
# Merge key indices on period
merged_idx = None
for ki in key_indices:
    if ki in quarterly_indices:
        sub = quarterly_indices[ki].copy()
        sub = sub.rename(columns={'value': ki})
        if merged_idx is None:
            merged_idx = sub
        else:
            merged_idx = merged_idx.merge(sub, on='period', how='outer')

if merged_idx is not None:
    merged_idx = merged_idx.sort_values('period').dropna(subset=key_indices, how='all')
    print(f"Merged panel: {len(merged_idx)} quarters")
    print(merged_idx.head(10).to_string())

    # Compute correlations
    print(f"\n===== Correlations among quarterly indices =====")
    vals = merged_idx[key_indices].dropna()
    if len(vals) > 5:
        corr_mat = vals.corr()
        print(corr_mat.to_string())

    # Annualize: average each index per year
    merged_idx['Year'] = merged_idx['period'].str[:4].astype(int)
    annual_idx = merged_idx.groupby('Year')[key_indices].mean().reset_index()

    # Merge with LLM policy scores
    annual_idx = annual_idx.merge(
        est[['Year', 'policy_intensity_total', 'total_chunks']].copy(),
        on='Year', how='inner')
    annual_idx['policy_residual'] = annual_idx['policy_intensity_total'] - \
        (b_resid[0] + b_resid[1] * annual_idx['total_chunks'])

    print(f"\n===== Annualized Indices vs LLM Policy Scores =====")
    print(f"Years: {list(annual_idx['Year'].astype(int).values)}")
    if len(annual_idx) >= 3:
        for ki in key_indices:
            if ki in annual_idx.columns:
                r_raw = np.corrcoef(
                    annual_idx['policy_intensity_total'].values,
                    annual_idx[ki].values)[0, 1]
                r_resid = np.corrcoef(
                    annual_idx['policy_residual'].values,
                    annual_idx[ki].values)[0, 1]
                print(f"  r(LLM_policy_raw, {ki}) = {r_raw:+.4f}")
                print(f"  r(LLM_policy_resid, {ki}) = {r_resid:+.4f}")

# ═══════════════════════════════════════════════════════════
# FINAL TABLE: Comprehensive Results (for paper)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("FINAL RESULTS TABLE — Permutation-Based Inference")
print("="*80)
print(f"Sample: n={n} years (2015-2024)")
print(f"Permutations: {N_PERMUTATIONS} (block-respecting pre/post 2017)")
print()
print(f"{'Panel A: Annual ITS (DV = ln Textile Firms)':>60}")
print(f"{'Variable':<30} {'Coef':>10} {'Permut p':>10} {'OLS p':>10}")
# Recompute everything cleanly
results_a = [
    ("Raw Policy Intensity", policy_total),
    ("Residualized Policy", policy_residual),
    ("PC1 (7 dims, resid)", pc1_resid),
    ("Environment (resid)", env_resid),
    ("Media Events", media_arr),
]
for label, var in results_a:
    X_tmp = np.column_stack([np.ones(n), var, time, post])
    k_tmp = 4
    b_tmp = np.linalg.inv(X_tmp.T @ X_tmp) @ X_tmp.T @ y_ln
    r_tmp = y_ln - X_tmp @ b_tmp
    s2_tmp = np.sum(r_tmp**2) / (n - k_tmp)
    se_tmp = np.sqrt(s2_tmp * np.diag(np.linalg.inv(X_tmp.T @ X_tmp)))
    p_ols = 2 * (1 - stats.t.cdf(abs(b_tmp[1]/se_tmp[1]), n - k_tmp))
    _, p_perm = permutation_test_block(y_ln, X_tmp)
    r2_tmp = 1 - np.sum(r_tmp**2) / np.sum((y_ln - y_ln.mean())**2)
    sig = "**" if p_perm < 0.05 else ("*" if p_perm < 0.10 else "")
    print(f"{label:<30} {b_tmp[1]:>10.4f} {p_perm:>10.4f} {p_ols:>10.4f} {sig:>5}  R2={r2_tmp:.3f}")

print(f"\n{'Panel B: Construct Validity (LLM vs Official Indices)':>60}")
if len(annual_idx) >= 3:
    for ki in key_indices:
        if ki in annual_idx.columns:
            r_val = np.corrcoef(annual_idx['policy_residual'].values, annual_idx[ki].values)[0, 1]
            n_val = annual_idx[[ki, 'policy_residual']].dropna().shape[0]
            print(f"  r(Residualized LLM Policy, Official {ki}) = {r_val:+.4f}  (n={n_val})")

print(f"\n{'Panel C: Quarterly Index Autocorrelation Structure':>60}")
for ki in key_indices:
    if ki in merged_idx.columns:
        series = merged_idx[ki].dropna().values
        if len(series) > 4:
            ac1 = np.corrcoef(series[:-1], series[1:])[0, 1]
            print(f"  {ki}: AR(1) = {ac1:+.4f}  (n={len(series)} quarters)")

print("\nDone.")
