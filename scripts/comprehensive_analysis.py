"""
综合分析脚本 - 9个探究方向
基于实际数据可用性优化
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def sig_star(p):
    """返回显著性标记"""
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return 'ns'

def load_data():
    """加载数据"""
    master = pd.read_csv('output/master_panel_data_v2.csv')
    master['County_Code'] = master['County_Code'].astype(str)
    gy = master[master['County_Code'] == '130628'].sort_values('Year').reset_index(drop=True)
    
    policy = pd.read_csv('output/policy_scores_panel.csv')
    policy['County_Code'] = policy['County_Code'].astype(str)
    
    ent = pd.read_csv('output/gaoyang_enterprise_registration.csv')
    
    gy['net_entry'] = gy['textile_firms'].diff()
    
    return gy, policy, ent

# ============================================================
# 方向1：纺织指数与政策强度关联分析
# ============================================================
def analysis1_textile_indices(gy):
    print("\n" + "="*80)
    print("探究1：纺织指数与政策强度的关联分析")
    print("="*80)
    
    indices = [c for c in gy.columns if c.startswith('index_')]
    
    print(f"\n【数据概况】纺织指数非空数量")
    for idx in indices:
        n = gy[idx].notna().sum()
        print(f"  {idx}: {n}个非空值")
    
    # 1.1 纺织指数的时间趋势
    print(f"\n【1.1】纺织指数2017前后对比")
    print(f"  {'指数名称':<30} {'2017前':>10} {'2017后':>10} {'变化':>8} {'p值':>10} {'显著'}")
    print(f"  {'-'*78}")
    
    for idx in indices:
        valid = gy.dropna(subset=[idx])
        if len(valid) < 4:
            continue
        
        pre = valid[valid['Year'] < 2017][idx]
        post = valid[valid['Year'] >= 2017][idx]
        
        if len(pre) < 2 or len(post) < 2:
            continue
        
        t_stat, p_val = sp_stats.ttest_ind(post, pre)
        mean_pre = pre.mean()
        mean_post = post.mean()
        change = mean_post - mean_pre
        
        print(f"  {idx:<30} {mean_pre:>10.2f} {mean_post:>10.2f} {change:>+8.2f} {p_val:>10.4f} {sig_star(p_val)}")
    
    # 1.2 政策强度→纺织指数（面板回归）
    print(f"\n【1.2】政策强度→纺织指数回归")
    print(f"  {'指数名称':<30} {'beta':>10} {'p值':>10} {'R2':>8} {'n':>4} {'显著'}")
    print(f"  {'-'*70}")
    
    results = {}
    for idx in indices:
        valid = gy.dropna(subset=[idx, 'policy_intensity_total'])
        if len(valid) < 5:
            continue
        
        y = valid[idx]
        X = sm.add_constant(valid[['policy_intensity_total']])
        m = sm.OLS(y, X).fit()
        
        beta = m.params['policy_intensity_total']
        p = m.pvalues['policy_intensity_total']
        r2 = m.rsquared
        
        results[idx] = {'beta': beta, 'p': p, 'r2': r2}
        print(f"  {idx:<30} {beta:>10.4f} {p:>10.4f} {r2:>8.4f} {len(valid):>4} {sig_star(p)}")
    
    # 1.3 各政策维度对产业竞争力的影响
    print(f"\n【1.3】各政策维度对产业竞争力的影响")
    if 'index_competitiveness' in gy.columns:
        dims = ['equipment_index', 'environment_index', 'ecommerce_index',
                'cluster_index', 'finance_index']
        
        print(f"  {'政策维度':<25} {'beta':>10} {'p值':>10} {'R2':>8} {'n':>4} {'显著'}")
        print(f"  {'-'*65}")
        
        for dim in dims:
            valid = gy.dropna(subset=['index_competitiveness', dim])
            if len(valid) < 5:
                continue
            
            y = valid['index_competitiveness']
            X = sm.add_constant(valid[[dim]])
            m = sm.OLS(y, X).fit()
            
            beta = m.params[dim]
            p = m.pvalues[dim]
            r2 = m.rsquared
            print(f"  {dim:<25} {beta:>10.4f} {p:>10.4f} {r2:>8.4f} {len(valid):>4} {sig_star(p)}")

# ============================================================
# 方向2：政策协同效应细化
# ============================================================
def analysis2_policy_synergy(gy):
    print("\n" + "="*80)
    print("探究2：政策协同效应细化分析")
    print("="*80)
    
    dims = ['equipment_index', 'environment_index', 'ecommerce_index',
            'cluster_index', 'finance_index']
    
    # 2.1 测试所有政策组合
    print(f"\n【2.1】所有政策组合的协同效应")
    print(f"  {'组合':<35} {'交互项beta':>12} {'p值':>10} {'R2':>8} {'n':>4} {'结论'}")
    print(f"  {'-'*85}")
    
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            d1, d2 = dims[i], dims[j]
            valid = gy.dropna(subset=['textile_firms', d1, d2])
            if len(valid) < 8:
                continue
            
            valid = valid.copy()
            valid['interaction'] = valid[d1] * valid[d2]
            
            y = valid['textile_firms']
            X = sm.add_constant(valid[[d1, d2, 'interaction']])
            m = sm.OLS(y, X).fit()
            
            beta_int = m.params['interaction']
            p_int = m.pvalues['interaction']
            
            if p_int < 0.1 and beta_int > 0:
                conclusion = "协同增效"
            elif p_int < 0.1 and beta_int < 0:
                conclusion = "替代效应"
            else:
                conclusion = "不显著"
            
            print(f"  {d1}x{d2:<28} {beta_int:>12.4f} {p_int:>10.4f} {m.rsquared:>8.4f} {len(valid):>4} {conclusion}")
    
    # 2.2 三向交互项
    print(f"\n【2.2】三向交互效应（设备x环保x电商）")
    valid = gy.dropna(subset=['textile_firms', 'equipment_index', 'environment_index', 'ecommerce_index'])
    if len(valid) >= 10:
        valid = valid.copy()
        valid['equip_env'] = valid['equipment_index'] * valid['environment_index']
        valid['triple'] = valid['equipment_index'] * valid['environment_index'] * valid['ecommerce_index']
        
        y = valid['textile_firms']
        X = sm.add_constant(valid[['equipment_index', 'environment_index', 'ecommerce_index',
                                    'equip_env', 'triple']])
        m = sm.OLS(y, X).fit()
        
        beta_triple = m.params['triple']
        p_triple = m.pvalues['triple']
        print(f"  三向交互项: beta={beta_triple:.4f}, p={p_triple:.4f} {sig_star(p_triple)}")
        print(f"  R2 = {m.rsquared:.4f}, n={len(valid)}")
        
        if p_triple < 0.1:
            direction = "高阶协同" if beta_triple > 0 else "高阶替代"
            print(f"  [{direction}]")
    else:
        print(f"  样本不足({len(valid)}), 跳过")

# ============================================================
# 方向3：空间溢出效应
# ============================================================
def analysis3_spatial_spillover(policy_df):
    print("\n" + "="*80)
    print("探究3：政策空间溢出效应（省→市→县传导）")
    print("="*80)
    
    policy_df['County_Code'] = policy_df['County_Code'].astype(str)
    
    hebei = policy_df[policy_df['County_Code'] == '130000'][['Year', 'policy_intensity_total']].copy()
    hebei.columns = ['Year', 'hebei_policy']
    
    baoding = policy_df[policy_df['County_Code'] == '130600'][['Year', 'policy_intensity_total']].copy()
    baoding.columns = ['Year', 'baoding_policy']
    
    gaoyang = policy_df[policy_df['County_Code'] == '130628'][['Year', 'policy_intensity_total']].copy()
    gaoyang.columns = ['Year', 'gaoyang_policy']
    
    merged = hebei.merge(baoding, on='Year', how='left')
    merged = merged.merge(gaoyang, on='Year', how='left')
    merged = merged.dropna(subset=['hebei_policy', 'gaoyang_policy']).sort_values('Year')
    
    if len(merged) < 8:
        print(f"  样本不足({len(merged)}), 跳过")
        return
    
    print(f"\n【3.1】省级政策→县级政策（传导效应）")
    y = merged['gaoyang_policy']
    X = sm.add_constant(merged[['hebei_policy']])
    m = sm.OLS(y, X).fit()
    
    beta = m.params['hebei_policy']
    p = m.pvalues['hebei_policy']
    print(f"  hebei_policy -> gaoyang_policy: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")
    print(f"  R2 = {m.rsquared:.4f}, n={len(merged)}")
    
    if p < 0.05:
        print(f"  [验证通过] 省级政策对县级政策有显著传导效应")
        print(f"  河北省政策每增加1单位，高阳县政策增加{beta:.2f}单位")
    
    # 3.2 构建"政策压力指数"
    print(f"\n【3.2】政策压力指数 = 上级政策强度 x 时间衰减")
    merged['hebei_lag1'] = merged['hebei_policy'].shift(1)
    merged['hebei_lag2'] = merged['hebei_policy'].shift(2)
    
    valid = merged.dropna(subset=['hebei_lag1', 'hebei_lag2']).copy()
    valid['policy_pressure'] = valid['hebei_policy'] + 0.7*valid['hebei_lag1'] + 0.5*valid['hebei_lag2']
    
    ent = pd.read_csv('output/gaoyang_enterprise_registration.csv')
    valid = valid.merge(ent[['Year', 'textile_firms']], on='Year', how='left')
    valid = valid.dropna(subset=['textile_firms', 'policy_pressure'])
    
    if len(valid) >= 8:
        y = valid['textile_firms']
        X = sm.add_constant(valid[['policy_pressure']])
        m = sm.OLS(y, X).fit()
        
        beta = m.params['policy_pressure']
        p = m.pvalues['policy_pressure']
        print(f"  policy_pressure -> textile_firms: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")
        print(f"  R2 = {m.rsquared:.4f}, n={len(valid)}")

# ============================================================
# 方向4：企业生命周期分析
# ============================================================
def analysis4_enterprise_lifecycle(gy):
    print("\n" + "="*80)
    print("探究4：企业生命周期与政策关系")
    print("="*80)
    
    # 4.1 新增企业 vs 新增纺织企业比例
    print(f"\n【4.1】新增企业结构变化（专业化程度）")
    valid = gy.dropna(subset=['new_textile_firms', 'new_total_firms'])
    if len(valid) > 0:
        valid = valid.copy()
        valid['textile_share_new'] = valid['new_textile_firms'] / valid['new_total_firms']
        
        pre = valid[(valid['Year'] >= 2008) & (valid['Year'] < 2017)]['textile_share_new'].dropna()
        post = valid[(valid['Year'] >= 2017)]['textile_share_new'].dropna()
        
        if len(pre) > 1 and len(post) > 1:
            print(f"  2008-2016新增纺织占比均值: {pre.mean():.3f}")
            print(f"  2017-2024新增纺织占比均值: {post.mean():.3f}")
            print(f"  变化: {post.mean() - pre.mean():+.3f}")
            
            t_stat, p_val = sp_stats.ttest_ind(post, pre)
            print(f"  t检验: p={p_val:.4f} {sig_star(p_val)}")
        else:
            print(f"  数据不足: pre={len(pre)}, post={len(post)}")
    else:
        print(f"  无有效数据")
    
    # 4.2 规模以上企业数量变化
    print(f"\n【4.2】规模以上工业企业数量变化")
    if 'above_scale_ind_firms' in gy.columns:
        valid = gy.dropna(subset=['above_scale_ind_firms'])
        if len(valid) >= 4:
            pre = valid[valid['Year'] < 2017]['above_scale_ind_firms']
            post = valid[valid['Year'] >= 2017]['above_scale_ind_firms']
            
            if len(pre) >= 2 and len(post) >= 2:
                print(f"  2017前规模以上企业均值: {pre.mean():.1f}")
                print(f"  2017后规模以上企业均值: {post.mean():.1f}")
                print(f"  变化: {post.mean() - pre.mean():+.1f}")
                
                t_stat, p_val = sp_stats.ttest_ind(post, pre)
                print(f"  t检验: p={p_val:.4f} {sig_star(p_val)}")
            else:
                print(f"  样本不足: pre={len(pre)}, post={len(post)}")
    
    # 4.3 企业存活率（净进入/新增）
    print(f"\n【4.3】企业存活率（净进入/新增）")
    gy = gy.copy()
    gy['survival_rate'] = gy['net_entry'] / gy['new_textile_firms']
    valid = gy.dropna(subset=['survival_rate'])
    
    pre = valid[valid['Year'] < 2017]['survival_rate']
    post = valid[valid['Year'] >= 2017]['survival_rate']
    
    if len(pre) > 1 and len(post) > 1:
        print(f"  2017前存活率均值: {pre.mean():.3f}")
        print(f"  2017后存活率均值: {post.mean():.3f}")
        print(f"  变化: {post.mean() - pre.mean():+.3f}")
    else:
        print(f"  数据不足: pre={len(pre)}, post={len(post)}")

# ============================================================
# 方向5：宏观经济变量控制检验
# ============================================================
def analysis5_macro_controls(gy):
    print("\n" + "="*80)
    print("探究5：宏观经济变量控制检验")
    print("="*80)
    
    macro_vars = ['secondary_industry_va', 'primary_industry_va',
                  'loan_balance', 'deposit_balance', 'local_fiscal_revenue']
    
    print(f"\n【数据概况】宏观变量非空数量")
    valid_macro = []
    for var in macro_vars:
        if var not in gy.columns:
            continue
        n = gy[var].notna().sum()
        print(f"  {var}: {n}个非空值")
        if n >= 8:
            valid_macro.append(var)
    
    if not valid_macro:
        print(f"\n  无足够数据的宏观变量（均需>=8个非空值）")
        print(f"  建议：使用其他控制策略或接受无法控制宏观变量的局限")
        return
    
    print(f"\n【5.1】候选控制变量与textile_firms的相关性")
    print(f"  {'变量':<30} {'n':>4} {'Pearson r':>12} {'p值':>10}")
    print(f"  {'-'*60}")
    
    selected_controls = []
    for var in valid_macro:
        valid = gy.dropna(subset=['textile_firms', var])
        if len(valid) < 8:
            continue
        
        corr, p_val = pearsonr(valid['textile_firms'], valid[var])
        print(f"  {var:<30} {len(valid):>4} {corr:>12.4f} {p_val:>10.4f}")
        
        if p_val < 0.2:
            selected_controls.append(var)
    
    # 5.2 基准回归 vs 控制回归
    print(f"\n【5.2】基准回归 vs 控制回归对比")
    
    base_data = gy.dropna(subset=['textile_firms', 'policy_intensity_total'])
    if len(base_data) < 5:
        print(f"  基准数据不足({len(base_data)})")
        return
    
    y = base_data['textile_firms']
    X = sm.add_constant(base_data[['policy_intensity_total']])
    m_base = sm.OLS(y, X).fit()
    
    print(f"\n  {'模型':<20} {'policy beta':>12} {'policy p':>10} {'R2':>8} {'n':>6}")
    print(f"  {'-'*60}")
    print(f"  {'基准(无控制)':<20} {m_base.params['policy_intensity_total']:>12.4f} {m_base.pvalues['policy_intensity_total']:>10.4f} {m_base.rsquared:>8.4f} {len(base_data):>6}")
    
    # 逐步加入控制变量
    for ctrl in selected_controls[:3]:
        valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total', ctrl])
        if len(valid) < len([ctrl]) + 3:
            continue
        
        y = valid['textile_firms']
        X = sm.add_constant(valid[['policy_intensity_total', ctrl]])
        m = sm.OLS(y, X).fit()
        
        print(f"  +{ctrl:<17} {m.params['policy_intensity_total']:>12.4f} {m.pvalues['policy_intensity_total']:>10.4f} {m.rsquared:>8.4f} {len(valid):>6}")
    
    # 所有控制变量一起
    if len(selected_controls) >= 2:
        ctrl_subset = selected_controls[:2]
        valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total'] + ctrl_subset)
        if len(valid) >= len(ctrl_subset) + 3:
            y = valid['textile_firms']
            X = sm.add_constant(valid[['policy_intensity_total'] + ctrl_subset])
            m_full = sm.OLS(y, X).fit()
            
            print(f"  {'全控制':<20} {m_full.params['policy_intensity_total']:>12.4f} {m_full.pvalues['policy_intensity_total']:>10.4f} {m_full.rsquared:>8.4f} {len(valid):>6}")
            
            beta_change = abs(m_full.params['policy_intensity_total'] - m_base.params['policy_intensity_total']) / abs(m_base.params['policy_intensity_total'])
            if beta_change < 0.2:
                print(f"\n  [结论] 控制宏观变量后，政策效应变化{beta_change*100:.1f}%，说明政策效应稳健")
            else:
                print(f"\n  [结论] 控制宏观变量后，政策效应变化{beta_change*100:.1f}%，部分效应可能被宏观变量吸收")

# ============================================================
# 方向6：分阶段分析
# ============================================================
def analysis6_phase_analysis(gy):
    print("\n" + "="*80)
    print("探究6：分阶段分析（政策演进视角）")
    print("="*80)
    
    phases = [
        (2000, 2008, '政策萌芽期'),
        (2009, 2016, '政策积累期'),
        (2017, 2024, '政策爆发期')
    ]
    
    print(f"\n  {'阶段':<15} {'年份':>10} {'n':>4} {'beta':>10} {'p值':>10} {'R2':>8} {'显著'}")
    print(f"  {'-'*68}")
    
    results = {}
    for start, end, name in phases:
        phase_data = gy[(gy['Year'] >= start) & (gy['Year'] <= end)]
        valid = phase_data.dropna(subset=['textile_firms', 'policy_intensity_total'])
        
        if len(valid) < 5:
            print(f"  {name:<15} {start}-{end:>4} {len(valid):>4} {'样本不足':>33}")
            results[name] = None
            continue
        
        y = valid['textile_firms']
        X = sm.add_constant(valid[['policy_intensity_total']])
        m = sm.OLS(y, X).fit()
        
        beta = m.params['policy_intensity_total']
        p = m.pvalues['policy_intensity_total']
        r2 = m.rsquared
        
        results[name] = {'beta': beta, 'p': p, 'r2': r2, 'n': len(valid)}
        print(f"  {name:<15} {start}-{end:>4} {len(valid):>4} {beta:>10.4f} {p:>10.4f} {r2:>8.4f} {sig_star(p)}")
    
    # 阶段间系数差异
    print(f"\n  【阶段间弹性比较】")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        items = list(valid_results.items())
        for i in range(len(items)-1):
            n1, r1 = items[i]
            n2, r2 = items[i+1]
            beta_diff = r2['beta'] - r1['beta']
            print(f"  {n1} -> {n2}: beta变化 = {beta_diff:+.4f}")
            if r1['beta'] > 0 and r2['beta'] > 0 and r2['beta'] > r1['beta']:
                print(f"    政策弹性递增（越到后期效果越强）")
            elif r1['beta'] > 0 and r2['beta'] > 0 and r2['beta'] < r1['beta']:
                print(f"    政策弹性递减（边际效应递减）")

# ============================================================
# 方向7：工具变量法改进（Lewbel异方差IV）
# ============================================================
def analysis7_lewbel_iv(gy):
    print("\n" + "="*80)
    print("探究7：Lewbel(2012)异方差工具变量法")
    print("不需要外生IV，利用模型异方差性构造IV")
    print("="*80)
    
    valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total'])
    if len(valid) < 10:
        print(f"  样本不足({len(valid)}), 跳过")
        return
    
    # 第一阶段回归
    y = valid['textile_firms']
    X = sm.add_constant(valid[['policy_intensity_total']])
    stage1 = sm.OLS(y, X).fit()
    
    # 获取残差和中心化解释变量
    residuals = stage1.resid
    policy_centered = valid['policy_intensity_total'].values - valid['policy_intensity_total'].mean()
    
    # 构造Lewbel IV：残差 x 中心化解释变量
    lewbel_iv = residuals * policy_centered
    
    # 第二阶段：用Lewbel IV作为工具变量
    X_iv = sm.add_constant(pd.DataFrame({'lewbel_iv': lewbel_iv}))
    stage2 = sm.OLS(y, X_iv).fit()
    
    beta_ols = stage1.params['policy_intensity_total']
    p_ols = stage1.pvalues['policy_intensity_total']
    
    beta_iv = stage2.params['lewbel_iv']
    p_iv = stage2.pvalues['lewbel_iv']
    
    print(f"\n  OLS:  beta={beta_ols:.4f}, p={p_ols:.4f} {sig_star(p_ols)}")
    print(f"  IV:   beta={beta_iv:.4f}, p={p_iv:.4f} {sig_star(p_iv)}")
    
    # 异方差检验（Breusch-Pagan简化版）
    print(f"\n  【异方差检验】")
    bp_stat = np.sum(lewbel_iv**2) / (2 * len(valid))
    bp_p = 1 - sp_stats.chi2.cdf(bp_stat, 1)
    print(f"  BP统计量 = {bp_stat:.4f}, p={bp_p:.4f}")
    
    if bp_p < 0.1:
        print(f"  存在异方差，Lewbel IV适用")
    else:
        print(f"  不存在显著异方差，Lewbel IV效果可能有限")

# ============================================================
# 方向8：贝叶斯回归
# ============================================================
def analysis8_bayesian_regression(gy):
    print("\n" + "="*80)
    print("探究8：贝叶斯回归（小样本下的更优推断）")
    print("="*80)
    
    valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total'])
    if len(valid) < 10:
        print(f"  样本不足, 跳过")
        return
    
    y = valid['textile_firms']
    X_df = sm.add_constant(valid[['policy_intensity_total']])
    
    # 先验：beta ~ N(0, 100), sigma2 ~ Inv-Gamma(2, 1)
    prior_beta_mean = np.array([0, 0])
    prior_beta_var = np.eye(2) * 100
    prior_nu = 2
    prior_ss = 1
    
    # 后验计算
    y_arr = y.values
    X_arr = X_df.values
    n = len(y_arr)
    k = X_arr.shape[1]
    
    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr
    beta_ols = np.linalg.solve(XtX, Xty)
    ssr = np.sum((y_arr - X_arr @ beta_ols)**2)
    
    # 后验参数
    post_beta_var = np.linalg.inv(np.linalg.inv(prior_beta_var) + XtX)
    post_beta_mean = post_beta_var @ (np.linalg.inv(prior_beta_var) @ prior_beta_mean + Xty)
    post_nu = prior_nu + n
    post_ss = prior_ss + ssr + (beta_ols - prior_beta_mean).T @ np.linalg.inv(prior_beta_var + np.linalg.inv(XtX)) @ (beta_ols - prior_beta_mean)
    
    # 后验均值和标准差
    beta_post_mean = post_beta_mean[1]
    beta_post_var = post_beta_var[1, 1]
    beta_post_sd = np.sqrt(beta_post_var)
    
    # 后验概率 P(beta > 0 | data)
    p_positive = 1 - sp_stats.t.cdf(0, post_nu, loc=beta_post_mean, scale=beta_post_sd)
    
    # 95%可信区间
    ci_lower = sp_stats.t.ppf(0.025, post_nu, loc=beta_post_mean, scale=beta_post_sd)
    ci_upper = sp_stats.t.ppf(0.975, post_nu, loc=beta_post_mean, scale=beta_post_sd)
    
    print(f"\n  【后验分布】policy_intensity_total")
    print(f"  后验均值: {beta_post_mean:.4f}")
    print(f"  后验标准差: {beta_post_sd:.4f}")
    print(f"  95%可信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  P(beta > 0 | data) = {p_positive:.4f}")
    
    if p_positive > 0.95:
        print(f"  [结论] 贝叶斯结论：政策效应为正的可信度 > 95%")
    elif p_positive > 0.90:
        print(f"  [结论] 贝叶斯结论：政策效应为正的可信度 > 90%（边缘显著）")
    else:
        print(f"  [结论] 贝叶斯结论：政策效应不显著")
    
    # 对比OLS
    m_ols = sm.OLS(y, X_df).fit()
    print(f"\n  【对比】OLS: beta={m_ols.params['policy_intensity_total']:.4f}, p={m_ols.pvalues['policy_intensity_total']:.4f}")
    print(f"         贝叶斯: P(beta>0|data)={p_positive:.4f}")

# ============================================================
# 方向9：机制的机制
# ============================================================
def analysis9_mechanism_of_mechanism(gy):
    print("\n" + "="*80)
    print("探究9：机制的机制（深层机制路径）")
    print("="*80)
    
    # 9.1 环保政策 -> 企业退出 -> 产业升级
    print(f"\n【9.1】环保规制 -> 产业升级路径")
    
    if 'above_scale_ind_firms' in gy.columns:
        valid = gy.copy()
        valid['textile_ratio'] = valid['textile_firms'] / valid['total_firms']
        
        valid_drop = valid.dropna(subset=['environment_index', 'textile_ratio'])
        if len(valid_drop) >= 5:
            y = valid_drop['textile_ratio']
            X = sm.add_constant(valid_drop[['environment_index']])
            m = sm.OLS(y, X).fit()
            
            beta = m.params['environment_index']
            p = m.pvalues['environment_index']
            print(f"  environment_index -> textile_ratio: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")
            
            if p < 0.1:
                direction = "提升" if beta > 0 else "降低"
                print(f"  [{direction}] 环保政策{direction}了产业专业化程度")
        
        # 环保政策对规模以上企业的影响
        valid_drop2 = valid.dropna(subset=['environment_index', 'above_scale_ind_firms'])
        if len(valid_drop2) >= 5:
            y = valid_drop2['above_scale_ind_firms']
            X = sm.add_constant(valid_drop2[['environment_index']])
            m2 = sm.OLS(y, X).fit()
            
            beta2 = m2.params['environment_index']
            p2 = m2.pvalues['environment_index']
            print(f"  environment_index -> above_scale_firms: beta={beta2:.4f}, p={p2:.4f} {sig_star(p2)}")
    
    # 9.2 设备政策 -> 产能扩张 -> 规模经济
    print(f"\n【9.2】设备政策 -> 产能扩张路径")
    if 'above_scale_industry_output' in gy.columns:
        valid = gy.dropna(subset=['equipment_index', 'above_scale_industry_output'])
        if len(valid) >= 5:
            y = valid['above_scale_industry_output']
            X = sm.add_constant(valid[['equipment_index']])
            m = sm.OLS(y, X).fit()
            
            beta = m.params['equipment_index']
            p = m.pvalues['equipment_index']
            print(f"  equipment_index -> industry_output: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")
    
    # 9.3 电商政策 -> 市场扩大 -> 新企业进入
    print(f"\n【9.3】电商政策 -> 新企业进入路径")
    valid = gy.dropna(subset=['ecommerce_index', 'new_textile_firms'])
    if len(valid) >= 5:
        y = valid['new_textile_firms']
        X = sm.add_constant(valid[['ecommerce_index']])
        m = sm.OLS(y, X).fit()
        
        beta = m.params['ecommerce_index']
        p = m.pvalues['ecommerce_index']
        print(f"  ecommerce_index -> new_textile_firms: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")

# ============================================================
# 主函数
# ============================================================
def run_comprehensive_analysis():
    print("\n" + "="*80)
    print("综合分析 - 9个探究方向全量执行")
    print("="*80)
    
    gy, policy_df, ent = load_data()
    
    # 方向1：纺织指数分析
    analysis1_textile_indices(gy)
    
    # 方向2：政策协同效应
    analysis2_policy_synergy(gy)
    
    # 方向3：空间溢出效应
    analysis3_spatial_spillover(policy_df)
    
    # 方向4：企业生命周期
    analysis4_enterprise_lifecycle(gy)
    
    # 方向5：宏观控制变量
    analysis5_macro_controls(gy)
    
    # 方向6：分阶段分析
    analysis6_phase_analysis(gy)
    
    # 方向7：Lewbel IV
    analysis7_lewbel_iv(gy)
    
    # 方向8：贝叶斯回归
    analysis8_bayesian_regression(gy)
    
    # 方向9：机制的机制
    analysis9_mechanism_of_mechanism(gy)
    
    print("\n" + "="*80)
    print("综合分析 - 全部完成")
    print("="*80)

if __name__ == "__main__":
    run_comprehensive_analysis()
