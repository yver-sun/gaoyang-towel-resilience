"""
机制检验完整脚本
对4个机制进行逐一深入探究
1. 政策时滞传导机制
2. 政策工具有效性层级
3. 环保规制创造性破坏
4. 政策过载倒U型效应
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(f"  [INFO] {msg}")

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载并整合数据"""
    policy = pd.read_csv('output/policy_scores_panel.csv')
    policy['County_Code'] = policy['County_Code'].astype(str)
    
    enterprise = pd.read_csv('output/gaoyang_enterprise_registration.csv')
    enterprise['County_Code'] = '130628'
    
    gy_policy = policy[policy['County_Code'] == '130628'][[
        'Year', 'policy_intensity_total', 'policy_intensity_production', 
        'policy_intensity_market', 'policy_mix_equipment_env',
        'environment_index', 'equipment_index', 'ecommerce_index',
        'brandquality_index', 'cluster_index', 'finance_index', 'education_index',
        'sample_reliable', 'scored', 'total_chunks'
    ]].copy()
    
    gy_ent = enterprise[['Year', 'total_firms', 'textile_firms', 
                         'textile_ratio', 'new_textile_firms',
                         'new_total_firms']].copy()
    
    gy = gy_policy.merge(gy_ent, on='Year', how='left')
    gy = gy.sort_values('Year').reset_index(drop=True)
    
    log(f"数据加载完成: {len(gy)}年 ({int(gy.Year.min())}-{int(gy.Year.max())})")
    log(f"企业数据可用: {gy.textile_firms.notna().sum()}年")
    
    return gy

# ============================================================
# 机制1：政策时滞传导机制
# ============================================================
def mechanism1_lag_transmission(gy):
    """
    机制1：政策时滞传导机制
    假说：政策效果通过"预期形成->投资决策->企业进入"传导，在滞后1期达到峰值
    """
    print("\n" + "="*80)
    print("机制1：政策时滞传导机制 (Policy Lag Transmission)")
    print("="*80)
    
    data = gy.dropna(subset=['textile_firms', 'policy_intensity_total']).copy()
    
    # 生成多期滞后
    for lag in range(0, 5):
        data[f'policy_L{lag}'] = data['policy_intensity_total'].shift(lag)
    
    data = data.dropna(subset=['policy_L0', 'policy_L1', 'policy_L2', 'policy_L3', 'policy_L4'])
    
    log(f"有效样本: {len(data)}年 ({int(data.Year.min())}-{int(data.Year.max())})")
    
    # 模型1：当期+滞后1~4期
    print("\n【模型1】多期滞后分布滞后模型 (Distributed Lag Model)")
    print(f"  textile_firms = beta0 + sum(beta_k * policy_t-k) + epsilon")
    
    lag_vars = [f'policy_L{i}' for i in range(5)]
    y = data['textile_firms']
    X = sm.add_constant(data[lag_vars])
    model = sm.OLS(y, X).fit()
    
    print(f"\n  {'变量':<20} {'系数':>10} {'标准误':>10} {'t值':>8} {'p值':>10} {'显著性':>6}")
    print(f"  {'-'*70}")
    
    lag_betas = []
    for var in lag_vars:
        beta = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<20} {beta:>10.4f} {se:>10.4f} {model.tvalues[var]:>8.2f} {p:>10.4f} {sig:>6}")
        lag_betas.append(beta)
    
    print(f"\n  R2 = {model.rsquared:.4f}, F = {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # 计算累积效应
    cumulative = np.cumsum(lag_betas)
    print(f"\n  【累积效应分析】")
    print(f"  {'滞后阶数':<10} {'当期效应':>12} {'累积效应':>12}")
    print(f"  {'-'*35}")
    for i, (beta, cum) in enumerate(zip(lag_betas, cumulative)):
        print(f"  L{i:<9} {beta:>12.4f} {cum:>12.4f}")
    
    # 检验1：滞后1期是否显著大于其他期
    print(f"\n  【机制检验1】滞后1期效应是否最大？")
    peak_lag = np.argmax(np.abs(lag_betas))
    peak_beta = lag_betas[peak_lag]
    print(f"  效应峰值在: L{peak_lag} (beta={peak_beta:.4f})")
    
    if peak_lag == 1:
        print(f"  [验证通过] 政策效果确实在滞后1期达到峰值")
        print(f"  机制解释: 政策发布(t)->投资决策(t)->企业注册(t+1)")
    else:
        print(f"  [部分验证] 效应峰值在L{peak_lag}，不完全符合假说")
    
    # 检验2：联合检验（政策是否有总体效应）
    print(f"\n  【机制检验2】政策总效应联合检验")
    total_effect = sum(lag_betas)
    print(f"  政策总效应 (sum of all lags) = {total_effect:.4f}")
    
    # 模型2：仅保留显著滞后项
    print(f"\n【模型2】精简模型（仅保留L0/L1）")
    y = data['textile_firms']
    X = sm.add_constant(data[['policy_L0', 'policy_L1']])
    model2 = sm.OLS(y, X).fit()
    
    print(f"  {'变量':<20} {'系数':>10} {'p值':>10} {'显著性':>6}")
    for var in ['policy_L0', 'policy_L1']:
        beta = model2.params[var]
        p = model2.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<20} {beta:>10.4f} {p:>10.4f} {sig:>6}")
    
    print(f"  R2 = {model2.rsquared:.4f}")
    
    return model, model2

# ============================================================
# 机制2：政策工具有效性层级
# ============================================================
def mechanism2_policy_tool_hierarchy(gy):
    """
    机制2：政策工具有效性层级
    假说：供给侧政策 > 规制型政策 > 需求侧政策
    """
    print("\n" + "="*80)
    print("机制2：政策工具有效性层级 (Policy Tool Hierarchy)")
    print("="*80)
    
    data = gy.dropna(subset=['textile_firms']).copy()
    
    # 政策工具分类
    supply_tools = ['equipment_index']      # 供给侧：设备升级
    regulation_tools = ['environment_index']  # 规制型：环保治理
    demand_tools = ['ecommerce_index', 'finance_index']  # 需求侧：电商/金融
    cluster_tools = ['cluster_index', 'brandquality_index']  # 集群型：产业集群/品牌
    
    print(f"\n【模型1】各政策维度单独回归")
    print(f"  因变量: textile_firms")
    
    all_results = {}
    for category, dims in [('供给侧', supply_tools), ('规制型', regulation_tools),
                           ('需求侧', demand_tools), ('集群型', cluster_tools)]:
        for dim in dims:
            valid_data = data.dropna(subset=[dim])
            if len(valid_data) < 8:
                continue
                
            y = valid_data['textile_firms']
            X = sm.add_constant(valid_data[[dim]])
            model = sm.OLS(y, X).fit()
            
            beta = model.params[dim]
            p = model.pvalues[dim]
            r2 = model.rsquared
            sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else 'ns'
            
            print(f"  {category}-{dim:<20} beta={beta:>8.3f}  p={p:.4f}  R2={r2:.3f}  {sig}")
            all_results[dim] = {'beta': beta, 'p': p, 'r2': r2, 'category': category}
    
    # 模型2：多变量回归（检验相对重要性）
    print(f"\n【模型2】多变量回归（所有维度同时进入）")
    
    all_dims = supply_tools + regulation_tools + demand_tools + cluster_tools
    valid_data = data.dropna(subset=all_dims)
    
    if len(valid_data) > len(all_dims) + 1:
        y = valid_data['textile_firms']
        X = sm.add_constant(valid_data[all_dims])
        model_mv = sm.OLS(y, X).fit()
        
        print(f"  {'维度':<20} {'系数':>10} {'p值':>10} {'显著性':>6}")
        for dim in all_dims:
            beta = model_mv.params[dim]
            p = model_mv.pvalues[dim]
            sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else 'ns'
            print(f"  {dim:<20} {beta:>10.3f} {p:>10.4f} {sig:>6}")
        
        print(f"  R2 = {model_mv.rsquared:.4f}")
    else:
        log("样本量不足，跳过多元回归")
    
    # 层级排序
    print(f"\n【机制检验】政策工具有效性层级排序")
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['p'])
    
    print(f"  {'排名':<5} {'维度':<20} {'类别':<8} {'beta':>8} {'p值':>8}")
    print(f"  {'-'*55}")
    for rank, (dim, res) in enumerate(sorted_results, 1):
        print(f"  {rank:<5} {dim:<20} {res['category']:<8} {res['beta']:>8.3f} {res['p']:>8.4f}")
    
    # 检验：供给侧是否显著优于其他类型
    supply_beta = all_results.get('equipment_index', {}).get('beta', 0)
    reg_beta = all_results.get('environment_index', {}).get('beta', 0)
    demand_betas = [all_results.get(d, {}).get('beta', 0) for d in demand_tools if d in all_results]
    
    print(f"\n  【层级验证】")
    print(f"  供给侧(Equipment) beta = {supply_beta:.3f}")
    print(f"  规制型(Environment) beta = {reg_beta:.3f}")
    print(f"  需求侧平均 beta = {np.mean(demand_betas):.3f}")
    
    if supply_beta > reg_beta > np.mean(demand_betas):
        print(f"  [验证通过] 供给侧 > 规制型 > 需求侧")
    else:
        print(f"  [部分验证] 层级关系不完全成立")
    
    return all_results

# ============================================================
# 机制3：环保规制创造性破坏
# ============================================================
def mechanism3_creative_destruction(gy):
    """
    机制3：环保规制创造性破坏
    假说：2017环保督察淘汰落后产能（破坏效应），同时提高进入门槛（创造效应）
    """
    print("\n" + "="*80)
    print("机制3：环保规制创造性破坏 (Creative Destruction)")
    print("="*80)
    
    data = gy.dropna(subset=['textile_firms']).copy()
    
    # 3.1 结构性断点检验
    print(f"\n【检验1】结构性断点检验 (Chow Test)")
    
    data['post_2017'] = (data['Year'] >= 2017).astype(int)
    data['time'] = data['Year'] - data['Year'].min()
    
    # 全样本
    y = data['textile_firms']
    X = sm.add_constant(data[['time']])
    model_full = sm.OLS(y, X).fit()
    ssr_full = model_full.ssr
    
    # 2017前
    pre = data[data['post_2017'] == 0]
    if len(pre) > 3:
        y_pre = pre['textile_firms']
        X_pre = sm.add_constant(pre[['time']])
        model_pre = sm.OLS(y_pre, X_pre).fit()
        ssr_pre = model_pre.ssr
        n_pre = len(pre)
    else:
        ssr_pre = 0
        n_pre = 0
    
    # 2017后
    post = data[data['post_2017'] == 1]
    if len(post) > 3:
        y_post = post['textile_firms']
        X_post = sm.add_constant(post[['time']])
        model_post = sm.OLS(y_post, X_post).fit()
        ssr_post = model_post.ssr
        n_post = len(post)
    else:
        ssr_post = 0
        n_post = 0
    
    n_total = n_pre + n_post
    k = 2  # 参数个数
    
    if ssr_pre > 0 and ssr_post > 0:
        f_stat = ((ssr_full - (ssr_pre + ssr_post)) / k) / ((ssr_pre + ssr_post) / (n_total - 2*k))
        f_pval = 1 - sp_stats.f.cdf(f_stat, k, n_total - 2*k)
        
        print(f"  SSR_full = {ssr_full:.2f}")
        print(f"  SSR_pre (2017前) = {ssr_pre:.2f} (n={n_pre})")
        print(f"  SSR_post (2017后) = {ssr_post:.2f} (n={n_post})")
        print(f"  Chow F = {f_stat:.2f} (p={f_pval:.4f})")
        
        if f_pval < 0.1:
            print(f"  [验证通过] 2017年存在显著结构性断点")
        else:
            print(f"  [未验证] 未发现显著结构性断点")
    else:
        print(f"  样本不足，跳过Chow检验")
    
    # 3.2 事件研究：分析2017前后企业数量变化
    print(f"\n【检验2】事件研究：2017环保督察前后企业数量变化")
    
    # 计算年度新增
    data['net_entry'] = data['textile_firms'].diff()
    
    print(f"\n  年份    企业总数   净进入数    政策强度")
    print(f"  {'-'*50}")
    for _, row in data[(data.Year >= 2014) & (data.Year <= 2020)].iterrows():
        net = row['net_entry'] if not pd.isna(row['net_entry']) else 0
        print(f"  {int(row.Year)}    {int(row.textile_firms):>6}    {net:>6.0f}    {row.policy_intensity_total:>6.2f}")
    
    # 3.3 2017前后对比
    pre_2017 = data[(data.Year >= 2010) & (data.Year < 2017)]['net_entry'].dropna()
    post_2017 = data[(data.Year >= 2017) & (data.Year <= 2022)]['net_entry'].dropna()
    
    if len(pre_2017) > 0 and len(post_2017) > 0:
        mean_pre = pre_2017.mean()
        mean_post = post_2017.mean()
        
        print(f"\n  2017前 (2010-2016) 平均净进入: {mean_pre:.1f} 家/年")
        print(f"  2017后 (2017-2022) 平均净进入: {mean_post:.1f} 家/年")
        print(f"  变化: {mean_post - mean_pre:+.1f} 家/年")
        
        # t检验
        t_stat, t_pval = sp_stats.ttest_ind(post_2017, pre_2017)
        print(f"  t检验: t={t_stat:.2f} (p={t_pval:.4f})")
        
        if mean_post < mean_pre:
            print(f"  [机制证据] 2017后净进入数下降，支持创造性破坏假说")
        else:
            print(f"  [不支持] 2017后净进入数未下降")
    
    # 3.4 企业进入 vs 退出分析
    print(f"\n【检验3】企业进入 vs 退出分解")
    
    if 'new_textile_firms' in data.columns:
        data['exit_firms'] = data['textile_firms'].shift(1) + data['new_textile_firms'] - data['textile_firms']
        
        print(f"\n  年份    年初存量    新进入    退出    年末存量")
        print(f"  {'-'*55}")
        for _, row in data[(data.Year >= 2015) & (data.Year <= 2020)].iterrows():
            start = row['textile_firms'] if not pd.isna(row['textile_firms']) else 0
            new = row['new_textile_firms'] if not pd.isna(row['new_textile_firms']) else 0
            exit_n = row['exit_firms'] if not pd.isna(row['exit_firms']) else 0
            end = row['textile_firms'] if not pd.isna(row['textile_firms']) else 0
            print(f"  {int(row.Year)}    {start:>6}    {new:>6}    {exit_n:>6.0f}    {end:>6}")
    
    return {}

# ============================================================
# 机制4：政策过载倒U型效应
# ============================================================
def mechanism4_overload_effect(gy):
    """
    机制4：政策过载倒U型效应
    假说：政策强度超过阈值后，合规成本过高导致效应递减
    """
    print("\n" + "="*80)
    print("机制4：政策过载倒U型效应 (Policy Overload Effect)")
    print("="*80)
    
    data = gy.dropna(subset=['textile_firms', 'policy_intensity_total']).copy()
    
    # 4.1 二次项检验
    print(f"\n【检验1】二次项回归")
    
    data['policy_sq'] = data['policy_intensity_total'] ** 2
    
    y = data['textile_firms']
    X = sm.add_constant(data[['policy_intensity_total', 'policy_sq']])
    model = sm.OLS(y, X).fit()
    
    beta1 = model.params['policy_intensity_total']
    beta2 = model.params['policy_sq']
    p1 = model.pvalues['policy_intensity_total']
    p2 = model.pvalues['policy_sq']
    
    print(f"  一次项: beta={beta1:.4f} (p={p1:.4f})")
    print(f"  二次项: beta={beta2:.4f} (p={p2:.4f})")
    
    if beta2 < 0:
        turning_point = -beta1 / (2 * beta2)
        print(f"\n  二次项系数为负 -> 倒U型关系")
        print(f"  拐点: policy = {turning_point:.2f}")
        print(f"  政策强度分布: min={data.policy_intensity_total.min():.2f}, max={data.policy_intensity_total.max():.2f}")
        
        # 计算有多少样本在拐点右侧
        right_of_turning = (data['policy_intensity_total'] > turning_point).sum()
        print(f"  样本在拐点右侧: {right_of_turning}/{len(data)} ({right_of_turning/len(data)*100:.1f}%)")
        
        if p2 < 0.1:
            print(f"  [验证通过] 二次项显著为负，存在倒U型关系")
        else:
            print(f"  [证据较弱] 二次项不显著 (p={p2:.3f})，但方向符合倒U型")
    else:
        print(f"\n  二次项系数为正 -> U型关系（非倒U型）")
        print(f"  [不支持] 不支持倒U型假说")
    
    # 4.2 分段线性回归
    print(f"\n【检验2】分段线性回归")
    
    # 尝试不同阈值
    thresholds = [50, 80, 100, 120, 150]
    print(f"  {'阈值':<8} {'左侧beta':>10} {'右侧beta':>10} {'R2':>8}")
    print(f"  {'-'*45}")
    
    for threshold in thresholds:
        data['policy_below'] = np.minimum(data['policy_intensity_total'], threshold)
        data['policy_above'] = np.maximum(0, data['policy_intensity_total'] - threshold)
        
        y = data['textile_firms']
        X = sm.add_constant(data[['policy_below', 'policy_above']])
        m = sm.OLS(y, X).fit()
        
        b_below = m.params['policy_below']
        b_above = m.params['policy_above']
        print(f"  {threshold:<8} {b_below:>10.4f} {b_above:>10.4f} {m.rsquared:>8.4f}")
    
    # 4.3 经济解释
    print(f"\n  【机制解释】")
    print(f"  政策过载可能的原因:")
    print(f"  1. 合规成本过高：企业为满足环保/设备标准，投资成本增加")
    print(f"  2. 政策不确定性：高强度政策可能暗示未来更严规制")
    print(f"  3. 资源错配：过度补贴可能导致低效率企业进入")
    
    return model

# ============================================================
# 主函数
# ============================================================
def run_mechanism_analysis():
    print("\n" + "="*80)
    print("机制检验完整分析")
    print("="*80)
    
    gy = load_data()
    
    # 机制1：政策时滞传导
    mechanism1_lag_transmission(gy)
    
    # 机制2：政策工具有效性层级
    mechanism2_policy_tool_hierarchy(gy)
    
    # 机制3：环保规制创造性破坏
    mechanism3_creative_destruction(gy)
    
    # 机制4：政策过载倒U型效应
    mechanism4_overload_effect(gy)
    
    print("\n" + "="*80)
    print("机制检验 - 全部完成")
    print("="*80)

if __name__ == "__main__":
    run_mechanism_analysis()
