"""
稳健性改进脚本 - 弥补数据缺陷
1. Bootstrap重采样（解决小样本n=25问题）
2. 零膨胀模型（解决66.7%零值问题）
3. 合成控制法（单案例因果推断）
4. 事件研究法（多政策冲击）
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
    
    return gy, policy, ent

# ============================================================
# 改进1：Bootstrap重采样分析
# ============================================================
def improvement1_bootstrap(gy, n_bootstrap=10000):
    """
    Bootstrap重采样
    目的：在小样本(n=25)下获得更稳健的置信区间和p值
    方法：有放回抽样10000次，计算经验分布
    """
    print("\n" + "="*80)
    print("改进1：Bootstrap重采样分析")
    print(f"目标：解决n=25小样本下标准误不可靠问题")
    print(f"方法：有放回抽样{n_bootstrap}次")
    print("="*80)
    
    valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total'])
    if len(valid) < 10:
        print(f"  样本不足({len(valid)})")
        return
    
    y = valid['textile_firms']
    X = sm.add_constant(valid[['policy_intensity_total']])
    
    # 原始OLS
    model_orig = sm.OLS(y, X).fit()
    beta_orig = model_orig.params['policy_intensity_total']
    se_orig = model_orig.bse['policy_intensity_total']
    p_orig = model_orig.pvalues['policy_intensity_total']
    
    print(f"\n【原始OLS结果】")
    print(f"  beta = {beta_orig:.4f}, SE = {se_orig:.4f}, p = {p_orig:.4f} {sig_star(p_orig)}")
    print(f"  95% CI = [{beta_orig - 1.96*se_orig:.4f}, {beta_orig + 1.96*se_orig:.4f}]")
    
    # Bootstrap重采样
    print(f"\n【Bootstrap重采样】({n_bootstrap}次)")
    np.random.seed(42)
    n = len(y)
    y_arr = y.values
    X_arr = X.values
    beta_boot = []
    se_boot = []
    
    for i in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(n, size=n, replace=True)
        y_boot = y_arr[indices]
        X_boot = X_arr[indices]
        
        # 检查X是否有变异（避免共线性问题）
        if X_boot[:, 1].std() < 1e-10:
            continue
        
        try:
            m = sm.OLS(y_boot, X_boot).fit()
            beta_boot.append(m.params[1])  # 用整数索引
            se_boot.append(m.bse[1])
        except:
            continue
    
    beta_boot = np.array(beta_boot)
    se_boot = np.array(se_boot)
    
    # 检查是否成功
    if len(beta_boot) < 100:
        print(f"  Bootstrap有效样本太少({len(beta_boot)})，可能存在数据问题")
        return
    
    beta_boot = np.array(beta_boot)
    se_boot = np.array(se_boot)
    
    # 计算Bootstrap统计量
    beta_mean = beta_boot.mean()
    beta_sd = beta_boot.std()
    beta_median = np.median(beta_boot)
    
    # 百分位数法置信区间
    ci_lower_25 = np.percentile(beta_boot, 2.5)
    ci_upper_25 = np.percentile(beta_boot, 97.5)
    
    # BCa法（偏差校正）
    z0 = sp_stats.norm.ppf(np.mean(beta_boot < beta_orig))
    ci_lower_bca = np.percentile(beta_boot, 100 * sp_stats.norm.cdf(2*z0 - 1.96))
    ci_upper_bca = np.percentile(beta_boot, 100 * sp_stats.norm.cdf(2*z0 + 1.96))
    
    # Bootstrap p值（单侧：P(beta < 0)）
    p_bootstrap = np.mean(beta_boot < 0)
    p_bootstrap_two = 2 * min(p_bootstrap, 1 - p_bootstrap)
    
    print(f"\n  Bootstrap分布统计:")
    print(f"    均值: {beta_mean:.4f}")
    print(f"    中位数: {beta_median:.4f}")
    print(f"    标准差: {beta_sd:.4f}")
    print(f"    偏度: {sp_stats.skew(beta_boot):.4f}")
    print(f"    峰度: {sp_stats.kurtosis(beta_boot):.4f}")
    
    print(f"\n  95%置信区间:")
    print(f"    原始OLS: [{beta_orig - 1.96*se_orig:.4f}, {beta_orig + 1.96*se_orig:.4f}]")
    print(f"    Bootstrap百分位法: [{ci_lower_25:.4f}, {ci_upper_25:.4f}]")
    print(f"    Bootstrap BCa法: [{ci_lower_bca:.4f}, {ci_upper_bca:.4f}]")
    
    print(f"\n  p值比较:")
    print(f"    原始OLS: {p_orig:.4f}")
    print(f"    Bootstrap单侧: {p_bootstrap:.4f}")
    print(f"    Bootstrap双侧: {p_bootstrap_two:.4f}")
    
    # 结论
    if p_bootstrap_two < 0.05:
        print(f"\n  [结论] Bootstrap验证：政策效应在5%水平显著，结果稳健")
        print(f"         {np.sum(beta_boot > 0)/len(beta_boot)*100:.1f}%的Bootstrap样本beta>0")
    elif p_bootstrap_two < 0.1:
        print(f"\n  [结论] Bootstrap验证：政策效应在10%水平边缘显著")
        print(f"         {np.sum(beta_boot > 0)/len(beta_boot)*100:.1f}%的Bootstrap样本beta>0")
    else:
        print(f"\n  [结论] Bootstrap验证：政策效应不显著，原始结果可能不稳定")
    
    # 可视化数据（供后续绘图使用）
    print(f"\n  Bootstrap分布数据已保存（可用于绘制直方图）")
    print(f"    存储: beta_boot.npy ({len(beta_boot)}个值)")
    np.save('output/bootstrap_beta_values.npy', beta_boot)
    
    return {
        'beta_orig': beta_orig,
        'se_orig': se_orig,
        'p_orig': p_orig,
        'beta_boot_mean': beta_mean,
        'beta_boot_sd': beta_sd,
        'ci_lower_pct': ci_lower_25,
        'ci_upper_pct': ci_upper_25,
        'ci_lower_bca': ci_lower_bca,
        'ci_upper_bca': ci_upper_bca,
        'p_bootstrap': p_bootstrap_two,
        'n_boot': len(beta_boot)
    }

# ============================================================
# 改进2：零膨胀模型（Zero-Inflated Model）
# ============================================================
def improvement2_zero_inflated(gy):
    """
    零膨胀模型
    目的：解决66.7%政策强度为0的问题
    方法1：Hurdle Model（两阶段）
      - 阶段1：Logit（是否有政策）
      - 阶段2：OLS（政策强度>0时，对企业数量的影响）
    方法2：Zero-Inflated Poisson (ZIP) 简化版
    """
    print("\n" + "="*80)
    print("改进2：零膨胀模型（Zero-Inflated / Hurdle Model）")
    print("目标：解决66.7%政策强度为零的问题")
    print("="*80)
    
    valid = gy.dropna(subset=['textile_firms', 'policy_intensity_total'])
    n_total = len(valid)
    n_zero = (valid['policy_intensity_total'] == 0).sum()
    n_positive = n_total - n_zero
    
    print(f"\n【数据概况】")
    print(f"  总样本: {n_total}")
    print(f"  零值: {n_zero} ({n_zero/n_total*100:.1f}%)")
    print(f"  正值: {n_positive} ({n_positive/n_total*100:.1f}%)")
    
    if n_positive < 8:
        print(f"  正值样本不足({n_positive})，无法进行两阶段回归")
        return
    
    # 方法1：Hurdle Model
    print(f"\n【方法1】Hurdle Model（两阶段模型）")
    
    # 阶段1：Logit（是否有政策）
    print(f"\n  阶段1：Logit回归（政策是否>0）")
    valid = valid.copy()
    valid['has_policy'] = (valid['policy_intensity_total'] > 0).astype(int)
    
    # 用时间趋势作为解释变量
    valid['time'] = valid['Year'] - valid['Year'].min()
    
    try:
        logit_model = sm.Logit(valid['has_policy'], sm.add_constant(valid[['time']])).fit(disp=0)
        
        print(f"    {'变量':<15} {'系数':>10} {'p值':>10} {'显著'}")
        for var in ['const', 'time']:
            beta = logit_model.params[var]
            p = logit_model.pvalues[var]
            print(f"    {var:<15} {beta:>10.4f} {p:>10.4f} {sig_star(p)}")
        
        # 预测概率
        valid['prob_policy'] = logit_model.predict()
        print(f"    平均预测概率: {valid['prob_policy'].mean():.3f}")
        accuracy = ((valid['prob_policy'] > 0.5).astype(int) == valid['has_policy']).mean() * 100
        print(f"    模型准确率: {accuracy:.1f}%")
    except Exception as e:
        print(f"    Logit回归失败: {e}")
        valid['prob_policy'] = valid['has_policy']
    
    # 阶段2：OLS（仅正值样本）
    print(f"\n  阶段2：OLS回归（仅政策强度>0的样本）")
    valid_positive = valid[valid['policy_intensity_total'] > 0]
    
    if len(valid_positive) >= 5:
        y = valid_positive['textile_firms']
        X = sm.add_constant(valid_positive[['policy_intensity_total']])
        m_ols = sm.OLS(y, X).fit()
        
        beta = m_ols.params['policy_intensity_total']
        p = m_ols.pvalues['policy_intensity_total']
        r2 = m_ols.rsquared
        
        print(f"    policy_intensity: beta={beta:.4f}, p={p:.4f} {sig_star(p)}")
        print(f"    R2 = {r2:.4f}, n = {len(valid_positive)}")
        
        # 与全样本OLS比较
        m_full = sm.OLS(gy.dropna(subset=['textile_firms', 'policy_intensity_total'])['textile_firms'],
                        sm.add_constant(gy.dropna(subset=['textile_firms', 'policy_intensity_total'])[['policy_intensity_total']])).fit()
        beta_full = m_full.params['policy_intensity_total']
        p_full = m_full.pvalues['policy_intensity_total']
        
        print(f"\n  【比较】")
        print(f"    全样本OLS (含零值): beta={beta_full:.4f}, p={p_full:.4f}, n={len(m_full.resid)}")
        print(f"    Hurdle阶段2 (仅正值): beta={beta:.4f}, p={p:.4f}, n={len(valid_positive)}")
        
        beta_diff = abs(beta - beta_full) / abs(beta_full) * 100
        print(f"    系数差异: {beta_diff:.1f}%")
        
        if p < 0.1:
            print(f"  [结论] 剔除零值后，政策效应仍然显著，说明结果不是由零值驱动的")
        else:
            print(f"  [结论] 剔除零值后效应减弱，说明零值样本对结果有影响")
    
    # 方法2：分样本回归（零值 vs 非零值）
    print(f"\n【方法2】分样本回归对比")
    
    # 零值样本
    valid_zero = valid[valid['policy_intensity_total'] == 0]
    if len(valid_zero) >= 5:
        y_zero = valid_zero['textile_firms']
        print(f"  零值样本 (n={len(valid_zero)}):")
        print(f"    企业数均值: {y_zero.mean():.1f}")
        print(f"    企业数标准差: {y_zero.std():.1f}")
        print(f"    企业数范围: [{y_zero.min():.0f}, {y_zero.max():.0f}]")
    
    # 非零值样本
    if len(valid_positive) >= 5:
        y_pos = valid_positive['textile_firms']
        print(f"  非零值样本 (n={len(valid_positive)}):")
        print(f"    企业数均值: {y_pos.mean():.1f}")
        print(f"    企业数标准差: {y_pos.std():.1f}")
        print(f"    企业数范围: [{y_pos.min():.0f}, {y_pos.max():.0f}]")
        print(f"    政策强度均值: {valid_positive['policy_intensity_total'].mean():.2f}")
    
    # 两样本t检验
    if len(valid_zero) >= 2 and len(valid_positive) >= 2:
        t_stat, t_pval = sp_stats.ttest_ind(valid_positive['textile_firms'], valid_zero['textile_firms'])
        print(f"\n  两组企业数差异t检验: t={t_stat:.2f}, p={t_pval:.4f}")
        if t_pval < 0.1:
            print(f"  [发现] 有政策 vs 无政策年份的企业数量存在显著差异")
        else:
            print(f"  [发现] 有政策 vs 无政策年份的企业数量无显著差异")

# ============================================================
# 改进3：合成控制法（Synthetic Control Method）
# ============================================================
def improvement3_synthetic_control(gy, policy_df):
    """
    合成控制法
    目的：为高阳县构建"反事实"对照组
    方法：用河北省其他县的加权平均合成"非高阳"
    """
    print("\n" + "="*80)
    print("改进3：合成控制法（Synthetic Control Method）")
    print("目标：为高阳县构建反事实对照组")
    print("="*80)
    
    policy_df['County_Code'] = policy_df['County_Code'].astype(str)
    
    # 获取所有县级单位（排除高阳）
    counties = policy_df[policy_df['County_Code'].str.len() == 6]['County_Code'].unique()
    counties = [c for c in counties if c != '130628']
    
    if len(counties) < 3:
        print(f"  可用对照组县数量不足({len(counties)}), 需要至少3个")
        print(f"  改用省级和市级作为对照")
        donor_pool = ['130000', '130600']  # 河北省、保定市
    else:
        donor_pool = counties[:5]  # 取前5个县
    
    print(f"\n【捐赠池】对照组单位: {donor_pool}")
    
    # 提取高阳县和对照组数据
    gy_data = policy_df[policy_df['County_Code'] == '130628'][['Year', 'policy_intensity_total']].copy()
    gy_data.columns = ['Year', 'gaoyang']
    
    donor_data = []
    for county in donor_pool:
        cd = policy_df[policy_df['County_Code'] == county][['Year', 'policy_intensity_total']].copy()
        cd.columns = ['Year', county]
        donor_data.append(cd)
    
    if len(donor_data) == 0:
        print(f"  无有效对照组数据")
        return
    
    merged = gy_data
    for dd in donor_data:
        merged = merged.merge(dd, on='Year', how='outer')
    
    merged = merged.dropna(subset=['gaoyang'])
    merged = merged.sort_values('Year')
    
    if len(merged) < 10:
        print(f"  合并后样本不足({len(merged)})")
        return
    
    print(f"\n  样本期: {int(merged.Year.min())}-{int(merged.Year.max())} (n={len(merged)})")
    
    # 简单合成：等权重平均
    donor_cols = [c for c in merged.columns if c != 'Year' and c != 'gaoyang']
    merged['synthetic'] = merged[donor_cols].mean(axis=1)
    
    # 计算处理效应（2017年后）
    merged['post'] = (merged['Year'] >= 2017).astype(int)
    merged['effect'] = merged['gaoyang'] - merged['synthetic']
    
    pre_effect = merged[merged['Year'] < 2017]['effect'].mean()
    post_effect = merged[merged['Year'] >= 2017]['effect'].mean()
    treatment_effect = post_effect - pre_effect
    
    print(f"\n【合成控制结果】")
    print(f"  处理前(2000-2016)平均差异: {pre_effect:.2f}")
    print(f"  处理后(2017-2024)平均差异: {post_effect:.2f}")
    print(f"  处理效应 (DID): {treatment_effect:.2f}")
    
    # 显著性检验
    pre = merged[merged['Year'] < 2017]['effect']
    post = merged[merged['Year'] >= 2017]['effect']
    if len(pre) >= 2 and len(post) >= 2:
        t_stat, t_pval = sp_stats.ttest_ind(post, pre)
        print(f"  t检验: t={t_stat:.2f}, p={t_pval:.4f}")
        
        if t_pval < 0.1:
            print(f"  [发现] 2017年后高阳县与对照组差异显著扩大")
        else:
            print(f"  [发现] 2017年后高阳县与对照组差异不显著")
    
    # 可视化数据
    print(f"\n  合成控制数据（年度）:")
    print(f"  {'年份':>6} {'高阳县':>10} {'合成对照':>10} {'差异':>8}")
    for _, row in merged.iterrows():
        print(f"  {int(row.Year):>6} {row.gaoyang:>10.2f} {row.synthetic:>10.2f} {row.effect:>8.2f}")

# ============================================================
# 改进4：事件研究法（多政策冲击）
# ============================================================
def improvement4_event_study(gy):
    """
    事件研究法
    目的：识别多个政策冲击事件，增加统计功效
    方法：计算政策冲击前后的累积效应
    """
    print("\n" + "="*80)
    print("改进4：事件研究法（多政策冲击分析）")
    print("目标：识别多个政策事件，增加统计功效")
    print("="*80)
    
    # 定义政策事件
    events = [
        {'year': 2008, 'name': '2008环保法修订', 'window': 3},
        {'year': 2013, 'name': '2013大气污染防治行动计划', 'window': 3},
        {'year': 2015, 'name': '2015供给侧改革', 'window': 3},
        {'year': 2017, 'name': '2017中央环保督察', 'window': 3},
        {'year': 2020, 'name': '2020双碳目标', 'window': 3},
    ]
    
    valid = gy.dropna(subset=['textile_firms']).copy()
    
    print(f"\n【政策事件列表】")
    print(f"  {'事件':>4} {'事件名称':<30} {'事件前均值':>10} {'事件后均值':>10} {'变化':>8} {'显著'}")
    print(f"  {'-'*75}")
    
    results = []
    for event in events:
        year = event['year']
        window = event['window']
        
        # 事件前window年
        pre = valid[(valid['Year'] >= year - window) & (valid['Year'] < year)]['textile_firms']
        # 事件后window年
        post = valid[(valid['Year'] >= year) & (valid['Year'] <= year + window)]['textile_firms']
        
        if len(pre) < 2 or len(post) < 2:
            print(f"  {year:>4} {event['name']:<30} {'样本不足':>35}")
            continue
        
        mean_pre = pre.mean()
        mean_post = post.mean()
        change = mean_post - mean_pre
        change_pct = change / mean_pre * 100 if mean_pre != 0 else 0
        
        t_stat, t_pval = sp_stats.ttest_ind(post, pre)
        sig = sig_star(t_pval)
        
        print(f"  {year:>4} {event['name']:<30} {mean_pre:>10.1f} {mean_post:>10.1f} {change:>+7.1f} {sig}")
        
        results.append({
            'year': year,
            'name': event['name'],
            'change': change,
            'change_pct': change_pct,
            'p_value': t_pval,
            'significant': t_pval < 0.1
        })
    
    # 汇总
    sig_events = [r for r in results if r['significant']]
    print(f"\n【汇总】")
    print(f"  总事件数: {len(results)}")
    print(f"  显著事件: {len(sig_events)}")
    print(f"  显著性比例: {len(sig_events)/len(results)*100:.1f}%")
    
    if len(sig_events) > 0:
        print(f"\n  显著事件详情:")
        for r in sig_events:
            direction = "增加" if r['change'] > 0 else "减少"
            print(f"    {r['name']}: 企业数{direction}{abs(r['change']):.1f}家 ({r['change_pct']:+.1f}%), p={r['p_value']:.4f}")

# ============================================================
# 主函数
# ============================================================
def run_robustness_improvements():
    print("\n" + "="*80)
    print("稳健性改进 - 4个方法全量执行")
    print("="*80)
    
    gy, policy_df, ent = load_data()
    
    # 改进1：Bootstrap
    bootstrap_results = improvement1_bootstrap(gy, n_bootstrap=10000)
    
    # 改进2：零膨胀模型
    improvement2_zero_inflated(gy)
    
    # 改进3：合成控制法
    improvement3_synthetic_control(gy, policy_df)
    
    # 改进4：事件研究法
    improvement4_event_study(gy)
    
    print("\n" + "="*80)
    print("稳健性改进 - 全部完成")
    print("="*80)

if __name__ == "__main__":
    run_robustness_improvements()
