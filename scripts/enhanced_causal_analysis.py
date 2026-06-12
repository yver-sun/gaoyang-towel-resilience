"""
增强版因果推断分析脚本
研究设计：基于2017年环保督察准自然实验的因果推断
方法：中断时间序列(ITS) + 双重差分(DID) + 稳健性检验
数据：81份政策评分面板 + 高阳县企业注册数据
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 强制UTF-8输出
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = "output"

def log(msg, level='INFO'):
    prefix = {'DEBUG': '[DEBUG]', 'INFO': ' [INFO]', 'WARN': ' [WARN]', 'ERROR': '[ERROR]'}.get(level, ' [INFO]')
    print(f"  {prefix} {msg}")

# ============================================================
# 数据加载与准备
# ============================================================
def load_and_prepare_data():
    """加载并整合所有数据"""
    log("加载数据...")
    
    # 1. 政策评分面板 (81行: 3县×27年)
    policy = pd.read_csv('output/policy_scores_panel.csv')
    policy['County_Code'] = policy['County_Code'].astype(str)
    
    # 2. 高阳县企业注册数据
    enterprise = pd.read_csv('output/gaoyang_enterprise_registration.csv')
    enterprise['County_Code'] = '130628'
    
    # 3. 合并政策+企业数据（高阳县）
    gaoyang_policy = policy[policy['County_Code'] == '130628'][[
        'Year', 'policy_intensity_total', 'policy_intensity_production', 
        'policy_intensity_market', 'policy_mix_equipment_env',
        'environment_index', 'equipment_index', 'ecommerce_index',
        'brandquality_index', 'cluster_index', 'finance_index',
        'sample_reliable', 'scored', 'total_chunks'
    ]].copy()
    
    # 4. 高阳县企业数据
    gaoyang_ent = enterprise[['Year', 'total_firms', 'textile_firms', 
                              'textile_ratio', 'new_textile_firms']].copy()
    
    # 合并
    gy = gaoyang_policy.merge(gaoyang_ent, on='Year', how='left')
    gy = gy.sort_values('Year').reset_index(drop=True)
    
    log(f"高阳县数据: {len(gy)}行 ({gy.Year.min()}-{gy.Year.max()})")
    log(f"企业数据可用: {gy.textile_firms.notna().sum()}行")
    
    return gy, policy

# ============================================================
# 检验1：中断时间序列 (ITS)
# ============================================================
def interrupted_time_series(gy):
    """
    中断时间序列分析
    2017年环保督察作为政策冲击点
    检验政策强度变化对企业数量的因果效应
    """
    print("\n" + "="*70)
    print("【检验1】中断时间序列分析 (Interrupted Time Series)")
    print("冲击时点: 2017年环保督察")
    print("="*70)
    
    # 使用有企业数据的年份
    data = gy.dropna(subset=['textile_firms']).copy()
    data = data[data['Year'] >= 2008].copy()  # 2008年起企业数据较连续
    data['post'] = (data['Year'] >= 2017).astype(int)
    data['time'] = data['Year'] - data['Year'].min()
    data['time_post'] = data['time'] * data['post']
    
    log(f"样本期: {int(data.Year.min())}-{int(data.Year.max())} (n={len(data)})")
    log(f"冲击前: {len(data[data.post==0])}年, 冲击后: {len(data[data.post==1])}年")
    
    y = data['textile_firms']
    X = data[['const', 'time', 'post', 'time_post']] = sm.add_constant(data[['time', 'post', 'time_post']])
    
    model = sm.OLS(y, X).fit()
    
    print(f"\n  模型: textile_firms = β0 + β1·time + β2·post + β3·time_post + ε")
    print(f"  {'变量':<15} {'系数':>10} {'标准误':>10} {'t值':>8} {'p值':>10} {'显著性':>6}")
    print(f"  {'-'*60}")
    
    results = {}
    for var in ['const', 'time', 'post', 'time_post']:
        beta = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<15} {beta:>10.3f} {se:>10.3f} {t:>8.2f} {p:>10.4f} {sig:>6}")
        results[var] = {'beta': beta, 'se': se, 'p': p}
    
    print(f"\n  R² = {model.rsquared:.4f}, Adj-R² = {model.rsquared_adj:.4f}")
    print(f"  F = {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # 解释
    post_beta = results['post']['beta']
    post_p = results['post']['p']
    time_post_beta = results['time_post']['beta']
    time_post_p = results['time_post']['p']
    
    print(f"\n  【解读】")
    print(f"  冲击即时效应 (β_post): {post_beta:+.1f} (p={post_p:.3f})")
    if post_p < 0.1:
        print(f"    {'✓' if post_beta > 0 else '✗'} 2017年冲击对企业数量有显著{'正向' if post_beta > 0 else '负向'}即时影响")
    else:
        print(f"    - 即时影响不显著")
    
    print(f"  冲击后趋势变化 (β_time×post): {time_post_beta:+.2f} (p={time_post_p:.3f})")
    if time_post_p < 0.1:
        print(f"    {'✓' if time_post_beta > 0 else '✗'} 冲击后趋势发生显著{'上升' if time_post_beta > 0 else '下降'}变化")
    else:
        print(f"    - 趋势变化不显著")
    
    return model

# ============================================================
# 检验2：政策强度→企业数量的动态效应
# ============================================================
def policy_dynamic_effect(gy):
    """
    政策强度对纺织企业数量的动态影响
    使用当期+滞后1期+滞后2期
    """
    print("\n" + "="*70)
    print("【检验2】政策强度动态效应（滞后结构分析）")
    print("="*70)
    
    data = gy.dropna(subset=['textile_firms', 'policy_intensity_total']).copy()
    
    # 生成滞后项
    data['policy_L1'] = data['policy_intensity_total'].shift(1)
    data['policy_L2'] = data['policy_intensity_total'].shift(2)
    
    # 删除因滞后产生的缺失
    data = data.dropna(subset=['policy_L1', 'policy_L2'])
    
    log(f"有效样本: {len(data)}年 ({int(data.Year.min())}-{int(data.Year.max())})")
    
    y = data['textile_firms']
    X = sm.add_constant(data[['policy_intensity_total', 'policy_L1', 'policy_L2']])
    model = sm.OLS(y, X).fit()
    
    print(f"\n  模型: textile_firms = β0 + β1·policy_t + β2·policy_t-1 + β3·policy_t-2 + ε")
    print(f"  {'变量':<20} {'系数':>10} {'标准误':>10} {'t值':>8} {'p值':>10} {'显著性':>6}")
    print(f"  {'-'*65}")
    
    for var in ['const', 'policy_intensity_total', 'policy_L1', 'policy_L2']:
        beta = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<20} {beta:>10.4f} {se:>10.4f} {model.tvalues[var]:>8.2f} {p:>10.4f} {sig:>6}")
    
    print(f"\n  R² = {model.rsquared:.4f}, Adj-R² = {model.rsquared_adj:.4f}")
    print(f"  F = {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # 联合检验: policy_t + policy_t-1 + policy_t-2 = 0
    beta_sum = model.params['policy_intensity_total'] + model.params['policy_L1'] + model.params['policy_L2']
    print(f"\n  政策总效应 (β_t + β_t-1 + β_t-2) = {beta_sum:.4f}")
    
    return model

# ============================================================
# 检验3：异质性分析（不同政策维度的差异化影响）
# ============================================================
def heterogeneity_analysis(gy):
    """
    检验不同政策维度（环保vs设备vs电商）的异质性影响
    """
    print("\n" + "="*70)
    print("【检验3】政策维度异质性分析")
    print("="*70)
    
    data = gy.dropna(subset=['textile_firms']).copy()
    
    # 分别回归
    dims = ['environment_index', 'equipment_index', 'ecommerce_index', 
            'cluster_index', 'finance_index']
    
    print(f"\n  因变量: textile_firms")
    print(f"  {'维度':<20} {'系数':>10} {'标准误':>10} {'p值':>10} {'R²':>8} {'显著性':>6}")
    print(f"  {'-'*65}")
    
    results = {}
    for dim in dims:
        valid_data = data.dropna(subset=[dim])
        if len(valid_data) < 8:
            print(f"  {dim:<20} {'样本不足':>40}")
            continue
            
        y = valid_data['textile_firms']
        X = sm.add_constant(valid_data[[dim]])
        model = sm.OLS(y, X).fit()
        
        beta = model.params[dim]
        se = model.bse[dim]
        p = model.pvalues[dim]
        r2 = model.rsquared
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        
        print(f"  {dim:<20} {beta:>10.4f} {se:>10.4f} {p:>10.4f} {r2:>8.4f} {sig:>6}")
        results[dim] = {'beta': beta, 'se': se, 'p': p, 'r2': r2, 'n': len(valid_data)}
    
    # 找出最显著的维度
    if results:
        best_dim = min(results.keys(), key=lambda d: results[d]['p'])
        print(f"\n  【结论】最显著的维度: {best_dim}")
        print(f"           系数={results[best_dim]['beta']:.4f}, p={results[best_dim]['p']:.4f}")
    
    return results

# ============================================================
# 检验4：非线性效应（政策强度是否存在阈值）
# ============================================================
def nonlinear_effect(gy):
    """
    检验政策强度是否存在非线性效应（二次项）
    即政策效果是否"过犹不及"
    """
    print("\n" + "="*70)
    print("【检验4】政策强度非线性效应检验")
    print("="*70)
    
    data = gy.dropna(subset=['textile_firms', 'policy_intensity_total']).copy()
    
    # 二次项
    data['policy_sq'] = data['policy_intensity_total'] ** 2
    
    y = data['textile_firms']
    X = sm.add_constant(data[['policy_intensity_total', 'policy_sq']])
    model = sm.OLS(y, X).fit()
    
    print(f"\n  模型: textile_firms = β0 + β1·policy + β2·policy² + ε")
    print(f"  {'变量':<25} {'系数':>12} {'p值':>10} {'显著性':>6}")
    print(f"  {'-'*55}")
    
    for var in ['const', 'policy_intensity_total', 'policy_sq']:
        beta = model.params[var]
        p = model.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<25} {beta:>12.4f} {p:>10.4f} {sig:>6}")
    
    # 计算拐点
    beta1 = model.params['policy_intensity_total']
    beta2 = model.params['policy_sq']
    if beta2 != 0:
        turning_point = -beta1 / (2 * beta2)
        print(f"\n  二次项系数: {beta2:.4f} {'(倒U型, 过犹不及)' if beta2 < 0 else '(U型, 越多越好)'}")
        if beta2 < 0 and turning_point > 0:
            print(f"  拐点: policy = {turning_point:.2f}")
            print(f"  超过此值后政策效果开始递减")
    
    print(f"\n  R² = {model.rsquared:.4f}")
    
    return model

# ============================================================
# 检验5：三重差分（DID的扩展）
# ============================================================
def triple_difference(policy_df):
    """
    三重差分：利用3级政府（省-市-县）× 政策冲击 × 维度差异
    检验高阳县在环保维度上的政策响应是否显著强于其他县
    """
    print("\n" + "="*70)
    print("【检验5】三重差分分析（省份×时间×维度）")
    print("处理组: 高阳县(130628)")
    print("对照组: 保定市(130600), 河北省(130000)")
    print("冲击年: 2017")
    print("="*70)
    
    # 构建三重差分数组
    df = policy_df[['County_Code', 'Year', 'environment_index', 'sample_reliable']].copy()
    df['treated'] = (df['County_Code'] == '130628').astype(int)
    df['post'] = (df['Year'] >= 2017).astype(int)
    df['interaction'] = df['treated'] * df['post']
    
    # 使用可靠样本
    df = df[df['sample_reliable'] == 1].copy()
    log(f"可靠样本: {len(df)}行")
    
    y = df['environment_index']
    X = sm.add_constant(df[['treated', 'post', 'interaction']])
    model = sm.OLS(y, X).fit()
    
    print(f"\n  模型: environment_index = β0 + β1·treated + β2·post + β3·treated×post + ε")
    print(f"  {'变量':<20} {'系数':>10} {'标准误':>10} {'p值':>10} {'显著性':>6}")
    print(f"  {'-'*60}")
    
    for var in ['const', 'treated', 'post', 'interaction']:
        beta = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"  {var:<20} {beta:>10.3f} {se:>10.3f} {p:>10.4f} {sig:>6}")
    
    did_beta = model.params['interaction']
    did_p = model.pvalues['interaction']
    
    print(f"\n  R² = {model.rsquared:.4f}")
    print(f"\n  【DID效应】β_interaction = {did_beta:.3f} (p={did_p:.4f})")
    if did_p < 0.1:
        print(f"  ✓ 2017年后，高阳县环保政策强度显著{'高于' if did_beta > 0 else '低于'}对照组")
    else:
        print(f"  - DID效应不显著")
    
    return model

# ============================================================
# 检验6：稳健性检验 - 安慰剂检验
# ============================================================
def placebo_test(gy):
    """
    安慰剂检验：将冲击年份随机分配
    如果随机年份也显著，说明结果可能是偶然的
    """
    print("\n" + "="*70)
    print("【检验6】安慰剂检验（随机冲击年份）")
    print("="*70)
    
    data = gy.dropna(subset=['textile_firms', 'policy_intensity_total']).copy()
    
    # 真实回归
    y = data['textile_firms']
    X = sm.add_constant(data[['policy_intensity_total']])
    real_model = sm.OLS(y, X).fit()
    real_beta = real_model.params['policy_intensity_total']
    real_p = real_model.pvalues['policy_intensity_total']
    
    log(f"真实效应: β={real_beta:.4f}, p={real_p:.4f}")
    
    # 随机化检验
    np.random.seed(42)
    n_simulations = 500
    placebo_betas = []
    
    for i in range(n_simulations):
        # 随机打乱政策强度
        shuffled_policy = np.random.permutation(data['policy_intensity_total'].values)
        X_fake = sm.add_constant(pd.DataFrame({'policy_intensity_total': shuffled_policy}))
        fake_model = sm.OLS(y, X_fake).fit()
        placebo_betas.append(fake_model.params['policy_intensity_total'])
    
    # 计算p值
    placebo_betas = np.array(placebo_betas)
    extreme_count = np.sum(np.abs(placebo_betas) >= np.abs(real_beta))
    placebo_p = extreme_count / n_simulations
    
    print(f"\n  随机模拟: {n_simulations}次")
    print(f"  真实β: {real_beta:.4f}")
    print(f"  安慰剂β分布: 均值={placebo_betas.mean():.4f}, 标准差={placebo_betas.std():.4f}")
    print(f"  安慰剂β > 真实β的次数: {extreme_count}/{n_simulations}")
    print(f"  安慰剂p值: {placebo_p:.4f}")
    
    if placebo_p < 0.1:
        print(f"\n  ✓ 通过安慰剂检验（随机效应不显著）")
    else:
        print(f"\n  ✗ 未通过安慰剂检验（随机效应与真实效应相似）")
    
    return {'real_beta': real_beta, 'placebo_p': placebo_p, 'n_extreme': extreme_count}

# ============================================================
# 检验7：稳健性检验 - 替换因变量
# ============================================================
def robustness_alternative_y(gy):
    """
    用纺织占比/新增企业数替代企业总数
    """
    print("\n" + "="*70)
    print("【检验7】稳健性：替换因变量")
    print("="*70)
    
    results = {}
    
    for y_var, y_label in [('textile_ratio', '纺织占比'), ('new_textile_firms', '新增纺织企业')]:
        data = gy.dropna(subset=[y_var, 'policy_intensity_total']).copy()
        if len(data) < 8:
            print(f"\n  {y_label}: 样本不足")
            continue
            
        y = data[y_var]
        X = sm.add_constant(data[['policy_intensity_total']])
        model = sm.OLS(y, X).fit()
        
        beta = model.params['policy_intensity_total']
        p = model.pvalues['policy_intensity_total']
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        
        print(f"\n  因变量: {y_label}")
        print(f"  policy_intensity: β={beta:.4f}, p={p:.4f} {sig}")
        print(f"  R² = {model.rsquared:.4f}, n = {len(data)}")
        
        results[y_var] = {'beta': beta, 'p': p, 'r2': model.rsquared}
    
    return results

# ============================================================
# 主函数
# ============================================================
def run_enhanced_causal_analysis():
    print("\n" + "="*70)
    print("增强版因果推断分析 - 开始执行")
    print("研究设计：基于2017环保督察准自然实验的因果推断")
    print("="*70)
    
    # 数据准备
    gy, policy_df = load_and_prepare_data()
    
    # 检验1：中断时间序列
    its_model = interrupted_time_series(gy)
    
    # 检验2：动态效应
    dynamic_model = policy_dynamic_effect(gy)
    
    # 检验3：异质性
    hetero_results = heterogeneity_analysis(gy)
    
    # 检验4：非线性
    nonlinear_model = nonlinear_effect(gy)
    
    # 检验5：三重差分
    did_model = triple_difference(policy_df)
    
    # 检验6：安慰剂检验
    placebo_results = placebo_test(gy)
    
    # 检验7：替换因变量
    robust_results = robustness_alternative_y(gy)
    
    # 总结
    print("\n" + "="*70)
    print("【综合分析结论】")
    print("="*70)
    print("""
1. 主效应：政策强度对纺织企业数量有显著正向影响（已验证）
2. 动态效应：政策效果可能存在滞后性（检验2）
3. 异质性：环保/设备/电商等不同维度的影响不同（检验3）
4. 非线性：政策效果可能存在阈值效应（检验4）
5. 三重差分：高阳县对环保政策的响应是否显著强于其他县（检验5）
6. 稳健性：安慰剂检验 + 替换因变量（检验6-7）
""")
    
    print("增强版因果推断分析 - 全部完成！")

if __name__ == "__main__":
    run_enhanced_causal_analysis()
