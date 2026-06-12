"""
Phase C: 合成控制法(SCM) - 用保定24区县构建"合成高阳"
比较真实高阳与合成高阳在2017年后的纺织企业数差异
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

PANEL = "output/baoding_county_panel.csv"
OUTPUT_DIR = "analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TREATMENT_YEAR = 2017
TREATED_CODE = '130628'


def load_panel():
    df = pd.read_csv(PANEL, encoding='utf-8-sig')
    df['Year'] = df['Year'].astype(int)
    df['County_Code'] = df['County_Code'].astype(str)
    return df


def build_scm_matrices(df):
    """构建SCM矩阵: Y_pre (JxT0), Y_post (JxT1), X_pred (JxK)"""
    years = sorted(df['Year'].unique())
    pre_years = [y for y in years if y < TREATMENT_YEAR]
    post_years = [y for y in years if y >= TREATMENT_YEAR]

    counties = sorted(df['County_Code'].unique())
    donor_counties = [c for c in counties if c != TREATED_CODE]

    # 结果变量: textile_firms
    pivot = df.pivot(index='County_Code', columns='Year', values='textile_firms')

    # 处理组 pre-treatment 结果
    Y1_pre = pivot.loc[TREATED_CODE, pre_years].values.astype(float)

    # 对照组 pre-treatment 结果矩阵 (J x T0)
    Y0_pre = pivot.loc[donor_counties, pre_years].values.astype(float)

    # 预测变量: pre-treatment均值 + 纺织业占比趋势
    pre_mean = pivot.loc[donor_counties, pre_years].mean(axis=1).values
    pre_trend = np.array([
        np.polyfit(range(len(pre_years)), pivot.loc[c, pre_years].values.astype(float), 1)[0]
        for c in donor_counties
    ])

    # 纺织业占比预测变量
    ratio_pivot = df.pivot(index='County_Code', columns='Year', values='textile_ratio')
    pre_ratio_mean = ratio_pivot.loc[donor_counties, pre_years].mean(axis=1).values

    X0 = np.column_stack([pre_mean, pre_trend, pre_ratio_mean])
    X1 = np.array([
        pivot.loc[TREATED_CODE, pre_years].mean(),
        np.polyfit(range(len(pre_years)), Y1_pre, 1)[0],
        ratio_pivot.loc[TREATED_CODE, pre_years].mean()
    ])

    # 后处理结果
    Y1_post = pivot.loc[TREATED_CODE, post_years].values.astype(float)
    Y0_post = pivot.loc[donor_counties, post_years].values.astype(float)

    return {
        'pre_years': pre_years,
        'post_years': post_years,
        'donor_counties': donor_counties,
        'Y1_pre': Y1_pre,
        'Y0_pre': Y0_pre,
        'X0': X0,
        'X1': X1,
        'Y1_post': Y1_post,
        'Y0_post': Y0_post,
        'pivot': pivot,
    }


def scm_optimize(data, V=None):
    """优化SCM权重"""
    X0, X1 = data['X0'], data['X1']
    J = X0.shape[0]
    K = X0.shape[1]

    if V is None:
        V = np.eye(K)

    def loss(W):
        return np.sum((X1 - X0.T @ W)**2)

    # 约束: W_i >= 0, sum(W) = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(J)]
    w0 = np.ones(J) / J

    result = minimize(loss, w0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-12})

    return result.x


def placebo_test(data, W_opt):
    """安慰剂检验：将每个对照县作为假处理组，比较效应大小"""
    pivot = data['pivot']
    pre_years = data['pre_years']
    post_years = data['post_years']
    donor_counties = data['donor_counties']

    placebo_effects = {}
    for county in donor_counties:
        # 以county为假处理组
        Y1_pre_fake = pivot.loc[county, pre_years].values.astype(float)
        Y1_post_fake = pivot.loc[county, post_years].values.astype(float)

        # 用其他donor构建合成
        other_donors = [c for c in donor_counties if c != county]
        Y0_pre_fake = pivot.loc[other_donors, pre_years].values.astype(float)

        # 简化预测变量
        pre_mean_fake = pivot.loc[other_donors, pre_years].mean(axis=1).values
        X0_fake = np.column_stack([pre_mean_fake,
                                    np.zeros(len(other_donors)),
                                    np.zeros(len(other_donors))])
        X1_fake = np.array([Y1_pre_fake.mean(), 0, 0])

        fake_data = {
            'X0': X0_fake, 'X1': X1_fake,
            'Y1_pre': Y1_pre_fake, 'Y0_pre': Y0_pre_fake,
            'pre_years': pre_years, 'post_years': post_years,
            'donor_counties': other_donors, 'pivot': pivot,
            'Y1_post': Y1_post_fake,
            'Y0_post': pivot.loc[other_donors, post_years].values.astype(float),
        }

        try:
            W_fake = scm_optimize(fake_data)
            # 计算处理效应
            synth_post = fake_data['Y0_post'].T @ W_fake
            effect = np.mean(Y1_post_fake - synth_post)
            # pre-treatment拟合质量
            synth_pre = Y0_pre_fake.T @ W_fake
            pre_rmse = np.sqrt(np.mean((Y1_pre_fake - synth_pre)**2))
            placebo_effects[county] = {'effect': effect, 'pre_rmse': pre_rmse}
        except:
            pass

    return placebo_effects


def main():
    print("=" * 60)
    print("Phase C: 合成控制法(SCM)分析")
    print("=" * 60)

    df = load_panel()
    print(f"面板: {len(df)}行, {df['County_Code'].nunique()}县, {df['Year'].min()}-{df['Year'].max()}")

    data = build_scm_matrices(df)
    print(f"处理组: 高阳县(130628)")
    print(f"对照组: {len(data['donor_counties'])}县")
    print(f"Pre-treatment: {len(data['pre_years'])}年 ({data['pre_years'][0]}-{data['pre_years'][-1]})")
    print(f"Post-treatment: {len(data['post_years'])}年 ({data['post_years'][0]}-{data['post_years'][-1]})")

    # 优化SCM权重
    W_opt = scm_optimize(data)
    significant_donors = [(data['donor_counties'][i], w) for i, w in enumerate(W_opt) if w > 0.01]
    significant_donors.sort(key=lambda x: -x[1])

    print(f"\nSCM权重 (>{0.01}):")
    for code, w in significant_donors[:8]:
        name = df[df['County_Code'] == code]['County_Name'].iloc[0]
        print(f"  {name}({code}): {w:.4f}")

    # Pre-treatment 拟合
    synth_pre = data['Y0_pre'].T @ W_opt
    pre_rmse = np.sqrt(np.mean((data['Y1_pre'] - synth_pre)**2))
    print(f"\nPre-treatment RMSE: {pre_rmse:.1f}")

    # Post-treatment 处理效应
    synth_post = data['Y0_post'].T @ W_opt
    gap = data['Y1_post'] - synth_post
    avg_effect = np.mean(gap)
    print(f"平均处理效应 (ATT): {avg_effect:.1f} 纺织企业/年")
    print(f"  (正数表示真实高阳的纺织企业数高于合成高阳)")

    # 逐年差距
    print(f"\n逐年差距 (真实 - 合成):")
    for i, year in enumerate(data['post_years']):
        print(f"  {year}: 真实={data['Y1_post'][i]:.0f}, 合成={synth_post[i]:.0f}, 差距={gap[i]:.0f}")

    # 安慰剂检验
    print(f"\n--- 安慰剂检验 ---")
    placebo = placebo_test(data, W_opt)

    real_effect_abs = abs(avg_effect)
    n_better = sum(1 for v in placebo.values() if abs(v['effect']) >= real_effect_abs)
    p_value = (n_better + 1) / (len(placebo) + 1)
    print(f"  真实效应绝对值: {real_effect_abs:.1f}")
    print(f"  安慰剂中|效应| ≥ 真实效应的比例: {n_better}/{len(placebo)}")
    print(f"  置换p值: {p_value:.4f}")
    if p_value < 0.1:
        print(f"  [通过] p < 0.1, 真实效应显著大于随机置换")
    else:
        print(f"  [未通过] p >= 0.1, 真实效应在置换分布内不显著")

    # ===== 保存结果 =====
    # SCM时间序列
    scm_series = pd.DataFrame({
        'Year': data['pre_years'] + data['post_years'],
        'Gaoyang_actual': list(data['Y1_pre']) + list(data['Y1_post']),
        'Synthetic_Gaoyang': list(synth_pre) + list(synth_post),
        'Gap': list(data['Y1_pre'] - synth_pre) + list(gap),
        'Period': ['Pre'] * len(data['pre_years']) + ['Post'] * len(data['post_years']),
    })
    scm_series.to_csv(os.path.join(OUTPUT_DIR, "scm_results.csv"), index=False, encoding='utf-8-sig')

    # 权重
    weights_df = pd.DataFrame({
        'County_Code': data['donor_counties'],
        'Weight': W_opt,
    })
    weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
    weights_df.to_csv(os.path.join(OUTPUT_DIR, "scm_weights.csv"), index=False, encoding='utf-8-sig')

    # 安慰剂检验结果
    placebo_df = pd.DataFrame([
        {'County_Code': k, 'Effect': v['effect'], 'Pre_RMSE': v['pre_rmse']}
        for k, v in placebo.items()
    ])
    placebo_df.to_csv(os.path.join(OUTPUT_DIR, "scm_placebo.csv"), index=False, encoding='utf-8-sig')

    print(f"\n结果已保存至 {OUTPUT_DIR}/")
    print("  - scm_results.csv (SCM时间序列)")
    print("  - scm_weights.csv (合成权重)")
    print("  - scm_placebo.csv (安慰剂检验)")


if __name__ == "__main__":
    main()
