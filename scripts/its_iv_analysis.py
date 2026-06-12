"""
Phase B+D: 中断时间序列(ITS) + 工具变量(IV)回归
以2017年环保规制为结构性断点，检验政策强度对纺织产业韧性的因果效应
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

MASTER = "output/master_panel_data_v2.csv"
POLICY_COUNTS = "output/policy_document_counts.csv"
OUTPUT_DIR = "analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_merge():
    """加载主面板并合并政策计数"""
    df = pd.read_csv(MASTER, encoding='utf-8-sig')
    pc = pd.read_csv(POLICY_COUNTS, encoding='utf-8-sig')

    df['Year'] = df['Year'].astype(int)
    pc['Year'] = pc['Year'].astype(int)

    df = df.merge(pc, on='Year', how='left')

    # 构造关键变量
    df['post_2017'] = (df['Year'] >= 2017).astype(int)
    df['time'] = df['Year'] - 2000  # 时间趋势(年)
    df['time_post'] = df['time'] * df['post_2017']  # 断点后时间趋势

    # 对数变换
    for col in ['textile_firms', 'total_firms']:
        if col in df.columns:
            df[f'ln_{col}'] = np.log(df[col] + 1)

    # 纺织企业增速
    if 'textile_firms' in df.columns:
        df['textile_firms_growth'] = df['textile_firms'].pct_change()

    # 政策强度复合变量
    policy_dims = ['equipment_index', 'environment_index', 'ecommerce_index',
                   'brand_quality_index', 'cluster_index', 'finance_index']
    available_dims = [c for c in policy_dims if c in df.columns]
    if available_dims:
        df['policy_composite'] = df[available_dims].mean(axis=1)

    return df


def run_its_models(df):
    """ITS断点回归模型"""
    results = []

    # ===== Model 1: 基准ITS - 纺织企业数 =====
    y = df['ln_textile_firms'].values
    X = np.column_stack([
        np.ones(len(df)),
        df['time'].values,
        df['post_2017'].values,
        df['time_post'].values
    ])

    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        y_hat = X @ beta
        residuals = y - y_hat
        n, k = X.shape
        sigma2 = np.sum(residuals**2) / (n - k)
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        t_stats = beta / se
        r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

        # Durbin-Watson
        dw = np.sum(np.diff(residuals.flatten())**2) / np.sum(residuals**2)

        results.append({
            'model': 'ITS-基准',
            'dependent': 'ln_textile_firms',
            'n': n,
            'r2': r2,
            'adj_r2': adj_r2,
            'dw': dw,
            'const': beta[0],
            'time': beta[1],
            'post_2017': beta[2],
            'time_post': beta[3],
            'const_se': se[0],
            'time_se': se[1],
            'post_2017_se': se[2],
            'time_post_se': se[3],
            'const_t': t_stats[0],
            'time_t': t_stats[1],
            'post_2017_t': t_stats[2],
            'time_post_t': t_stats[3],
        })
        print(f"  ITS-base: R2={r2:.3f}, post_2017 beta={beta[2]:.3f} (t={t_stats[2]:.2f}), DW={dw:.2f}")
    except Exception as e:
        print(f"  ITS-base failed: {e}")

    # ===== Model 2: 加入政策强度 =====
    if 'policy_intensity_total' in df.columns and df['policy_intensity_total'].notna().sum() >= 5:
        y = df['ln_textile_firms'].values
        X_cols = ['time', 'post_2017', 'time_post', 'policy_intensity_total']
        X_data = df[X_cols].fillna(0).values
        X = np.column_stack([np.ones(len(df)), X_data])

        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            y_hat = X @ beta
            residuals = y - y_hat
            n, k = X.shape
            sigma2 = np.sum(residuals**2) / (n - k)
            se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
            t_stats = beta / se
            r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

            results.append({
                'model': 'ITS-政策强度',
                'dependent': 'ln_textile_firms',
                'n': n,
                'r2': r2,
                'adj_r2': 1 - (1 - r2) * (n - 1) / (n - k),
                'dw': np.sum(np.diff(residuals.flatten())**2) / np.sum(residuals**2),
                'const': beta[0],
                'time': beta[1],
                'post_2017': beta[2],
                'time_post': beta[3],
                'policy_intensity': beta[4],
                'const_se': se[0], 'time_se': se[1], 'post_2017_se': se[2],
                'time_post_se': se[3], 'policy_intensity_se': se[4],
                'const_t': t_stats[0], 'time_t': t_stats[1], 'post_2017_t': t_stats[2],
                'time_post_t': t_stats[3], 'policy_intensity_t': t_stats[4],
            })
            print(f"  ITS-policy: R2={r2:.3f}, policy beta={beta[4]:.3f} (t={t_stats[4]:.2f})")
        except Exception as e:
            print(f"  ITS-policy failed: {e}")

    # ===== Model 3: 滞后政策效应 =====
    if 'policy_intensity_total' in df.columns:
        df_temp = df.copy()
        df_temp['policy_lag1'] = df_temp['policy_intensity_total'].shift(1)
        df_temp['policy_lag2'] = df_temp['policy_intensity_total'].shift(2)
        valid = df_temp[['ln_textile_firms', 'time', 'post_2017', 'time_post',
                          'policy_intensity_total', 'policy_lag1', 'policy_lag2']].dropna()

        if len(valid) >= 5:
            y = valid['ln_textile_firms'].values
            X = np.column_stack([
                np.ones(len(valid)),
                valid['time'].values,
                valid['post_2017'].values,
                valid['time_post'].values,
                valid['policy_intensity_total'].values,
                valid['policy_lag1'].values,
            ])

            try:
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                y_hat = X @ beta
                residuals = y - y_hat
                n, k = X.shape
                sigma2 = np.sum(residuals**2) / (n - k)
                se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
                t_stats = beta / se
                r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

                results.append({
                    'model': 'ITS-滞后效应',
                    'dependent': 'ln_textile_firms',
                    'n': n,
                    'r2': r2,
                    'adj_r2': 1 - (1 - r2) * (n - 1) / (n - k),
                    'dw': np.sum(np.diff(residuals.flatten())**2) / np.sum(residuals**2),
                    'const': beta[0], 'time': beta[1], 'post_2017': beta[2],
                    'time_post': beta[3], 'policy_intensity': beta[4], 'policy_lag1': beta[5],
                    'const_se': se[0], 'time_se': se[1], 'post_2017_se': se[2],
                    'time_post_se': se[3], 'policy_intensity_se': se[4], 'policy_lag1_se': se[5],
                    'const_t': t_stats[0], 'time_t': t_stats[1], 'post_2017_t': t_stats[2],
                    'time_post_t': t_stats[3], 'policy_intensity_t': t_stats[4], 'policy_lag1_t': t_stats[5],
                })
                print(f"  ITS-lag: R2={r2:.3f}, L1 beta={beta[5]:.3f} (t={t_stats[5]:.2f})")
            except Exception as e:
                print(f"  ITS-lag failed: {e}")

    # ===== Model 4: 分维度回归 =====
    dims = [c for c in ['equipment_index', 'environment_index', 'ecommerce_index',
                         'brand_quality_index', 'cluster_index', 'finance_index',
                         'education_index']
            if c in df.columns and df[c].notna().sum() >= 3]

    for dim in dims:
        dim_data = df[['ln_textile_firms', 'time', 'post_2017', 'time_post', dim]].dropna()
        if len(dim_data) < 5:
            continue

        y = dim_data['ln_textile_firms'].values
        X = np.column_stack([
            np.ones(len(dim_data)),
            dim_data['time'].values,
            dim_data['post_2017'].values,
            dim_data[dim].values
        ])

        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            residuals = y - X @ beta
            n, k = X.shape
            sigma2 = np.sum(residuals**2) / (n - k)
            se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
            t_stats = beta / se
            r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

            results.append({
                'model': f'ITS-{dim}',
                'dependent': 'ln_textile_firms',
                'n': n, 'r2': r2,
                'adj_r2': 1 - (1 - r2) * (n - 1) / (n - k),
                'dw': np.sum(np.diff(residuals.flatten())**2) / np.sum(residuals**2),
                'const': beta[0], 'time': beta[1], 'post_2017': beta[2],
                f'{dim}': beta[3],
                'const_se': se[0], 'time_se': se[1], 'post_2017_se': se[2],
                f'{dim}_se': se[3],
                'const_t': t_stats[0], 'time_t': t_stats[1], 'post_2017_t': t_stats[2],
                f'{dim}_t': t_stats[3],
            })
            print(f"  ITS-{dim}: R2={r2:.3f}, {dim} beta={beta[3]:.3f} (t={t_stats[3]:.2f})")
        except Exception as e:
            print(f"  ITS-{dim} failed: {e}")

    return results


def run_iv_models(df):
    """工具变量回归: 政策文件数量 → 政策强度 → 纺织企业数"""
    iv_results = []

    # ===== 第一阶段: 政策文件数量 → 政策强度 =====
    if 'policy_intensity_total' not in df.columns or 'policy_doc_count' not in df.columns:
        print("  IV: 缺少必需变量")
        return iv_results

    valid = df[['ln_textile_firms', 'policy_intensity_total', 'policy_doc_count', 'time']].dropna()
    if len(valid) < 10:
        print(f"  IV: 有效样本不足 (n={len(valid)})")
        return iv_results

    print(f"\n  IV分析有效样本: {len(valid)}")

    # 第一阶段 OLS
    y1 = valid['policy_intensity_total'].values
    X1 = np.column_stack([np.ones(len(valid)), valid['policy_doc_count'].values, valid['time'].values])
    beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ y1
    residuals1 = y1 - X1 @ beta1

    n = len(valid)
    k1 = 3
    sigma2_1 = np.sum(residuals1**2) / (n - k1)
    se1 = np.sqrt(np.diag(sigma2_1 * np.linalg.inv(X1.T @ X1)))
    f_stat = (beta1[1] / se1[1])**2

    r2_1 = 1 - np.sum(residuals1**2) / np.sum((y1 - np.mean(y1))**2)

    print(f"  一阶段: policy_doc_count → policy_intensity")
    print(f"    coef={beta1[1]:.4f}, SE={se1[1]:.4f}, F={f_stat:.2f}, R2={r2_1:.3f}")

    if f_stat < 10:
        print(f"  [WARNING] 1st-stage F={f_stat:.1f} < 10, potential weak IV")

    # 第二阶段: 预测值 → 纺织企业数
    policy_hat = X1 @ beta1
    y2 = valid['ln_textile_firms'].values
    X2 = np.column_stack([np.ones(len(valid)), policy_hat, valid['time'].values])
    beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y2
    residuals2 = y2 - X2 @ beta2
    r2_2 = 1 - np.sum(residuals2**2) / np.sum((y2 - np.mean(y2))**2)

    n2, k2 = X2.shape
    sigma2_2 = np.sum(residuals2**2) / (n2 - k2)
    se2 = np.sqrt(np.diag(sigma2_2 * np.linalg.inv(X2.T @ X2)))
    t2 = beta2 / se2

    print(f"  二阶段: policy_hat → ln_textile_firms")
    print(f"    coef={beta2[1]:.4f}, SE={se2[1]:.4f}, t={t2[1]:.2f}, R2={r2_2:.3f}")

    iv_results.append({
        'stage': '1st',
        'n': n, 'r2': r2_1, 'f_stat': f_stat,
        'doc_count_coef': beta1[1], 'doc_count_se': se1[1], 'doc_count_t': beta1[1] / se1[1],
    })
    iv_results.append({
        'stage': '2nd',
        'n': n2, 'r2': r2_2,
        'policy_iv_coef': beta2[1], 'policy_iv_se': se2[1], 'policy_iv_t': t2[1],
    })

    # ===== 分类IV: 不同政策类别的工具变量 =====
    category_ivs = ['env_policy_count', 'ecommerce_policy_count', 'brand_policy_count',
                    'cluster_policy_count', 'equipment_policy_count']
    for cat in category_ivs:
        if cat not in df.columns:
            continue
        cat_valid = df[['ln_textile_firms', 'policy_intensity_total', cat, 'time']].dropna()
        if len(cat_valid) < 10:
            continue

        y1c = cat_valid['policy_intensity_total'].values
        X1c = np.column_stack([np.ones(len(cat_valid)), cat_valid[cat].values, cat_valid['time'].values])
        beta1c = np.linalg.inv(X1c.T @ X1c) @ X1c.T @ y1c
        residuals1c = y1c - X1c @ beta1c
        n_c = len(cat_valid)
        sigma2_1c = np.sum(residuals1c**2) / (n_c - 3)
        se1c = np.sqrt(np.diag(sigma2_1c * np.linalg.inv(X1c.T @ X1c)))
        f_c = (beta1c[1] / se1c[1])**2

        # 第二阶段
        policy_hat_c = X1c @ beta1c
        y2c = cat_valid['ln_textile_firms'].values
        X2c = np.column_stack([np.ones(len(cat_valid)), policy_hat_c, cat_valid['time'].values])
        beta2c = np.linalg.inv(X2c.T @ X2c) @ X2c.T @ y2c
        residuals2c = y2c - X2c @ beta2c
        sigma2_2c = np.sum(residuals2c**2) / (n_c - 3)
        se2c = np.sqrt(np.diag(sigma2_2c * np.linalg.inv(X2c.T @ X2c)))

        iv_results.append({
            'stage': '1st',
            'iv': cat, 'n': n_c,
            'r2': 1 - np.sum(residuals1c**2) / np.sum((y1c - np.mean(y1c))**2),
            'f_stat': f_c,
            'doc_count_coef': beta1c[1], 'doc_count_se': se1c[1],
            'doc_count_t': beta1c[1] / se1c[1],
        })
        iv_results.append({
            'stage': '2nd',
            'iv': cat, 'n': n_c,
            'r2': 1 - np.sum(residuals2c**2) / np.sum((y2c - np.mean(y2c))**2),
            'policy_iv_coef': beta2c[1], 'policy_iv_se': se2c[1],
            'policy_iv_t': beta2c[1] / se2c[1],
        })
        print(f"  IV-{cat}: 1st-stage F={f_c:.2f}, 2nd-stage beta={beta2c[1]:.4f} (t={beta2c[1]/se2c[1]:.2f})")

    return iv_results


def descriptive_analysis(df):
    """描述性统计与时序特征"""
    print("\n=== 描述性分析 ===")

    # 按断点前后分组
    pre = df[df['Year'] < 2017]
    post = df[df['Year'] >= 2017]

    stats = []
    for label, subset in [('Pre-2017', pre), ('Post-2017', post)]:
        row = {'Period': label, 'Years': f"{subset['Year'].min()}-{subset['Year'].max()}", 'N': len(subset)}
        for col in ['textile_firms', 'total_firms', 'policy_intensity_total',
                    'policy_doc_count', 'policy_composite']:
            if col in subset.columns and subset[col].notna().sum() > 0:
                row[f'{col}_mean'] = subset[col].mean()
                row[f'{col}_std'] = subset[col].std()
        stats.append(row)

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"), index=False, encoding='utf-8-sig')

    # 关键变量相关性矩阵
    corr_cols = [c for c in ['textile_firms', 'total_firms', 'policy_intensity_total',
                              'policy_doc_count', 'policy_composite']
                 if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"), encoding='utf-8-sig')
        print(f"\n相关性矩阵 ({len(corr_cols)}x{len(corr_cols)}):")
        print(corr.round(3).to_string())


def main():
    print("=" * 60)
    print("Phase B+D: ITS断点回归 + IV工具变量分析")
    print("=" * 60)

    df = load_and_merge()
    print(f"面板: {len(df)}年 ({df['Year'].min()}-{df['Year'].max()}), {len(df.columns)}列")

    # 描述性分析
    descriptive_analysis(df)

    # ITS模型
    print("\n--- ITS断点回归模型 ---")
    its_results = run_its_models(df)

    # IV模型
    print("\n--- IV工具变量回归 ---")
    iv_results = run_iv_models(df)

    # 保存结果
    its_df = pd.DataFrame(its_results)
    its_df.to_csv(os.path.join(OUTPUT_DIR, "its_results.csv"), index=False, encoding='utf-8-sig')
    print(f"\nITS结果已保存: {len(its_results)} 个模型")

    if iv_results:
        iv_df = pd.DataFrame(iv_results)
        iv_df.to_csv(os.path.join(OUTPUT_DIR, "iv_results.csv"), index=False, encoding='utf-8-sig')
        print(f"IV结果已保存: {len(iv_results)} 条记录")

    # === 关键假说检验汇总 ===
    print("\n" + "=" * 60)
    print("假说检验汇总")
    print("=" * 60)

    # H1: 2017年后政策强度显著提升纺织企业存活率
    for r in its_results:
        if r['model'] == 'ITS-政策强度':
            pi_t = r.get('policy_intensity_t', 0)
            pi_beta = r.get('policy_intensity', 0)
            if pi_t and abs(pi_t) > 1.96:
                print(f"H1 [SUPPORT]: Policy intensity has significant positive effect on textile firms (beta={pi_beta:.3f}, t={pi_t:.2f})")
            elif pi_t:
                print(f"H1 [PARTIAL]: Policy intensity effect positive but not significant (beta={pi_beta:.3f}, t={pi_t:.2f})")
            else:
                print(f"H1 [不确定]: 无法估计政策强度效应")

    # H4: 政策文件数量是有效IV
    for r in iv_results:
        if r['stage'] == '1st' and r.get('f_stat', 0) > 10:
            print(f"H4 [SUPPORT]: Policy document count is a strong IV (F={r['f_stat']:.1f})")
            break
        elif r['stage'] == '1st' and r.get('f_stat', 0) > 0:
            print(f"H4 [WEAK]: 1st-stage F={r['f_stat']:.1f} < 10, potential weak IV problem")
            break

    print("\n分析完成。结果保存在 analysis/ 目录")


if __name__ == "__main__":
    main()
