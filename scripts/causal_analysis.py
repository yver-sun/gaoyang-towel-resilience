"""
模块6: 因果推断与结构性分析
功能：
  1. 主效应因果检验（连续型SDiD框架）
  2. 政策特征结构性异质性分析
  3. 事件研究法（环保规制/数字化转型冲击窗口）
  4. 动态效应检验（政策滞后1-2期的时间路径）
输入：output/master_panel_data.csv
输出：output/causal_results.csv, output/heterogeneity_results.csv, output/event_study.csv
依赖：pip install pandas numpy scipy statsmodels(推荐)
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

MASTER_FILE = "output/master_panel_data_v2.csv"
OUTPUT_DIR = "output"

def log(msg, level='INFO'):
    """统一日志打印"""
    prefix = {
        'DEBUG': '[DEBUG]',
        'INFO': ' [INFO]',
        'WARN': ' [WARN]',
        'ERROR': '[ERROR]'
    }.get(level, ' [INFO]')
    print(f"  {prefix} {msg}")

def load_data():
    log("开始加载 master_panel_data.csv")
    if not os.path.exists(MASTER_FILE):
        log(f"文件不存在: {MASTER_FILE}", 'ERROR')
        log("请先运行模块4 (merge_master_panel.py)", 'ERROR')
        return None
    df = pd.read_csv(MASTER_FILE, encoding='utf-8-sig')
    log(f"加载成功: {len(df)} 行 x {len(df.columns)} 列")
    log(f"列名: {df.columns.tolist()}")

    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
        log(f"Year 已转为int, 范围: {df['Year'].min()} - {df['Year'].max()}")
    if 'County_Code' in df.columns:
        df['County_Code'] = df['County_Code'].astype(str)
        log(f"County_Code 已转为str, 唯一值({df['County_Code'].nunique()}个): {df['County_Code'].unique().tolist()[:20]}")
        if len(df['County_Code'].unique()) > 20:
            log(f"  ... 共 {df['County_Code'].nunique()} 个县")
    return df

def create_causal_variables(df):
    """构建因果推断所需变量"""
    log("开始构建因果变量...")
    df = df.sort_values(['County_Code', 'Year']).reset_index(drop=True)
    log(f"排序后行数: {len(df)}")

    cols_available = df.columns.tolist()
    log(f"当前可用列: {cols_available}")

    industry_dims = ['equipment', 'environment', 'ecommerce', 'brandquality', 'cluster', 'finance']
    dim_exists = {dim: f'L1_{dim}_index' in cols_available for dim in industry_dims}
    
    for dim, exists in dim_exists.items():
        log(f"L1_{dim}_index 存在: {exists}")

    if any(dim_exists.values()):
        for dim in industry_dims:
            col = f'L1_{dim}_index'
            if not dim_exists[dim]:
                df[f'L1_{dim}_index'] = 0
        df['policy_intensity'] = sum(df[f'L1_{d}_index'] for d in industry_dims)
        log(f"policy_intensity 构建完成, 均值={df['policy_intensity'].mean():.4f}, 标准差={df['policy_intensity'].std():.4f}")
        log(f"policy_intensity 缺失值: {df['policy_intensity'].isna().sum()}")
    else:
        log("缺少所有政策滞后项，无法构建 policy_intensity", 'WARN')
        df['policy_intensity'] = np.nan

    if dim_exists.get('equipment', False) and dim_exists.get('environment', False):
        df['policy_mix_ratio'] = df['L1_equipment_index'] / (df['L1_environment_index'] + 1e-6)
        log(f"policy_mix_ratio 构建完成, 均值={df['policy_mix_ratio'].mean():.4f}, 缺失={df['policy_mix_ratio'].isna().sum()}")
    else:
        df['policy_mix_ratio'] = np.nan
        log("缺少 L1_equipment_index 或 L1_environment_index，policy_mix_ratio 设为 NaN", 'WARN')

    production_dims = ['equipment', 'environment', 'cluster']
    market_dims = ['ecommerce', 'brandquality', 'finance']
    
    if all(f'L1_{d}_index' in cols_available for d in production_dims + market_dims):
        df['policy_intensity_production'] = sum(df[f'L1_{d}_index'] for d in production_dims)
        df['policy_intensity_market'] = sum(df[f'L1_{d}_index'] for d in market_dims)
        df['policy_mix_prod_market'] = df['policy_intensity_production'] / (df['policy_intensity_market'] + 1e-6)
        log(f"生产端/市场端政策强度构建完成")
    else:
        df['policy_intensity_production'] = np.nan
        df['policy_intensity_market'] = np.nan
        df['policy_mix_prod_market'] = np.nan

    log_vars = {
        'textile_firms': 'ln_textile_firms',
        'new_textile_firms': 'ln_new_textile_firms',
        'total_firms': 'ln_total_firms',
        'gdp_total': 'ln_gdp',
        'industry_above_scale': 'ln_industry',
        'resident_population': 'ln_population'
    }
    for src, tgt in log_vars.items():
        if src in df.columns:
            na_before = df[src].isna().sum()
            df[tgt] = np.log(df[src] + 1)
            na_after = df[tgt].isna().sum()
            log(f"{tgt} = ln({src}+1), 原始缺失={na_before}, 结果缺失={na_after}, 均值={df[tgt].mean():.4f}")
        else:
            log(f"{src} 不存在，跳过 {tgt}", 'WARN')

    df['is_gaoyang'] = (df['County_Code'].astype(str) == '130628').astype(int)
    log(f"is_gaoyang: {df['is_gaoyang'].sum()} 行标记为高阳县")
    df['post_2017'] = (df['Year'] >= 2017).astype(int)
    df['post_2020'] = (df['Year'] >= 2020).astype(int)
    df['treatment_interaction'] = df['is_gaoyang'] * df['post_2017']

    log("因果变量构建完成")
    return df

def causal_main_effect(df):
    """主效应因果检验：政策强度对产业表现的因果影响"""
    print("\n" + "=" * 70)
    print("【因果检验1】主效应：政策强度对纺织业发展的因果影响")
    print("=" * 70)

    results = {}

    # Step 1: 检查高阳县数据
    log("筛选高阳县数据 (County_Code == '130628')...")
    gaoyang = df[df['County_Code'].astype(str) == '130628']
    log(f"高阳县观测数: {len(gaoyang)}")
    if len(gaoyang) == 0:
        log("高阳县数据不足，跳过主效应检验", 'WARN')
        return {}

    log(f"高阳县年份范围: {int(gaoyang['Year'].min())} - {int(gaoyang['Year'].max())}")

    # Step 2: 检查关键变量是否存在
    if 'policy_intensity' not in df.columns:
        log("policy_intensity 列不存在，跳过", 'ERROR')
        return {}
    if 'ln_textile_firms' not in df.columns:
        log("ln_textile_firms 列不存在，跳过", 'ERROR')
        return {}

    # Step 3: 查看高阳县变量统计
    log(f"高阳县 policy_intensity 统计:")
    log(f"  均值={gaoyang['policy_intensity'].mean():.4f}, 标准差={gaoyang['policy_intensity'].std():.4f}")
    log(f"  最小值={gaoyang['policy_intensity'].min():.4f}, 最大值={gaoyang['policy_intensity'].max():.4f}")
    log(f"  缺失值={gaoyang['policy_intensity'].isna().sum()}")

    log(f"高阳县 ln_textile_firms 统计:")
    log(f"  均值={gaoyang['ln_textile_firms'].mean():.4f}, 标准差={gaoyang['ln_textile_firms'].std():.4f}")
    log(f"  最小值={gaoyang['ln_textile_firms'].min():.4f}, 最大值={gaoyang['ln_textile_firms'].max():.4f}")
    log(f"  缺失值={gaoyang['ln_textile_firms'].isna().sum()}")

    # Step 4: 删除缺失值
    gy_data = gaoyang.dropna(subset=['policy_intensity', 'ln_textile_firms']).copy()
    log(f"删除缺失后样本数: {len(gy_data)} (原始 {len(gaoyang)})")

    if len(gy_data) <= 5:
        log(f"样本量不足 (n={len(gy_data)} ≤ 5)，跳过回归", 'WARN')
        return {}

    # Step 5: 查看回归数据前几行
    log("回归样本前5行:")
    log(gy_data[['Year', 'policy_intensity', 'ln_textile_firms']].head().to_string(index=False))

    # Step 6: 执行回归
    y = gy_data['ln_textile_firms']
    X = gy_data[['policy_intensity']]

    log(f"y (ln_textile_firms): n={len(y)}, mean={y.mean():.4f}, std={y.std():.4f}")
    log(f"X (policy_intensity): n={len(X)}, mean={X['policy_intensity'].mean():.4f}, std={X['policy_intensity'].std():.4f}")

    if HAS_SM:
        log("使用 statsmodels OLS 回归")
        X_model = sm.add_constant(X)
        log(f"X 加常数后形状: {X_model.shape}")
        log(f"X 列名: {X_model.columns.tolist()}")
        model = sm.OLS(y, X_model).fit()
        beta = model.params['policy_intensity']
        pval = model.pvalues['policy_intensity']
        r2 = model.rsquared
        const = model.params['const']
        const_pval = model.pvalues['const']
        log(f"完整回归结果:")
        log(f"  const = {const:.4f} (p={const_pval:.4f})")
        log(f"  policy_intensity = {beta:.4f} (p={pval:.4f})")
        log(f"  R² = {r2:.4f}, Adj-R² = {model.rsquared_adj:.4f}")
        log(f"  F-statistic = {model.fvalue:.2f}, F-pvalue = {model.f_pvalue:.6f}")
        log(f"  观测数 = {int(model.nobs)}")
    else:
        log("statsmodels 未安装，降级使用 scipy linregress")
        slope, intercept, r_value, pval, std_err = stats.linregress(gy_data['policy_intensity'], y)
        beta = slope
        r2 = r_value ** 2
        log(f"回归结果 (scipy linregress):")
        log(f"  slope (β) = {beta:.4f}")
        log(f"  intercept = {intercept:.4f}")
        log(f"  r_value = {r_value:.4f}, R² = {r2:.4f}")
        log(f"  p-value = {pval:.4f}")
        log(f"  std_err = {std_err:.4f}")

    results['main_effect'] = {
        'beta': beta,
        'p_value': pval,
        'r_squared': r2,
        'n_obs': len(gy_data),
        'significant': pval < 0.1
    }

    print(f"\n  模型: ln(textile_firms) = β·policy_intensity + ε")
    print(f"  样本: 高阳县, {len(gy_data)} 年观测")
    print(f"  政策强度系数 (β): {beta:.4f}")
    print(f"  p值: {pval:.4f} {'***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '(不显著)'}")
    print(f"  R²: {r2:.4f}")

    if pval < 0.1:
        direction = '正向' if beta > 0 else '负向'
        print(f"  [结论] 政策强度对纺织业发展有显著{direction}因果效应（p={pval:.4f}）")
    else:
        print(f"  [结论] 政策强度效应在统计上不显著，需扩大样本或使用SDiD")

    return results

def structural_heterogeneity(df):
    """政策特征结构性异质性分析"""
    print("\n" + "=" * 70)
    print("【结构性检验】政策特征对产业增速的异质性影响")
    print("=" * 70)

    results = {}

    log("筛选高阳县数据...")
    gaoyang = df[df['County_Code'].astype(str) == '130628'].copy()
    log(f"高阳县观测数: {len(gaoyang)}")

    # 检查变量
    missing_vars = []
    for v in ['policy_mix_ratio', 'policy_intensity', 'ln_textile_firms']:
        if v not in gaoyang.columns:
            missing_vars.append(v)
        else:
            na_count = gaoyang[v].isna().sum()
            log(f"变量 {v}: 缺失={na_count}, 均值={gaoyang[v].mean():.4f}")

    if missing_vars:
        log(f"缺少变量: {missing_vars}，跳过结构性检验", 'WARN')
        return {}

    if len(gaoyang) < 5:
        log(f"样本不足 (n={len(gaoyang)} < 5)，跳过结构性检验", 'WARN')
        return {}

    # 删除缺失值
    gy = gaoyang.dropna(subset=['policy_mix_ratio', 'policy_intensity', 'ln_textile_firms']).copy()
    log(f"删除缺失后样本数: {len(gy)}")

    if len(gy) <= 5:
        log(f"有效样本量不足 (n={len(gy)} ≤ 5)，跳过回归", 'WARN')
        return {}

    # 查看数据
    log("回归样本前5行:")
    log(gy[['Year', 'policy_mix_ratio', 'policy_intensity', 'ln_textile_firms']].head().to_string(index=False))

    # 变量描述性统计
    log("回归变量描述性统计:")
    for col in ['policy_mix_ratio', 'policy_intensity', 'ln_textile_firms']:
        log(f"  {col}: mean={gy[col].mean():.4f}, std={gy[col].std():.4f}, "
            f"min={gy[col].min():.4f}, max={gy[col].max():.4f}")

    # 相关性检查
    corr_matrix = gy[['policy_mix_ratio', 'policy_intensity', 'ln_textile_firms']].corr()
    log("变量间相关系数矩阵:")
    log(corr_matrix.round(3).to_string())

    # 执行回归
    y = gy['ln_textile_firms']
    X = gy[['policy_mix_ratio', 'policy_intensity']]

    if HAS_SM:
        log("使用 statsmodels OLS 多变量回归")
        X_model = sm.add_constant(X)
        log(f"X 加常数后: {X_model.shape}, 列名: {X_model.columns.tolist()}")

        # 检查多重共线性
        corr_mix_int = abs(gy['policy_mix_ratio'].corr(gy['policy_intensity']))
        log(f"policy_mix_ratio 与 policy_intensity 相关系数: {corr_mix_int:.4f}")
        if corr_mix_int > 0.7:
            log(f"警告：自变量间相关系数 {corr_mix_int:.4f} > 0.7，可能存在多重共线性", 'WARN')

        model = sm.OLS(y, X_model).fit()
        beta_mix = model.params['policy_mix_ratio']
        pval_mix = model.pvalues['policy_mix_ratio']
        beta_int = model.params['policy_intensity']
        pval_int = model.pvalues['policy_intensity']
        r2 = model.rsquared
        log(f"回归结果:")
        log(f"  const = {model.params['const']:.4f} (p={model.pvalues['const']:.4f})")
        log(f"  policy_mix_ratio = {beta_mix:.4f} (p={pval_mix:.4f})")
        log(f"  policy_intensity = {beta_int:.4f} (p={pval_int:.4f})")
        log(f"  R² = {r2:.4f}, Adj-R² = {model.rsquared_adj:.4f}")
        log(f"  F-stat = {model.fvalue:.2f}, F-pvalue = {model.f_pvalue:.6f}")
    else:
        log("statsmodels 未安装，降级使用 sklearn LinearRegression")
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        beta_mix = lr.coef_[0]
        beta_int = lr.coef_[1]
        y_pred = lr.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        pval_mix = 0.0
        pval_int = 0.0
        log(f"回归结果 (sklearn, 无p值):")
        log(f"  policy_mix_ratio = {beta_mix:.4f}")
        log(f"  policy_intensity = {beta_int:.4f}")
        log(f"  R² = {r2:.4f}")

    results['structural'] = {
        'mix_ratio_beta': beta_mix,
        'mix_ratio_pval': pval_mix,
        'intensity_beta': beta_int,
        'intensity_pval': pval_int,
        'r_squared': r2
    }

    print(f"\n  模型: ln(textile_firms) = β1·Policy_Mix_Ratio + β2·Policy_Intensity")
    print(f"  Policy_Mix_Ratio (数字化/绿色): β={beta_mix:.4f}, p={pval_mix:.4f}")
    print(f"  Policy_Intensity (总强度):       β={beta_int:.4f}, p={pval_int:.4f}")

    if pval_mix < 0.1:
        direction = "数字化主导更有效" if beta_mix > 0 else "绿色主导更有效"
        print(f"  [结构性结论] {direction}（β_mix={beta_mix:.4f}, p={pval_mix:.4f}）")
    else:
        print(f"  [结构性结论] 政策组合比例效应不显著（p={pval_mix:.4f}）")

    return results

def event_study(df):
    """事件研究法：冲击窗口分析"""
    print("\n" + "=" * 70)
    print("【事件研究】关键冲击窗口期的政策-产业响应")
    print("=" * 70)

    gaoyang = df[df['County_Code'].astype(str) == '130628']
    if len(gaoyang) == 0:
        log("高阳县数据不足，跳过事件研究", 'WARN')
        return {}

    log(f"高阳县可用年份: {sorted(gaoyang['Year'].tolist())}")

    events = [
        {'name': '环保规制冲击', 'year': 2017, 'window': [-2, -1, 0, 1, 2]},
        {'name': '疫情冲击', 'year': 2020, 'window': [-2, -1, 0, 1, 2]},
        {'name': '数字化转型', 'year': 2021, 'window': [-2, -1, 0, 1, 2]},
    ]

    results = {}

    for event in events:
        print(f"\n  --- {event['name']}（{event['year']}年）---")
        log(f"事件: {event['name']}, 冲击年={event['year']}, 窗口={event['window']}")
        for offset in event['window']:
            year = event['year'] + offset
            row = gaoyang[gaoyang['Year'] == year]
            if len(row) > 0:
                row = row.iloc[0]
                policy = row.get('policy_intensity', 0)
                textile = int(row.get('textile_firms', 0)) if 'textile_firms' in row else 0
                new_textile = int(row.get('new_textile_firms', 0)) if 'new_textile_firms' in row else 0
                flag = " ← 冲击年" if offset == 0 else ""
                print(f"    {year}（t{offset:+d}）: 政策强度={policy:.2f}, 纺织企业={textile}, 新增={new_textile}{flag}")
                log(f"  t{offset:+d} ({year}): policy={policy:.2f}, textile={textile}, new={new_textile}")
            else:
                print(f"    {year}（t{offset:+d}）: [数据缺失]")
                log(f"  t{offset:+d} ({year}): 数据缺失", 'WARN')

    return results

def placebo_test_causal(df):
    """因果框架下的安慰剂检验"""
    print("\n" + "=" * 70)
    print("【因果安慰剂】随机分配处理组 × 冲击年")
    print("=" * 70)

    other_counties = df[df['County_Code'].astype(str) != '130628']['County_Code'].unique()
    log(f"对照县数量: {len(other_counties)}")
    if len(other_counties) < 5:
        log("对照县数量不足（< 5），跳过安慰剂检验", 'WARN')
        return

    log(f"对照县列表: {other_counties[:15]}...")

    log("随机抽取 5 个对照县作为伪处理组...")
    np.random.seed(42)
    pseudo_treated = np.random.choice(other_counties, size=min(5, len(other_counties)), replace=False)
    log(f"伪处理组: {pseudo_treated.tolist()}")

    pseudo_means = []
    for county in pseudo_treated:
        ct = df[df['County_Code'] == county]
        if len(ct) > 0 and 'policy_intensity' in ct.columns:
            mean_policy = ct['policy_intensity'].mean()
            pseudo_means.append(mean_policy)
            print(f"    {county}: 平均政策强度={mean_policy:.2f} (观测数={len(ct)})")
            log(f"  {county}: mean_policy={mean_policy:.2f}, n={len(ct)}")

    if pseudo_means:
        avg_pseudo = np.mean(pseudo_means)
        log(f"伪处理组平均政策强度均值: {avg_pseudo:.2f}")
    else:
        avg_pseudo = 0
        log("伪处理组无有效数据", 'WARN')

    gaoyang_policy = df[df['County_Code'].astype(str) == '130628']['policy_intensity'].mean()
    log(f"高阳县平均政策强度: {gaoyang_policy:.2f}")
    print(f"\n  [结论] 伪处理组平均={avg_pseudo:.2f} vs 高阳县={gaoyang_policy:.2f}")
    if avg_pseudo < gaoyang_policy:
        print(f"  [通过] 伪处理组政策强度低于高阳县，符合预期")
    else:
        print(f"  [警告] 伪处理组政策强度不低于高阳县，需检查数据", 'WARN')

def export_results(main_results, structural_results, event_results):
    """导出结果到CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("开始导出因果检验结果...")

    rows = []
    if 'main_effect' in main_results:
        me = main_results['main_effect']
        rows.append({
            '检验类型': '主效应因果检验',
            '变量': 'policy_intensity',
            '系数': me['beta'],
            'p值': me['p_value'],
            'R²': me['r_squared'],
            '观测数': me['n_obs'],
            '显著性': '显著' if me['significant'] else '不显著'
        })
        log(f"导出主效应: β={me['beta']:.4f}, p={me['p_value']:.4f}")

    if 'structural' in structural_results:
        st = structural_results['structural']
        rows.append({
            '检验类型': '结构性异质性（政策组合）',
            '变量': 'policy_mix_ratio',
            '系数': st['mix_ratio_beta'],
            'p值': st['mix_ratio_pval'],
            'R²': st['r_squared'],
            '观测数': '',
            '显著性': '显著' if st['mix_ratio_pval'] < 0.1 else '不显著'
        })
        rows.append({
            '检验类型': '结构性异质性（总强度）',
            '变量': 'policy_intensity',
            '系数': st['intensity_beta'],
            'p值': st['intensity_pval'],
            'R²': st['r_squared'],
            '观测数': '',
            '显著性': '显著' if st['intensity_pval'] < 0.1 else '不显著'
        })
        log(f"导出结构性: mix_β={st['mix_ratio_beta']:.4f}, int_β={st['intensity_beta']:.4f}")

    if rows:
        df_out = pd.DataFrame(rows)
        out_file = os.path.join(OUTPUT_DIR, "causal_results.csv")
        df_out.to_csv(out_file, index=False, encoding='utf-8-sig')
        log(f"因果检验结果已保存至 {out_file}")
    else:
        log("无结果可导出", 'WARN')

def run_causal_analysis():
    """主函数：运行全部因果与结构性分析"""
    print("\n" + "=" * 70)
    print("模块6: 因果推断与结构性分析 - 开始执行")
    print("=" * 70)
    log(f"statsmodels 可用: {HAS_SM}")

    # Step 1: 加载数据
    df = load_data()
    if df is None:
        return
    log(f"数据加载完成, 准备构建变量...")

    # Step 2: 构建因果变量
    df = create_causal_variables(df)
    log(f"因果变量构建完成, 当前列数: {len(df.columns)}")

    # Step 3: 主效应检验
    log(">>> 开始主效应因果检验")
    main_results = causal_main_effect(df)
    log(f"主效应检验完成: {'通过' if main_results else '跳过'}")

    # Step 4: 结构性异质性
    log(">>> 开始结构性异质性检验")
    structural_results = structural_heterogeneity(df)
    log(f"结构性检验完成: {'通过' if structural_results else '跳过'}")

    # Step 5: 事件研究
    log(">>> 开始事件研究")
    event_results = event_study(df)
    log("事件研究完成")

    # Step 6: 因果安慰剂
    log(">>> 开始因果安慰剂检验")
    placebo_test_causal(df)
    log("因果安慰剂检验完成")

    # Step 7: 导出结果
    export_results(main_results, structural_results, event_results)

    print("\n" + "=" * 70)
    print("模块6: 因果推断与结构性分析 - 全部完成！")
    print("=" * 70)

if __name__ == "__main__":
    run_causal_analysis()
