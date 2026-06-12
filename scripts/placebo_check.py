"""
模块5: 安慰剂检验与描述性统计
功能：
  1. 验证Education维度与产业维度的区分度
  2. 输出描述性统计表（均值、标准差、分位数）
  3. 生成变量间相关系数矩阵
  4. 输出Stata回归预备检查报告
输入：output/master_panel_data.csv
输出：output/descriptive_stats.csv, output/correlation_matrix.csv, output/placebo_report.csv
依赖：pip install pandas numpy
"""
import os
import pandas as pd
import numpy as np

MASTER_FILE = "output/master_panel_data.csv"
OUTPUT_DIR = "output"

INDUSTRY_DIMENSIONS = ["equipment_index", "environment_index", "ecommerce_index", 
                       "brandquality_index", "cluster_index", "finance_index"]
PLACEBO_DIMENSIONS = ["education_index"]
ALL_DIMENSIONS = INDUSTRY_DIMENSIONS + PLACEBO_DIMENSIONS

LAG_DIMENSIONS = [f"L1_{d}" for d in ALL_DIMENSIONS] + [f"L2_{d}" for d in ALL_DIMENSIONS]

POLICY_VARS = ALL_DIMENSIONS + [f"L1_{d}" for d in ALL_DIMENSIONS] + [f"L2_{d}" for d in ALL_DIMENSIONS]
MACRO_VARS = ['gdp_total', 'industry_above_scale', 'resident_population', 'import_export',
              'fiscal_revenue', 'fiscal_expenditure', 'finance_deposit', 'finance_loan',
              'fixed_asset_invest', 'retail_sales', 'primary_industry', 'secondary_industry',
              'tertiary_industry', 'urban_income', 'rural_income']
MECHANISM_VARS = ['express_delivery_volume', 'cotton_yarn_throughput', 'centralized_steam_volume',
                  'eco_park_area', 'sewage_treatment_capacity', 'brand_authorized_firms',
                  'textile_enterprises_count', 'above_scale_textile_firms', 'export_volume_textile',
                  'ecommerce_shipments']
ENTERPRISE_VARS = ['total_firms', 'textile_firms', 'textile_ratio', 'new_total_firms',
                   'new_textile_firms', 'textile_firms_ma3']

def load_master_panel():
    if not os.path.exists(MASTER_FILE):
        print(f"错误：未找到 {MASTER_FILE}")
        print("请先运行模块4 (merge_master_panel.py)")
        return None
    df = pd.read_csv(MASTER_FILE, encoding='utf-8-sig')
    print(f"加载面板数据: {len(df)} 行 x {len(df.columns)} 列")
    return df

def placebo_test(df):
    print("\n" + "=" * 60)
    print("【安慰剂检验】Education维度与产业维度相关性")
    print("=" * 60)

    available_dims = [d for d in ALL_DIMENSIONS if d in df.columns]
    corr = df[available_dims].corr()

    corr_file = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_file, encoding='utf-8-sig')
    print(f"\n相关系数矩阵已保存至 {corr_file}")
    print(f"\n{corr.round(3).to_string()}")

    if 'education_index' in corr.columns:
        edu_corr = corr.loc['education_index', [d for d in INDUSTRY_DIMENSIONS if d in corr.columns]]
        max_corr = edu_corr.abs().max()
        print(f"\nEducation与产业维度最高相关系数: {max_corr:.3f}")
        if max_corr < 0.3:
            print("  [通过] 安慰剂维度纯净度良好（< 0.3）")
        else:
            print("  [警告] 相关系数偏高，建议调整Prompt区分度")

def descriptive_statistics(df):
    print("\n" + "=" * 60)
    print("【描述性统计】核心变量概览")
    print("=" * 60)

    key_vars = [v for v in POLICY_VARS + MACRO_VARS + MECHANISM_VARS + ENTERPRISE_VARS if v in df.columns]

    if key_vars:
        desc = df[key_vars].describe()
        desc_file = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
        desc.to_csv(desc_file, encoding='utf-8-sig')
        print(f"\n描述性统计已保存至 {desc_file}")
        print(f"\n{desc.round(2).to_string()}")

def panel_structure(df):
    print("\n" + "=" * 60)
    print("【面板结构】平衡性检查")
    print("=" * 60)

    if 'County_Code' in df.columns and 'Year' in df.columns:
        n_counties = df['County_Code'].nunique()
        n_years = df['Year'].nunique()
        n_obs = len(df)
        balanced = n_counties * n_years == n_obs

        print(f"  县数: {n_counties}")
        print(f"  年份数: {n_years}")
        print(f"  观测数: {n_obs}")
        print(f"  平衡面板: {'是' if balanced else '否'}")

        if not balanced:
            print(f"  期望观测数: {n_counties * n_years}")
            print(f"  缺失观测: {n_counties * n_years - n_obs}")

        year_range = df.groupby('County_Code')['Year'].agg(['min', 'max'])
        print(f"\n  各县年份覆盖:")
        for code, row in year_range.iterrows():
            print(f"    {code}: {int(row['min'])} - {int(row['max'])} ({int(row['max']) - int(row['min']) + 1}年)")

def variance_inflation_check(df):
    print("\n" + "=" * 60)
    print("【共线性检查】自变量相关系数（回归前预警）")
    print("=" * 60)

    available_dims = [f"L1_{d}" for d in ALL_DIMENSIONS if f"L1_{d}" in df.columns]
    if available_dims:
        corr = df[available_dims].corr()
        print(f"\n{corr.round(3).to_string()}")
        for i, col1 in enumerate(available_dims):
            for j, col2 in enumerate(available_dims):
                if i < j:
                    v = abs(corr.loc[col1, col2])
                    if v > 0.7:
                        print(f"  [警告] {col1} 与 {col2} 相关系数 {v:.3f} > 0.7，可能存在多重共线性")

def enterprise_trend_check(df):
    """注册企业趋势检查（仅高阳县）"""
    if 'textile_firms' not in df.columns:
        return

    gaoyang = df[df['County_Code'] == '130628']
    if len(gaoyang) == 0:
        return

    print("\n" + "=" * 60)
    print("【高阳县纺织业企业注册趋势】")
    print("=" * 60)

    for _, row in gaoyang.iterrows():
        y = int(row['Year'])
        tf = int(row.get('textile_firms', 0))
        ratio = row.get('textile_ratio', 0)
        flag = ""
        if y in [2017, 2018]:
            flag = " <- 环保规制期"
        elif y in [2019, 2020]:
            flag = " <- 疫情期"
        elif y in [2023, 2024]:
            flag = " <- 疫后复苏"
        print(f"  {y}: 纺织企业 {tf} 家, 占比 {ratio:.1%}{flag}")

def generate_regression_prep(df):
    print("\n" + "=" * 60)
    print("【Stata回归预备】建议命令")
    print("=" * 60)

    available_dims = [f"L1_{d}" for d in INDUSTRY_DIMENSIONS if f"L1_{d}" in df.columns]
    if available_dims:
        dep_var = "ln_industry_above_scale" if "industry_above_scale" in df.columns else "ln_gdp_total"
        indep_vars = " ".join(available_dims)
        controls = []
        for c in ["resident_population", "fiscal_revenue", "fixed_asset_invest"]:
            if c in df.columns:
                controls.append(f"ln_{c}")
        ctrl_str = " ".join(controls) if controls else ""

        print(f"\n* 基础模型（双向固定效应）:")
        print(f"xtset County_Code Year")
        print(f"reghdfe {dep_var} {indep_vars} {ctrl_str}, absorb(County_Code Year) cluster(County_Code)")

        print(f"\n* 产业韧性检验（纺织企业新增数作为因变量）:")
        print(f"reghdfe ln_new_textile_firms {indep_vars} {ctrl_str}, absorb(County_Code Year) cluster(County_Code)")

        print(f"\n* 机制检验一（环保降本路径）:")
        print(f"reghdfe {dep_var} L1_environment_index L1_cluster_index ln_centralized_steam_volume ln_sewage_treatment_capacity, absorb(County_Code Year) cluster(County_Code)")

        print(f"\n* 机制检验二（数字化品牌路径）:")
        print(f"reghdfe ln_express_delivery_volume L1_ecommerce_index L1_brandquality_index {ctrl_str}, absorb(County_Code Year) cluster(County_Code)")

        print(f"\n* 结构性异质性（设备环保比）:")
        print(f"reghdfe {dep_var} policy_mix_equipment_env L1_equipment_index L1_environment_index {ctrl_str}, absorb(County_Code Year) cluster(County_Code)")

        print(f"\n* SDiD模型（需安装sdid命令）:")
        print(f"sdid {dep_var} {indep_vars} {ctrl_str}, i(County_Code) t(Year) vce(bootstrap)")

def run_all_checks():
    df = load_master_panel()
    if df is None:
        return

    placebo_test(df)
    descriptive_statistics(df)
    panel_structure(df)
    variance_inflation_check(df)
    enterprise_trend_check(df)
    generate_regression_prep(df)

    print("\n" + "=" * 60)
    print("全部检查完成！输出文件清单:")
    print("=" * 60)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.csv'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size:.1f} KB)")

if __name__ == "__main__":
    run_all_checks()
