"""Pipeline 5: 河北省各县宏观面板提取"""
import pandas as pd
import numpy as np
import os

NATIONAL_CSV = "data/panel/中国县域统计面板数据_最终版.csv"
OUTPUT = "output/hebei_counties_macro_panel.csv"

# 核心变量：经济、产业、财政、人口
CORE_VARS = {
    'gdp': 'gdp',
    'above_scale_ind_firms': 'above_scale_ind_firms',
    'secondary_industry_va': 'secondary_industry_va',
    'primary_industry_va': 'primary_industry_va',
    'tertiary_industry_va': 'tertiary_industry_va',
    'local_fiscal_revenue': 'local_fiscal_revenue',
    'local_fiscal_expenditure': 'local_fiscal_expenditure',
    'public_fiscal_revenue': 'public_fiscal_revenue',
    'fixed_asset_invest': 'fixed_asset_invest',
    'registered_pop': 'registered_pop',
    'total_pop': 'total_pop',
    'rural_pop': 'rural_pop',
    'deposit_balance': 'deposit_balance',
    'loan_balance': 'loan_balance',
    'grain_output': 'grain_output',
    'meat_output': 'meat_output',
}


def main():
    df = pd.read_csv(NATIONAL_CSV, encoding='utf-8-sig')
    print(f"全国数据: {len(df)} 行 x {len(df.columns)} 列")

    # 筛选河北省
    hebei = df[df['province'].astype(str).str.contains('河北', na=False)].copy()
    print(f"河北省: {len(hebei)} 行, {hebei['county'].nunique()} 县")

    # 标准化列名
    hebei.rename(columns={'year': 'Year'}, inplace=True)
    hebei['Year'] = hebei['Year'].astype(int)

    # 提取核心变量（只保留存在的列）
    available_vars = {}
    for short_name, col_name in CORE_VARS.items():
        if col_name in hebei.columns:
            available_vars[short_name] = col_name

    print(f"可用核心变量: {len(available_vars)}/{len(CORE_VARS)}")

    # 构建输出
    keep_cols = ['Year', 'province', 'county'] + list(available_vars.values())
    out = hebei[keep_cols].copy()
    out.columns = ['Year', 'province', 'county'] + list(available_vars.keys())

    # 排序
    out = out.sort_values(['county', 'Year']).reset_index(drop=True)

    # === 年份缺失率分析 ===
    print("\n=== 各年份有效观测数 ===")
    year_stats = pd.DataFrame({
        'Year': out['Year'].unique(),
        'Counties': out.groupby('Year')['county'].nunique().values
    })
    year_stats = year_stats.sort_values('Year')
    print(year_stats.to_string(index=False))

    # === 变量缺失率 ===
    print("\n=== 变量完整率 ===")
    for var in available_vars.keys():
        complete = out[var].notna().sum()
        pct = complete / len(out) * 100
        print(f"  {var}: {complete}/{len(out)} ({pct:.1f}%)")

    os.makedirs("output", exist_ok=True)
    out.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n已保存至 {OUTPUT}")
    print(f"输出: {len(out)} 行 x {len(out.columns)} 列")
    print(f"年份范围: {out['Year'].min()}-{out['Year'].max()}")
    print(f"县数量: {out['county'].nunique()}")


if __name__ == "__main__":
    main()
