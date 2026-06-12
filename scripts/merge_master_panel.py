"""
模块4 v2: 最终数据融合 - 基于实际数据资产的重新设计
合并策略：
  1. 以 policy_scores_panel.csv 的 Gaoyang 部分为骨架 (2000-2026)
  2. Left-join gaoyang_enterprise_registration.csv (1979-2024 → 对齐到 2000-2024)
  3. Left-join 高阳县面板数据.csv (2003-2023 宏观指标)
  4. 缺失值线性插值
输出: output/master_panel_data_v2.csv
"""
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"

def load_csv_safe(file_path, name):
    if not os.path.exists(file_path):
        print(f"  [跳过] 未找到 {name}: {file_path}")
        return None
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"  [加载] {name}: {len(df)} 行 x {len(df.columns)} 列")
    return df

def merge_master_panel_v2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载政策评分（核心骨架） ===
    df_policy = load_csv_safe("output/policy_scores_panel.csv", "政策评分面板")
    if df_policy is None:
        print("错误: policy_scores_panel.csv 不存在")
        return

    df_policy['County_Code'] = df_policy['County_Code'].astype(str)
    df_policy['Year'] = df_policy['Year'].astype(int)

    # 提取高阳县
    df_gy = df_policy[df_policy['County_Code'] == '130628'].copy()
    print(f"  高阳县政策评分: {len(df_gy)} 行 ({df_gy['Year'].min()}-{df_gy['Year'].max()})")

    # === 2. 加载企业注册数据 ===
    df_ent = load_csv_safe("output/gaoyang_enterprise_registration.csv", "企业注册-高阳")
    if df_ent is not None:
        df_ent['Year'] = df_ent['Year'].astype(int)
        df_ent['County_Code'] = df_ent['County_Code'].astype(str)
        # 只保留 2000+ 年份
        df_ent = df_ent[df_ent['Year'] >= 2000].copy()
        print(f"  企业注册(2000+): {len(df_ent)} 行")

        df_gy = df_gy.merge(df_ent, on=['County_Code', 'Year'], how='left', suffixes=('', '_ent'))
        print(f"  合并后: {len(df_gy)} 行")

    # === 3. 加载高阳县宏观面板 ===
    df_macro = load_csv_safe("data/panel/高阳县面板数据.csv", "高阳宏观面板")
    if df_macro is not None:
        # 重命名：中文year列 → Year
        if 'year' in df_macro.columns:
            df_macro.rename(columns={'year': 'Year'}, inplace=True)
        df_macro['Year'] = df_macro['Year'].astype(int)
        df_macro['County_Code'] = '130628'

        # 去掉 province, county 冗余列
        drop_cols = [c for c in ['province', 'county'] if c in df_macro.columns]
        if drop_cols:
            df_macro = df_macro.drop(columns=drop_cols)

        # 只保留有数据的列（非全NaN）
        valid_cols = ['County_Code', 'Year']
        for c in df_macro.columns:
            if c in valid_cols:
                continue
            non_null = df_macro[c].notna().sum()
            if non_null > 0:
                valid_cols.append(c)
        df_macro = df_macro[valid_cols]
        print(f"  宏观面板有效列: {len(valid_cols)} (总{len(df_macro.columns)})")

        df_gy = df_gy.merge(df_macro, on=['County_Code', 'Year'], how='left', suffixes=('', '_macro'))
        print(f"  合并后: {len(df_gy)} 行")

    # === 4. 加载纺织指数数据 ===
    df_textile = load_csv_safe("output/textile_indices_annual.csv", "纺织指数")
    if df_textile is not None:
        if 'year' in df_textile.columns:
            df_textile.rename(columns={'year': 'Year'}, inplace=True)
        df_textile['Year'] = df_textile['Year'].astype(int)
        df_textile['County_Code'] = '130628'
        
        drop_cols = [c for c in ['county_code', 'province', 'county', 'city'] if c in df_textile.columns]
        if drop_cols:
            df_textile = df_textile.drop(columns=drop_cols)
        
        valid_cols = ['County_Code', 'Year']
        for c in df_textile.columns:
            if c in valid_cols:
                continue
            non_null = df_textile[c].notna().sum()
            if non_null > 0:
                valid_cols.append(c)
        df_textile = df_textile[valid_cols]
        
        # 过滤到2000+
        df_textile = df_textile[df_textile['Year'] >= 2000].copy()
        print(f"  纺织指数(2000+): {len(df_textile)} 行, {len(valid_cols)-2} 列")
        
        df_gy = df_gy.merge(df_textile, on=['County_Code', 'Year'], how='left', suffixes=('', '_textile'))

    # === 5. 缺失值处理 ===
    print(f"\n=== 缺失值处理 ===")
    before_na = df_gy.isna().sum().sum()
    print(f"  处理前总缺失: {before_na}")
    
    # 定义稀疏变量（原始数据点很少，不能插值/填充）
    sparse_threshold = 5  # 非空值<=5的变量视为稀疏
    sparse_vars = []
    numeric_cols = df_gy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        non_null_count = df_gy[col].notna().sum()
        if non_null_count <= sparse_threshold:
            sparse_vars.append(col)
            # 稀疏变量：保持原始NaN，不做任何插值或填充
            continue
        else:
            # 非稀疏变量：线性插值（仅内部）
            df_gy[col] = df_gy[col].interpolate(method='linear', limit_direction='forward', limit_area='inside')
            # 前端/后端仍缺失的，保留NaN（不用ffill/bfill）
    
    print(f"  稀疏变量（保持NaN）: {sparse_vars}")
    
    # 对 County_Code, Year, total_chunks, sample_reliable, keyword_passed, filter_passed, scored 这些标识列不插值
    id_cols = ['County_Code', 'Year', 'total_chunks', 'sample_reliable', 'keyword_passed', 'filter_passed', 'scored']
    id_cols_present = [c for c in id_cols if c in df_gy.columns]

    after_na = df_gy.isna().sum().sum()
    print(f"  插值后总缺失: {after_na}")
    
    # 不再使用ffill/bfill！保留NaN以反映真实数据可用性
    final_na = df_gy.isna().sum().sum()
    print(f"  最终缺失: {final_na}（保留原始缺失，不填充常值）")

    # === 6. 变量统计 ===
    print(f"\n=== 最终面板 ===")
    print(f"  行数: {len(df_gy)} ({df_gy['Year'].min()}-{df_gy['Year'].max()})")
    print(f"  列数: {len(df_gy.columns)}")
    print(f"  县代码: {df_gy['County_Code'].unique().tolist()}")

    # 列出关键变量组
    policy_cols = [c for c in df_gy.columns if 'index' in c or 'intensity' in c or 'mix' in c]
    ent_cols = [c for c in df_gy.columns if 'firm' in c or 'textile' in c.lower()]
    macro_cols = [c for c in df_gy.columns if c not in policy_cols and c not in ent_cols and c not in ['County_Code', 'Year', 'total_chunks', 'sample_reliable', 'keyword_passed', 'filter_passed', 'scored']]
    print(f"  政策变量: {len(policy_cols)}")
    print(f"  企业注册变量: {len(ent_cols)}")
    print(f"  宏观变量: {len(macro_cols)}")
    
    # 检查GDP是否有变异
    if 'gdp' in df_gy.columns:
        gdp_std = df_gy['gdp'].std()
        gdp_nunique = df_gy['gdp'].nunique()
        gdp_na = df_gy['gdp'].isna().sum()
        print(f"\n  GDP检查: 标准差={gdp_std:.2f}, 不同值={gdp_nunique}, 缺失数={gdp_na}")
        if gdp_std == 0:
            print(f"    ⚠️ GDP无时间变异！")
        else:
            print(f"    ✅ GDP有时间变异，可用于回归")

    out_path = os.path.join(OUTPUT_DIR, "master_panel_data_v2.csv")
    df_gy.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  已保存至 {out_path}")

    return df_gy

if __name__ == "__main__":
    merge_master_panel_v2()
