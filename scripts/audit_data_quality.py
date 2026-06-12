"""
数据质量全面审计脚本
检查:
1. policy_scores_panel.csv - 行数、列数、sample_reliable分布、异常值、高阳县关键年份
2. master_panel_data.csv vs master_panel_data_v2.csv - 对比完整性
3. government_reports/ TXT文件质量检查
"""
import pandas as pd
import os

BASE_DIR = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾'

print("=" * 80)
print("数据质量全面审计报告")
print("=" * 80)

# ============================================================
# 1. 检查 policy_scores_panel.csv
# ============================================================
print("\n" + "=" * 80)
print("1. policy_scores_panel.csv 数据质量检查")
print("=" * 80)

policy_df = pd.read_csv(os.path.join(BASE_DIR, 'output', 'policy_scores_panel.csv'))

print(f"\n[基本统计]")
print(f"  行数: {len(policy_df)}")
print(f"  列数: {len(policy_df.columns)}")
print(f"  列名: {list(policy_df.columns)}")

print(f"\n[sample_reliable 列分布]")
print(policy_df['sample_reliable'].value_counts().sort_index())
print(f"  sample_reliable=1 的比例: {policy_df['sample_reliable'].mean():.2%}")

# 检查各维度得分异常值 (>200 或极端大)
print(f"\n[各维度得分异常值检查 (>200)]")
score_cols = ['equipment_index', 'environment_index', 'ecommerce_index', 
              'brandquality_index', 'cluster_index', 'finance_index', 'education_index']
anomalies = {}
for col in score_cols:
    if col in policy_df.columns:
        extreme = policy_df[policy_df[col] > 200]
        if len(extreme) > 0:
            anomalies[col] = len(extreme)
            print(f"  {col}: 发现 {len(extreme)} 行 > 200")
            print(f"    最大值: {policy_df[col].max():.2f}")
            print(f"    示例: {extreme[['County_Code', 'Year', col]].head(3).to_string(index=False)}")
        else:
            print(f"  {col}: 正常 (max={policy_df[col].max():.2f})")

# policy_mix_equipment_env 极端值
print(f"\n[policy_mix_equipment_env 极端值检查]")
if 'policy_mix_equipment_env' in policy_df.columns:
    col_data = policy_df['policy_mix_equipment_env']
    print(f"  最小值: {col_data.min():.4f}")
    print(f"  最大值: {col_data.max():.4f}")
    print(f"  平均值: {col_data.mean():.4f}")
    print(f"  标准差: {col_data.std():.4f}")
    extreme_high = policy_df[col_data > 50]
    extreme_low = policy_df[col_data < -50]
    print(f"  >50 的行数: {len(extreme_high)}")
    print(f"  <-50 的行数: {len(extreme_low)}")
    if len(extreme_high) > 0:
        print(f"  极端高值示例:\n{extreme_high[['County_Code', 'Year', 'policy_mix_equipment_env']].head(3).to_string(index=False)}")
    if len(extreme_low) > 0:
        print(f"  极端低值示例:\n{extreme_low[['County_Code', 'Year', 'policy_mix_equipment_env']].head(3).to_string(index=False)}")

# 高阳县关键年份数据
print(f"\n[高阳县(130628) 关键年份数据]")
gaoyang_key_years = [2015, 2017, 2018, 2019]
gaoyang_data = policy_df[(policy_df['County_Code'] == 130628) & (policy_df['Year'].isin(gaoyang_key_years))]
if len(gaoyang_data) > 0:
    for _, row in gaoyang_data.iterrows():
        print(f"\n  年份 {int(row['Year'])}:")
        print(f"    total_chunks: {row['total_chunks']}")
        print(f"    sample_reliable: {row['sample_reliable']}")
        print(f"    scored: {row['scored']}")
        print(f"    policy_intensity_total: {row['policy_intensity_total']:.2f}")
        print(f"    equipment_index: {row['equipment_index']:.2f}")
        print(f"    cluster_index: {row['cluster_index']:.2f}")
        print(f"    policy_mix_equipment_env: {row['policy_mix_equipment_env']:.4f}")
else:
    print("  未找到高阳县关键年份数据!")

# ============================================================
# 2. 对比 master_panel_data.csv vs master_panel_data_v2.csv
# ============================================================
print("\n" + "=" * 80)
print("2. master_panel_data.csv vs master_panel_data_v2.csv 对比")
print("=" * 80)

v1_path = os.path.join(BASE_DIR, 'output', 'master_panel_data.csv')
v2_path = os.path.join(BASE_DIR, 'output', 'master_panel_data_v2.csv')

v1_df = pd.read_csv(v1_path)
v2_df = pd.read_csv(v2_path)

print(f"\n[v1 基本信息]")
print(f"  行数: {len(v1_df)}")
print(f"  列数: {len(v1_df.columns)}")
print(f"  列名: {list(v1_df.columns)}")

print(f"\n[v2 基本信息]")
print(f"  行数: {len(v2_df)}")
print(f"  列数: {len(v2_df.columns)}")
print(f"  列名前20: {list(v2_df.columns)[:20]}")

# 检查v2是否包含各数据源
print(f"\n[v2 数据源覆盖检查]")
v2_cols_lower = [c.lower() for c in v2_df.columns]

sources = {
    'policy_scores': ['equipment_index', 'environment_index', 'ecommerce_index', 'policy_intensity_total'],
    'macro': ['gdp', 'population', 'industry'],
    'enterprise': ['enterprise', 'company', '注册'],
    'textile': ['textile', 'index', '指数'],
}

for source, keywords in sources.items():
    matched = []
    for keyword in keywords:
        for col in v2_cols_lower:
            if keyword in col:
                matched.append(col)
    print(f"  {source}: 找到 {len(matched)} 列")
    if matched:
        print(f"    示例: {matched[:5]}")

# 检查v2包含的列类型分布
print(f"\n[v2 列类型分布]")
col_prefixes = {}
for col in v2_df.columns:
    prefix = col.split('_')[0] if '_' in col else col
    col_prefixes[prefix] = col_prefixes.get(prefix, 0) + 1

print("  列名前缀统计 (Top 15):")
for prefix, count in sorted(col_prefixes.items(), key=lambda x: -x[1])[:15]:
    print(f"    {prefix}: {count} 列")

# 检查v1和v2的重叠列
common_cols = set(v1_df.columns) & set(v2_df.columns)
v1_only = set(v1_df.columns) - set(v2_df.columns)
v2_only = set(v2_df.columns) - set(v1_df.columns)

print(f"\n[列重叠分析]")
print(f"  共同列: {len(common_cols)} - {common_cols}")
print(f"  v1独有: {len(v1_only)} - {v1_only}")
print(f"  v2独有: {len(v2_only)} 列 (前10: {list(v2_only)[:10]})")

# ============================================================
# 3. 检查 government_reports/ TXT文件
# ============================================================
print("\n" + "=" * 80)
print("3. government_reports/ TXT文件质量检查")
print("=" * 80)

reports_dir = os.path.join(BASE_DIR, 'data', 'government_reports')
txt_files = [f for f in os.listdir(reports_dir) if f.endswith('.txt')]
print(f"\n[文件总数]")
print(f"  TXT文件数量: {len(txt_files)}")

# 检查高阳县文件
gaoyang_files = [f for f in txt_files if '高阳县' in f]
gaoyang_files.sort()

print(f"\n[高阳县文件列表]")
for f in gaoyang_files:
    filepath = os.path.join(reports_dir, f)
    size = os.path.getsize(filepath)
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        char_count = len(content)
    
    year_str = f.split('_')[1]
    year = int(year_str)
    
    is_short = "<500字符" if char_count < 500 else ""
    year_flag = " *** 关键年份" if year in [2015, 2017, 2018, 2019] else ""
    
    print(f"  {f}: {char_count} 字符 {is_short}{year_flag}")

# 随机抽查5个文件看乱码
print(f"\n[随机抽查5个文件 - 乱码检查]")
import random
random.seed(42)
sample_files = random.sample(txt_files, min(5, len(txt_files)))

for f in sample_files:
    filepath = os.path.join(reports_dir, f)
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 检查是否有常见乱码特征
    has_garbled = False
    garbled_indicators = ['', '\x00', '\ufffd']
    for indicator in garbled_indicators:
        if indicator in content:
            has_garbled = True
            break
    
    # 检查中文字符比例
    chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
    chinese_ratio = chinese_chars / len(content) if len(content) > 0 else 0
    
    print(f"\n  {f}:")
    print(f"    字符数: {len(content)}")
    print(f"    中文字符比例: {chinese_ratio:.2%}")
    print(f"    乱码标志: {'是' if has_garbled else '否'}")
    print(f"    前100字符: {content[:100]}...")

# 高阳县2000-2014年摘要检查
print(f"\n[高阳县2000-2014年字符数汇总]")
for year in range(2000, 2015):
    filename = f"高阳县_{year}_report.txt"
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        status = "摘要(<500)" if len(content) < 500 else "完整"
        print(f"  {year}: {len(content)} 字符 - {status}")

# 高阳县关键年份字符数
print(f"\n[高阳县关键年份字符数]")
for year in [2015, 2017, 2018, 2019]:
    filename = f"高阳县_{year}_report.txt"
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"  {year}: {len(content)} 字符")

print("\n" + "=" * 80)
print("审计完成!")
print("=" * 80)
