"""
深度审计 - 第二轮检查
"""
import pandas as pd
import os

BASE_DIR = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾'

# ============================================================
# 1. policy_scores_panel.csv 深度检查
# ============================================================
print("=" * 80)
print("1. policy_scores_panel.csv 深度检查")
print("=" * 80)

policy_df = pd.read_csv(os.path.join(BASE_DIR, 'output', 'policy_scores_panel.csv'))

# 检查 sample_reliable=0 但 policy_intensity_total != 0 的行
print("\n[sample_reliable=0 但仍有policy_intensity_total值的异常行]")
anomaly = policy_df[(policy_df['sample_reliable'] == 0) & (policy_df['policy_intensity_total'] != 0)]
if len(anomaly) > 0:
    print(f"  发现 {len(anomaly)} 行:")
    print(anomaly[['County_Code', 'Year', 'sample_reliable', 'scored', 'policy_intensity_total', 'policy_mix_equipment_env']].to_string(index=False))
else:
    print("  无此异常")

# 检查 scored=0 但有 index 值的行
print("\n[scored=0 但有index非0值的异常行]")
score_cols = ['equipment_index', 'environment_index', 'ecommerce_index', 'brandquality_index', 'cluster_index', 'finance_index', 'education_index']
anomaly2 = policy_df[(policy_df['scored'] == 0) & (policy_df[score_cols].any(axis=1))]
if len(anomaly2) > 0:
    print(f"  发现 {len(anomaly2)} 行:")
    print(anomaly2[['County_Code', 'Year', 'scored'] + score_cols].to_string(index=False))
else:
    print("  无此异常")

# 检查缺失值模式
print("\n[缺失值统计]")
na_counts = policy_df.isna().sum()
na_cols = na_counts[na_counts > 0]
if len(na_cols) > 0:
    print("  有缺失值的列:")
    for col, count in na_cols.items():
        print(f"    {col}: {count} 个缺失值 ({count/len(policy_df):.1%})")
else:
    print("  无缺失值")

# 检查 sample_reliable=0 行的 L1/L2 指标
print("\n[sample_reliable=0 的行的 L1/L2 指标]")
unreliable = policy_df[policy_df['sample_reliable'] == 0]
l_cols = [c for c in policy_df.columns if c.startswith('L1_') or c.startswith('L2_')]
if len(unreliable) > 0:
    for _, row in unreliable.head(5).iterrows():
        print(f"  年份 {int(row['Year'])}, 县码 {int(row['County_Code'])}:")
        has_l1_l2 = False
        for c in l_cols:
            if pd.notna(row[c]) and row[c] != 0:
                has_l1_l2 = True
        if has_l1_l2:
            print(f"    警告: sample_reliable=0 但有非零L1/L2值!")
        else:
            print(f"    L1/L2 全为0或NaN (正常)")

# ============================================================
# 2. v2 数据深度检查
# ============================================================
print("\n" + "=" * 80)
print("2. master_panel_data_v2.csv 深度检查")
print("=" * 80)

v2_df = pd.read_csv(os.path.join(BASE_DIR, 'output', 'master_panel_data_v2.csv'))

print(f"\n[v2 年份范围]")
print(f"  最小年份: {v2_df['Year'].min()}")
print(f"  最大年份: {v2_df['Year'].max()}")
print(f"  年份数量: {v2_df['Year'].nunique()}")

print(f"\n[v2 缺失值检查]")
na_v2 = v2_df.isna().sum()
na_v2_cols = na_v2[na_v2 > 0]
if len(na_v2_cols) > 0:
    print("  有缺失值的列:")
    for col, count in na_v2_cols.items():
        print(f"    {col}: {count} ({count/len(v2_df):.1%})")
else:
    print("  无缺失值")

# 检查关键变量在关键年份的值
print(f"\n[v2 高阳县关键年份的核心变量]")
key_vars = ['gdp', 'total_firms', 'textile_firms', 'equipment_index', 'cluster_index', 'policy_intensity_total']
key_vars_exist = [v for v in key_vars if v in v2_df.columns]

for year in [2015, 2017, 2018, 2019]:
    row = v2_df[v2_df['Year'] == year]
    if len(row) > 0:
        print(f"\n  {year}:")
        for v in key_vars_exist:
            print(f"    {v}: {row[v].values[0]}")

# 检查v2是否真的没有企业注册数据
print(f"\n[v2 企业注册变量检查]")
ent_cols_v2 = [c for c in v2_df.columns if 'firm' in c.lower() or 'enterprise' in c.lower() or '注册' in c]
print(f"  企业相关列: {ent_cols_v2}")
if 'total_firms' in v2_df.columns:
    print(f"  total_firms 非零行数: {(v2_df['total_firms'] != 0).sum()}")
    print(f"  total_firms 值分布: {v2_df['total_firms'].describe()}")

# ============================================================
# 3. TXT 文件深入检查
# ============================================================
print("\n" + "=" * 80)
print("3. government_reports/ TXT 文件深入检查")
print("=" * 80)

reports_dir = os.path.join(BASE_DIR, 'data', 'government_reports')

# 检查乱码标志 - 具体分析
print("\n[乱码检测 - 具体分析]")
txt_files = [f for f in os.listdir(reports_dir) if f.endswith('.txt')]

import random
random.seed(42)
sample_files = random.sample(txt_files, min(5, len(txt_files)))

for f in sample_files:
    filepath = os.path.join(reports_dir, f)
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 更精细的乱码检测
    null_count = content.count('\x00')
    replacement_count = content.count('\ufffd')
    double_newline_count = content.count('\n\n')
    
    print(f"\n  {f}:")
    print(f"    总字符: {len(content)}")
    print(f"    \\x00 (null): {null_count}")
    print(f"    \\ufffd (replacement): {replacement_count}")
    print(f"    \\n\\n (双换行): {double_newline_count}")
    if null_count > 0 or replacement_count > 0:
        print(f"    *** 存在真正乱码 ***")
    elif double_newline_count > 0:
        print(f"    提示: 有 {double_newline_count} 处双换行，这是正常格式")

# 检查高阳县2000-2014是否真的是相同内容
print("\n[高阳县2000-2014内容重复性检查]")
contents = {}
for year in range(2000, 2015):
    filename = f"高阳县_{year}_report.txt"
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        contents[year] = content

# 比较相邻年份
unique_contents = set()
for year, content in contents.items():
    unique_contents.add(content)

print(f"  2000-2014共 {len(contents)} 个文件")
print(f"  唯一内容数量: {len(unique_contents)}")
if len(unique_contents) < len(contents):
    print(f"  *** 警告: 存在完全相同的内容! ***")
    # 找重复组
    from collections import Counter
    content_counter = Counter(contents.values())
    for content, count in content_counter.items():
        if count > 1:
            years_with_content = [y for y, c in contents.items() if c == content]
            print(f"    相同内容出现在: {years_with_content}")
            print(f"    内容前100字符: {content[:100]}...")

# ============================================================
# 4. scripts/ 下是否有v1版本的merge脚本
# ============================================================
print("\n" + "=" * 80)
print("4. scripts/ 脚本版本检查")
print("=" * 80)

scripts_dir = os.path.join(BASE_DIR, 'scripts')
all_scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
merge_scripts = [f for f in all_scripts if 'merge' in f.lower() or 'master' in f.lower() or 'panel' in f.lower()]
print(f"\n[与master panel相关的脚本]")
for s in merge_scripts:
    filepath = os.path.join(scripts_dir, s)
    size = os.path.getsize(filepath)
    print(f"  {s}: {size} 字节")

# 检查是否有v1版本的脚本
v1_scripts = [f for f in all_scripts if 'v1' in f.lower() or 'master_panel' in f.lower()]
print(f"\n[v1相关脚本]")
print(f"  发现: {v1_scripts}")

print("\n" + "=" * 80)
print("深度审计完成!")
print("=" * 80)
