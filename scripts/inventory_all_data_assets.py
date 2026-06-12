"""
全面梳理所有数据资产
按时间、结构、内容、指标维度整理
"""
import os
import re
import pandas as pd
from datetime import datetime

base_dir = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾"

print("=" * 100)
print("高阳县毛巾产业 数据资产全面梳理")
print("=" * 100)

all_data = []

# ============================================================
# 1. 政府工作报告 (81份, 2000-2026)
# ============================================================
print("\n【1】扫描政府工作报告...")

report_dir = os.path.join(base_dir, "政府工作报告")
report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('_report.txt')])

for filename in report_files:
    filepath = os.path.join(report_dir, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析字段
    level_match = re.match(r'^(高阳县|保定市|河北省)_(\d{4})_', filename)
    level = level_match.group(1) if level_match else ''
    year = int(level_match.group(2)) if level_match else 0
    
    title_match = re.search(r'^标题[：:]\s*(.+)$', content, re.MULTILINE)
    unit_match = re.search(r'^发文单位[：:]\s*(.+)$', content, re.MULTILINE)
    time_match = re.search(r'^发文时间[：:]\s*(.+)$', content, re.MULTILINE)
    body_match = re.search(r'^正文[：:]\s*\n(.+)$', content, re.MULTILINE | re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else ''
    pub_time = time_match.group(1).strip() if time_match else ''
    body = body_match.group(1).strip() if body_match else ''
    
    # 提取指标关键词
    indicators = []
    indicator_keywords = ['生产总值', 'GDP', '规模以上工业', '财政收入', '固定资产投资',
                         '社会消费品零售总额', '城乡居民人均可支配收入', '农民人均纯收入',
                         '纺织', '毛巾', '织机', '电商', '进出口', '就业', '脱贫',
                         'PM2.5', '污水处理', '学校', '医院', '城镇化']
    
    for kw in indicator_keywords:
        if kw in body:
            indicators.append(kw)
    
    all_data.append({
        '数据大类': '文本数据',
        '数据类型': '政府工作报告',
        '数据层级': level,
        '年份': year,
        '时间跨度': f'{year}年',
        '标题': title,
        '发文单位': unit_match.group(1).strip() if unit_match else '',
        '内容摘要': body[:100] + '...' if len(body) > 100 else body,
        '正文长度(字符)': len(body),
        '包含指标': ', '.join(indicators),
        '指标数量': len(indicators),
        '文件格式': 'TXT',
        '文件路径': f'政府工作报告/{filename}',
        '数据状态': '完整'
    })

print(f"  完成: {len(report_files)} 份报告")

# ============================================================
# 2. 高阳县政策文件 (92篇)
# ============================================================
print("\n【2】扫描高阳县政策文件...")

policy_dir = os.path.join(base_dir, "高阳县政策文件")
policy_files = [f for f in os.listdir(policy_dir) if f.endswith('.xlsx') and '全量' in f]

if policy_files:
    latest_policy = sorted(policy_files)[-1]
    policy_file = os.path.join(policy_dir, latest_policy)
    
    df_policy = pd.read_excel(policy_file)
    
    for idx, row in df_policy.iterrows():
        year_match = re.search(r'(\d{4})', str(row.get('发文时间', '')))
        year = int(year_match.group(1)) if year_match else 0
        
        # 提取指标
        body = str(row.get('正文', ''))
        indicators = []
        for kw in ['纺织', '毛巾', '产业', '电商', '园区', '税收', '技改', '转型', '品牌', '质量', '环保']:
            if kw in body:
                indicators.append(kw)
        
        all_data.append({
            '数据大类': '文本数据',
            '数据类型': '政策文件',
            '数据层级': '高阳县',
            '年份': year,
            '时间跨度': f'{year}年',
            '标题': str(row.get('标题', '')),
            '发文单位': str(row.get('发文单位', '')),
            '内容摘要': body[:100] + '...' if len(body) > 100 else body,
            '正文长度(字符)': len(body),
            '包含指标': ', '.join(indicators),
            '指标数量': len(indicators),
            '文件格式': 'Excel',
            '文件路径': f'高阳县政策文件/{latest_policy}',
            '数据状态': '完整'
        })

print(f"  完成: {len(df_policy)} 篇政策文件")

# ============================================================
# 3. 纺织指数数据
# ============================================================
print("\n【3】扫描纺织指数数据...")

index_files = [f for f in os.listdir(base_dir) if f.endswith('.xlsx') and '纺织指数' in f]

index_total = 0
for filename in index_files:
    filepath = os.path.join(base_dir, filename)
    try:
        df = pd.read_excel(filepath)
        index_total += len(df)
        
        # 提取年份范围
        years = []
        for col in df.columns:
            if '日期' in col or '时间' in col or '期' in col:
                year_matches = df[col].astype(str).str.extract(r'(\d{4})').dropna()[0].unique()
                years.extend([int(y) for y in year_matches if y.isdigit()])
        
        year_range = f"{min(years)}-{max(years)}" if years else '未知'
        
        all_data.append({
            '数据大类': '数值数据',
            '数据类型': '纺织指数',
            '数据层级': '高阳县',
            '年份': 0,
            '时间跨度': year_range,
            '标题': filename.replace('.xlsx', ''),
            '发文单位': '高阳县/河北省',
            '内容摘要': f'共{len(df)}条记录，{len(df.columns)}个字段',
            '正文长度(字符)': 0,
            '包含指标': ', '.join([c for c in df.columns if len(str(c)) < 30]),
            '指标数量': len(df.columns),
            '文件格式': 'Excel',
            '文件路径': filename,
            '数据状态': '完整'
        })
    except Exception as e:
        print(f"  读取失败: {filename} - {e}")

print(f"  完成: {len(index_files)} 个指数文件, {index_total} 条数据")

# ============================================================
# 4. 县域统计年鉴
# ============================================================
print("\n【4】扫描县域统计年鉴...")

yearbook_dir = os.path.join(base_dir, "县域统计年鉴")
yearbook_folders = sorted([f for f in os.listdir(yearbook_dir) 
                          if os.path.isdir(os.path.join(yearbook_dir, f))])

total_yearbook_files = 0
for folder in yearbook_folders:
    folder_path = os.path.join(yearbook_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.xls') or f.endswith('.xlsx')]
    total_yearbook_files += len(files)
    
    if files:
        all_data.append({
            '数据大类': '数值数据',
            '数据类型': '县域统计年鉴',
            '数据层级': '全国各县域',
            '年份': int(folder) if folder.isdigit() else 0,
            '时间跨度': f'{folder}年',
            '标题': f'{folder}年中国县域统计年鉴',
            '发文单位': '国家统计局',
            '内容摘要': f'共{len(files)}个Excel文件，包含各县域经济、社会、农业、工业等指标',
            '正文长度(字符)': 0,
            '包含指标': 'GDP、人口、农业、工业、固定资产投资、财政、教育、卫生等',
            '指标数量': 100,
            '文件格式': 'Excel',
            '文件路径': f'县域统计年鉴/{folder}/',
            '数据状态': '完整'
        })

print(f"  完成: {len(yearbook_folders)} 个年份, {total_yearbook_files} 个文件")

# ============================================================
# 5. 面板数据
# ============================================================
print("\n【5】扫描面板数据...")

panel_dir = os.path.join(base_dir, "高阳县面板数据")
if os.path.exists(panel_dir):
    panel_files = os.listdir(panel_dir)
    for filename in panel_files:
        filepath = os.path.join(panel_dir, filename)
        if filename.endswith('.csv') or filename.endswith('.xlsx'):
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath, encoding='utf-8-sig')
                else:
                    df = pd.read_excel(filepath)
                
                # 提取年份范围
                years = []
                for col in df.columns:
                    if 'year' in col.lower() or '年' in col:
                        year_vals = df[col].dropna().unique()
                        years.extend([int(y) for y in year_vals if str(y).isdigit()])
                
                year_range = f"{min(years)}-{max(years)}" if years else '未知'
                
                all_data.append({
                    '数据大类': '数值数据',
                    '数据类型': '面板数据',
                    '数据层级': '高阳县',
                    '年份': 0,
                    '时间跨度': year_range,
                    '标题': filename.replace('.csv', '').replace('.xlsx', ''),
                    '发文单位': '整理生成',
                    '内容摘要': f'共{len(df)}行，{len(df.columns)}列面板数据',
                    '正文长度(字符)': 0,
                    '包含指标': ', '.join([c for c in df.columns if len(str(c)) < 30]),
                    '指标数量': len(df.columns),
                    '文件格式': filename.split('.')[-1].upper(),
                    '文件路径': f'高阳县面板数据/{filename}',
                    '数据状态': '完整'
                })
            except Exception as e:
                print(f"  读取失败: {filename} - {e}")

print(f"  完成: {len(panel_files)} 个文件")

# ============================================================
# 6. 图表文件
# ============================================================
print("\n【6】扫描图表文件...")

figures_dir = os.path.join(base_dir, "figures")
if os.path.exists(figures_dir):
    fig_files = []
    for root, dirs, files in os.walk(figures_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                fig_files.append(os.path.join(root, f))
    
    if fig_files:
        all_data.append({
            '数据大类': '可视化数据',
            '数据类型': '图表',
            '数据层级': '高阳县',
            '年份': 0,
            '时间跨度': '多年度',
            '标题': f'高阳县毛巾产业相关图表',
            '发文单位': '整理生成',
            '内容摘要': f'共{len(fig_files)}个图表文件(PNG/JPG格式)',
            '正文长度(字符)': 0,
            '包含指标': '产业规模、企业数量、产值、电商销售等可视化图表',
            '指标数量': len(fig_files),
            '文件格式': 'PNG/JPG',
            '文件路径': 'figures/',
            '数据状态': '完整'
        })

print(f"  完成: {len(fig_files)} 个图表")

# ============================================================
# 创建汇总Excel
# ============================================================
print("\n" + "=" * 100)
print("生成数据资产清单...")
print("=" * 100)

df_all = pd.DataFrame(all_data)

# 生成输出文件
output_file = os.path.join(base_dir, f"高阳县毛巾产业_数据资产清单_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 总清单
    df_all.to_excel(writer, sheet_name='数据资产总清单', index=False)
    
    # 按大类统计
    for category in df_all['数据大类'].unique():
        df_cat = df_all[df_all['数据大类'] == category]
        sheet_name = category[:31]
        df_cat.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 时间维度统计
    df_with_year = df_all[df_all['年份'] > 0].copy()
    if len(df_with_year) > 0:
        time_stats = df_with_year.groupby(['数据大类', '数据类型', '年份']).size().reset_index(name='数量')
        time_stats.to_excel(writer, sheet_name='时间维度统计', index=False)
    
    # 结构维度统计
    structure_stats = df_all.groupby(['数据大类', '数据类型', '数据层级']).agg({
        '序号' if '序号' in df_all.columns else '年份': 'count'
    }).reset_index()
    structure_stats.columns = ['数据大类', '数据类型', '数据层级', '数量']
    structure_stats.to_excel(writer, sheet_name='结构维度统计', index=False)
    
    # 指标维度统计
    indicator_stats = df_all[df_all['包含指标'] != ''].groupby(['数据大类', '数据类型'])['包含指标'].apply(
        lambda x: ', '.join(set(', '.join(x.astype(str)).split(', ')))
    ).reset_index()
    indicator_stats.to_excel(writer, sheet_name='指标维度统计', index=False)
    
    # 汇总统计
    summary = pd.DataFrame({
        '统计项': [
            '数据大类数量',
            '数据类型数量',
            '数据总条目数',
            '时间跨度',
            '数据层级',
            '文本数据总量',
            '数值数据总量',
            '可视化数据总量'
        ],
        '数值': [
            df_all['数据大类'].nunique(),
            df_all['数据类型'].nunique(),
            len(df_all),
            f"{df_all[df_all['年份']>0]['年份'].min()}-{df_all[df_all['年份']>0]['年份'].max()}",
            ', '.join(df_all['数据层级'].unique()),
            len(df_all[df_all['数据大类'] == '文本数据']),
            len(df_all[df_all['数据大类'] == '数值数据']),
            len(df_all[df_all['数据大类'] == '可视化数据'])
        ]
    })
    summary.to_excel(writer, sheet_name='汇总统计', index=False)

print(f"\n数据资产清单已生成: {output_file}")

# ============================================================
# 控制台输出摘要
# ============================================================
print("\n" + "=" * 100)
print("数据资产摘要")
print("=" * 100)

print(f"\n【数据大类】")
for cat in df_all['数据大类'].unique():
    count = len(df_all[df_all['数据大类'] == cat])
    print(f"  {cat}: {count} 条")

print(f"\n【数据类型】")
for dtype in df_all['数据类型'].unique():
    count = len(df_all[df_all['数据类型'] == dtype])
    print(f"  {dtype}: {count} 条")

print(f"\n【时间跨度】")
valid_years = df_all[df_all['年份'] > 0]['年份']
if len(valid_years) > 0:
    print(f"  最早年份: {valid_years.min()}")
    print(f"  最晚年份: {valid_years.max()}")
    print(f"  覆盖年数: {valid_years.max() - valid_years.min() + 1}")

print(f"\n【数据层级】")
for level in df_all['数据层级'].unique():
    count = len(df_all[df_all['数据层级'] == level])
    print(f"  {level}: {count} 条")

print(f"\n【指标覆盖】")
all_indicators = set()
for indicators in df_all['包含指标'].dropna():
    for ind in str(indicators).split(', '):
        if ind.strip():
            all_indicators.add(ind.strip())

print(f"  总指标数: {len(all_indicators)}")
print(f"  主要指标: {', '.join(list(all_indicators)[:20])}")

print(f"\n{'=' * 100}")
print("梳理完成！")
print(f"{'=' * 100}")
