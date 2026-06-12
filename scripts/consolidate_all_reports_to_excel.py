"""
整合所有政府工作报告和政策文件到Excel
确保内容完整、格式统一
"""
import os
import re
import pandas as pd
from datetime import datetime

base_dir = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾"
report_dir = os.path.join(base_dir, "政府工作报告")
output_dir = base_dir

print("=" * 90)
print("整合所有文本到Excel")
print("=" * 90)

# ============================================================
# 1. 整合政府工作报告
# ============================================================
print("\n【1】整合政府工作报告...")

report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('_report.txt')])

reports_data = []
errors = []

for filename in report_files:
    filepath = os.path.join(report_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析字段
        title_match = re.search(r'^标题[：:]\s*(.+)$', content, re.MULTILINE)
        unit_match = re.search(r'^发文单位[：:]\s*(.+)$', content, re.MULTILINE)
        time_match = re.search(r'^发文时间[：:]\s*(.+)$', content, re.MULTILINE)
        url_match = re.search(r'^来源URL[：:]\s*(.+)$', content, re.MULTILINE)
        body_match = re.search(r'^正文[：:]\s*\n(.+)$', content, re.MULTILINE | re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else ''
        unit = unit_match.group(1).strip() if unit_match else ''
        pub_time = time_match.group(1).strip() if time_match else ''
        url = url_match.group(1).strip() if url_match else ''
        body = body_match.group(1).strip() if body_match else ''
        
        # 从文件名提取信息
        level_match = re.match(r'^(高阳县|保定市|河北省)_(\d{4})_', filename)
        level = level_match.group(1) if level_match else ''
        year = int(level_match.group(2)) if level_match else 0
        
        # 验证完整性
        issues = []
        if not title: issues.append('缺少标题')
        if not unit: issues.append('缺少发文单位')
        if not pub_time: issues.append('缺少发文时间')
        if not body: issues.append('缺少正文')
        if len(body) < 100: issues.append(f'正文过短({len(body)}字符)')
        
        if issues:
            errors.append(f"{filename}: {', '.join(issues)}")
        
        reports_data.append({
            '序号': len(reports_data) + 1,
            '层级': level,
            '年份': year,
            '标题': title,
            '发文单位': unit,
            '发文时间': pub_time,
            '正文长度': len(body),
            '正文': body,
            '来源URL': url,
            '文件名': filename,
            '完整性': '完整' if not issues else '异常: ' + '; '.join(issues)
        })
        
    except Exception as e:
        errors.append(f"{filename}: 读取错误 - {str(e)}")

# 创建DataFrame
df_reports = pd.DataFrame(reports_data)

# 按层级和年份排序
df_reports = df_reports.sort_values(['层级', '年份']).reset_index(drop=True)
df_reports['序号'] = range(1, len(df_reports) + 1)

# 保存到Excel
output_file_reports = os.path.join(output_dir, f"政府工作报告汇总_2000_2026_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

with pd.ExcelWriter(output_file_reports, engine='openpyxl') as writer:
    # 完整报告Sheet
    df_reports.to_excel(writer, sheet_name='政府工作报告', index=False)
    
    # 高阳县Sheet
    df_gaoyang = df_reports[df_reports['层级'] == '高阳县'].copy()
    df_gaoyang.to_excel(writer, sheet_name='高阳县', index=False)
    
    # 保定市Sheet
    df_baoding = df_reports[df_reports['层级'] == '保定市'].copy()
    df_baoding.to_excel(writer, sheet_name='保定市', index=False)
    
    # 河北省Sheet
    df_hebei = df_reports[df_reports['层级'] == '河北省'].copy()
    df_hebei.to_excel(writer, sheet_name='河北省', index=False)
    
    # 统计Sheet
    df_stats = pd.DataFrame({
        '统计项': ['总报告数', '高阳县报告数', '保定市报告数', '河北省报告数',
                   '完整报告数', '异常报告数', '覆盖年份', '平均正文字数'],
        '数值': [
            len(df_reports),
            len(df_gaoyang),
            len(df_baoding),
            len(df_hebei),
            len(df_reports[df_reports['完整性'] == '完整']),
            len(df_reports[df_reports['完整性'] != '完整']),
            f"{df_reports['年份'].min()}-{df_reports['年份'].max()}",
            df_reports['正文长度'].mean()
        ]
    })
    df_stats.to_excel(writer, sheet_name='统计汇总', index=False)

print(f"  政府报告: {len(df_reports)} 份")
print(f"  输出文件: {output_file_reports}")

# ============================================================
# 2. 整合高阳县政策文件（使用最新的全量版本）
# ============================================================
print("\n【2】整合高阳县政策文件...")

policy_dir = os.path.join(base_dir, "高阳县政策文件")
policy_files = [f for f in os.listdir(policy_dir) if f.endswith('.xlsx') and '全量' in f]

if policy_files:
    # 使用最新的文件
    latest_policy = sorted(policy_files)[-1]
    policy_file = os.path.join(policy_dir, latest_policy)
    
    df_policy = pd.read_excel(policy_file)
    
    # 保存到新的汇总文件
    output_file_policy = os.path.join(output_dir, f"高阳县毛巾产业政策文件_汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    with pd.ExcelWriter(output_file_policy, engine='openpyxl') as writer:
        df_policy.to_excel(writer, sheet_name='政策文件', index=False)
        
        # 添加统计Sheet
        if '发文时间' in df_policy.columns:
            df_policy['年份'] = df_policy['发文时间'].astype(str).str.extract(r'(\d{4})')
            year_stats = df_policy.groupby('年份').size().reset_index(name='数量')
            year_stats.to_excel(writer, sheet_name='年度统计', index=False)
        
        if '政策类别' in df_policy.columns:
            category_stats = df_policy.groupby('政策类别').size().reset_index(name='数量')
            category_stats.to_excel(writer, sheet_name='类别统计', index=False)
    
    print(f"  政策文件: {len(df_policy)} 篇")
    print(f"  输出文件: {output_file_policy}")
else:
    print("  未找到政策文件")

# ============================================================
# 3. 整合纺织指数数据
# ============================================================
print("\n【3】整合纺织指数数据...")

index_files = [f for f in os.listdir(base_dir) if f.endswith('.xlsx') and '纺织指数' in f]

if index_files:
    index_data_all = []
    
    for filename in index_files:
        filepath = os.path.join(base_dir, filename)
        try:
            df = pd.read_excel(filepath)
            # 添加来源列
            df['指数类别'] = filename.replace('河北·高阳纺织指数-', '').replace('.xlsx', '')
            index_data_all.append(df)
        except Exception as e:
            print(f"  读取失败: {filename} - {e}")
    
    if index_data_all:
        df_all_indices = pd.concat(index_data_all, ignore_index=True)
        
        output_file_indices = os.path.join(output_dir, f"高阳纺织指数_全量汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        with pd.ExcelWriter(output_file_indices, engine='openpyxl') as writer:
            df_all_indices.to_excel(writer, sheet_name='全量数据', index=False)
            
            # 按类别分Sheet
            for category in df_all_indices['指数类别'].unique():
                df_cat = df_all_indices[df_all_indices['指数类别'] == category]
                sheet_name = category[:31]  # Excel sheet name limit
                df_cat.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"  指数数据: {len(df_all_indices)} 条")
        print(f"  指数类别: {df_all_indices['指数类别'].nunique()} 个")
        print(f"  输出文件: {output_file_indices}")
else:
    print("  未找到纺织指数文件")

# ============================================================
# 4. 输出验证报告
# ============================================================
print("\n" + "=" * 90)
print("【4】验证报告")
print("=" * 90)

print(f"\n政府工作报告:")
print(f"  总数: {len(df_reports)} 份")
print(f"  完整: {len(df_reports[df_reports['完整性'] == '完整'])} 份")
print(f"  异常: {len(df_reports[df_reports['完整性'] != '完整'])} 份")

if errors:
    print(f"\n异常文件详情:")
    for err in errors:
        print(f"  - {err}")

print(f"\n高阳县政策文件:")
if policy_files:
    print(f"  总数: {len(df_policy)} 篇")
    years = df_policy['发文时间'].astype(str).str.extract(r'(\d{4})').dropna()
    if len(years) > 0:
        print(f"  时间跨度: {years.min()[0]}-{years.max()[0]}")

print(f"\n纺织指数数据:")
if index_files:
    print(f"  文件数: {len(index_files)} 个")
    print(f"  数据条数: {len(df_all_indices)} 条")

print("\n" + "=" * 90)
print("整合完成！")
print("=" * 90)

# 输出文件列表
print(f"\n生成的Excel文件:")
print(f"  1. {os.path.basename(output_file_reports)}")
if policy_files:
    print(f"  2. {os.path.basename(output_file_policy)}")
if index_files:
    print(f"  3. {os.path.basename(output_file_indices)}")
