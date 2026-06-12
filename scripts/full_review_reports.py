import os
import pandas as pd

report_dir = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports'

results = []
for fname in sorted(os.listdir(report_dir)):
    if not fname.endswith('.txt'):
        continue
    fpath = os.path.join(report_dir, fname)
    size = os.path.getsize(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        lines = content.strip().split('\n')
        num_lines = len(lines)
        num_chars = len(content)
    
    # 判断质量
    is_summary = False
    quality = '完整'
    reasons = []
    
    if size < 5000:
        is_summary = True
        quality = '摘要(<5KB)'
        reasons.append('文件过小')
    if num_lines < 50:
        is_summary = True
        quality = '摘要(<50行)'
        reasons.append('行数过少')
    if '摘要版本' in content or '年代久远' in content or '未能找到完整原文' in content or '仅供参考' in content:
        is_summary = True
        quality = '摘要(标记)'
        reasons.append('包含摘要标记')
    if num_chars < 3000:
        is_summary = True
        quality = '摘要(<3000字)'
        reasons.append('字符过少')
    
    # 解析文件名
    parts = fname.replace('_report.txt', '').split('_')
    level = parts[0]
    year = parts[1] if len(parts) > 1 else ''
    
    results.append({
        '文件': fname,
        '层级': level,
        '年份': year,
        '文件大小': f'{size/1024:.1f}KB',
        '行数': num_lines,
        '字符数': num_chars,
        '质量': quality,
        '原因': '; '.join(reasons) if reasons else '-'
    })

df = pd.DataFrame(results)
print(f"\n=== 报告质量全面审查 ===")
print(f"总文件数: {len(df)}")
print(f"完整报告: {len(df[df['质量'].str.startswith('完整')])} 份")
print(f"摘要版本: {len(df[df['质量'].str.startswith('摘要')])} 份")
print(f"\n--- 按层级统计 ---")
for level in ['河北省', '保定市', '高阳县']:
    sub = df[df['层级'] == level]
    complete = len(sub[sub['质量'].str.startswith('完整')])
    summary = len(sub[sub['质量'].str.startswith('摘要')])
    print(f"{level}: 总{len(sub)}份 | 完整{complete}份 | 摘要{summary}份 | 完整率{complete/len(sub)*100:.1f}%")

print(f"\n--- 不完整报告清单 ---")
for _, row in df[df['质量'].str.startswith('摘要')].iterrows():
    print(f"  {row['文件']} | {row['文件大小']} | {row['行数']}行 | {row['原因']}")

print(f"\n--- 完整报告清单 ---")
for _, row in df[df['质量'].str.startswith('完整')].iterrows():
    print(f"  {row['文件']} | {row['文件大小']} | {row['行数']}行")
