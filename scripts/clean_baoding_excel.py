import pandas as pd
import os
import re

excel_path = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\policies\保定市各县政府工作报告.xlsx'
output_dir = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports_from_excel'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(excel_path)
print(f'Excel total rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

content_col = None
year_col = None
county_col = None

for col in df.columns:
    sample = str(df[col].iloc[0])[:50]
    if any(kw in sample for kw in ['各位代表', '各位委员', '政府工作', '人民代表大会']):
        content_col = col
    if '年' in str(col) or str(col).strip() in ['year', '年份']:
        year_col = col

print(f'Content column: {content_col}')
print(f'Year column candidate: {year_col}')

if year_col is None:
    for col in df.columns:
        sample = str(df[col].iloc[:5].tolist())
        if re.search(r'20\d{2}', sample):
            year_col = col
            break
    print(f'Year column found by regex: {year_col}')

if county_col is None:
    for col in df.columns:
        sample = str(df[col].iloc[:10].tolist())
        if '高阳' in sample or '清苑' in sample or '保定' in sample:
            county_col = col
            break
    print(f'County column found: {county_col}')

saved_count = 0
for idx, row in df.iterrows():
    year = None
    county = None
    content = ''
    
    if year_col:
        year = str(row[year_col]).strip()
        year_match = re.search(r'20\d{2}', year)
        if year_match:
            year = year_match.group()
        else:
            year = None
    
    if county_col:
        county = str(row[county_col]).strip()
    
    if content_col:
        content = str(row[content_col]).strip()
    
    if year and content and len(content) > 500:
        fname = f'保定市_{county}_{year}_report.txt'
        fname = re.sub(r'[<>:"/\\|?*]', '_', fname)
        fpath = os.path.join(output_dir, fname)
        
        title = f'{county}{year}年政府工作报告'
        text = f"标题：{title}\n发文单位：{county}人民政府\n年份：{year}\n来源：保定市各县政府工作报告.xlsx\n\n正文：\n\n{content}"
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        saved_count += 1
        print(f'Saved: {fname} ({len(content)} chars)')

print(f'\nTotal saved: {saved_count} reports')
