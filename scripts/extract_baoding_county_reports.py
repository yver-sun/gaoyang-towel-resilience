import pandas as pd
import os
import re

excel_path = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\policies\保定市各县政府工作报告.xlsx'
output_dir = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports_from_excel'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(excel_path)
print(f'Excel total rows: {len(df)}')

saved_count = 0
empty_count = 0

for idx, row in df.iterrows():
    year = str(row['年份']).strip()
    county = str(row['区县名称']).strip()
    report_full = row['报告全文']
    text_length = int(row['文本总长度(字)']) if pd.notna(row['文本总长度(字)']) else 0
    
    if pd.isna(report_full) or text_length < 500:
        empty_count += 1
        continue
    
    report_text = str(report_full).strip()
    report_text = re.sub(r'\s+', '\n', report_text)
    
    fname = f'保定市_{county}_{year}_report.txt'
    fname = re.sub(r'[<>:"/\\|?*]', '_', fname)
    fpath = os.path.join(output_dir, fname)
    
    title = f'{county}{year}年政府工作报告'
    header = f"标题：{title}\n发文单位：{county}人民政府\n年份：{year}\n来源：保定市各县政府工作报告.xlsx\n\n正文：\n\n{report_text}"
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(header)
    
    saved_count += 1

print(f'Saved: {saved_count} reports')
print(f'Empty/skipped: {empty_count} reports')
