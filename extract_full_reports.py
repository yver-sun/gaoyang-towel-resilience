"""
从Excel中提取完整政府工作报告并生成TXT文件
"""
import pandas as pd
import os

# 读取Excel
df = pd.read_excel('data/policies/保定市各县政府工作报告.xlsx')

# 查看列名和数据
print("列名:", df.columns.tolist())
print("形状:", df.shape)

# 检查报告全文列是否有数据
print("\n报告全文列统计:")
print(f"非空数量: {df['报告全文'].notna().sum()}")
print(f"空值数量: {df['报告全文'].isna().sum()}")

# 查看高阳县的报告
gaoyang = df[df['区县名称'].str.contains('高阳', na=False)]
print(f"\n高阳县报告数量: {len(gaoyang)}")
for idx, row in gaoyang.iterrows():
    content = row.get('报告全文', '')
    print(f"  {row['年份']}年 - 标题: {row.get('报告标题', 'N/A')}")
    print(f"    文本长度: {row.get('文本总长度(字)', 'N/A')}")
    if pd.notna(content):
        print(f"    报告全文前100字: {str(content)[:100]}")
    else:
        print(f"    报告全文: NaN (空值)")

# 查看其他县的报告全文
print("\n\n检查其他县的报告全文:")
for idx, row in df.head(5).iterrows():
    content = row.get('报告全文', '')
    print(f"\n{row['区县名称']} {row['年份']}年:")
    if pd.notna(content):
        print(f"  报告全文前100字: {str(content)[:100]}")
    else:
        print(f"  报告全文: NaN (空值)")
