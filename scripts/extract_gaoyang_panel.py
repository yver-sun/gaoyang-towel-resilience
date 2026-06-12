"""
将高阳县所有数据提取并整理为单独的面板数据
"""
import pandas as pd
import os
import time

BASE = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾'
PANEL_CSV = os.path.join(BASE, '县域统计年鉴', '中国县域统计面板数据_最终版.csv')
OUT_DIR = os.path.join(BASE, '高阳县面板数据')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COUNTY = '高阳县'

print(f"提取{TARGET_COUNTY}数据...")

# 加载全国面板数据
df = pd.read_csv(PANEL_CSV, encoding='utf-8-sig')

# 提取高阳县数据
gy = df[df['county'] == TARGET_COUNTY].copy()
gy = gy.sort_values('year')

print(f"  原始数据: {len(gy)}行")
print(f"  年份: {sorted(gy['year'].astype(int).unique())}")

# 提取指标列
indicator_cols = [c for c in df.columns if c not in ['year', 'province', 'county']]
print(f"  指标数: {len(indicator_cols)}")

# 计算各年份的指标覆盖率
print("\n  各年份指标覆盖率:")
for y in sorted(gy['year'].unique()):
    gy_y = gy[gy['year'] == y]
    available = sum(1 for col in indicator_cols if gy_y[col].notna().any())
    print(f"    {int(y)}: {available}/{len(indicator_cols)} ({available/len(indicator_cols)*100:.1f}%)")

# 生成高阳县专属面板数据
gy_panel = gy.copy()
gy_panel['year'] = gy_panel['year'].astype(int)
gy_panel = gy_panel.set_index('year')

# 保存CSV
OUT_CSV = os.path.join(OUT_DIR, f'{TARGET_COUNTY}面板数据.csv')
gy_panel.to_csv(OUT_CSV, encoding='utf-8-sig')
print(f"\n  保存CSV: {OUT_CSV}")

# 生成Excel（含多个Sheet）
OUT_XLSX = os.path.join(OUT_DIR, f'{TARGET_COUNTY}面板数据.xlsx')
print(f"  生成Excel: {OUT_XLSX}")

with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
    # Sheet 1: 主面板数据
    gy_panel.to_excel(writer, sheet_name='面板数据')
    
    # Sheet 2: 指标说明
    indicator_info = pd.DataFrame({'指标名称': indicator_cols})
    indicator_info.to_excel(writer, sheet_name='指标列表', index=False)
    
    # Sheet 3: 描述性统计
    desc = gy_panel[indicator_cols].describe().T
    desc.index.name = '指标'
    desc.to_excel(writer, sheet_name='描述性统计')
    
    # Sheet 4: 数据可用性矩阵
    availability = pd.DataFrame(index=sorted(gy['year'].astype(int).unique()), columns=indicator_cols)
    for y in sorted(gy['year'].unique()):
        gy_y = gy[gy['year'] == y]
        for col in indicator_cols:
            availability.loc[int(y), col] = '有' if gy_y[col].notna().any() else ''
    availability.to_excel(writer, sheet_name='数据可用性')
    
    # Sheet 5: 时间趋势摘要
    trend_data = []
    for col in indicator_cols:
        vals = gy[col].dropna()
        if len(vals) >= 2:
            years = gy.loc[vals.index, 'year'].astype(int).values
            first = vals.iloc[0]
            last = vals.iloc[-1]
            change = ((last - first) / first * 100) if first != 0 else 0
            trend = '上升' if last > first else '下降' if last < first else '平稳'
            trend_data.append({
                '指标': col,
                '起始年': years[0],
                '结束年': years[-1],
                '起始值': first,
                '结束值': last,
                '增长率(%)': change,
                '趋势': trend
            })
    pd.DataFrame(trend_data).to_excel(writer, sheet_name='时间趋势', index=False)

print(f"\n{TARGET_COUNTY}面板数据生成完成!")
print(f"  文件位置: {OUT_DIR}")
