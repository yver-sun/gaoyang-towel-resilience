"""
========================================================================
  高阳县社会经济数据 - 描述性分析 + 时间趋势
========================================================================
包含：
  1. 基础描述性统计（均值、标准差、极值、变异系数等）
  2. 各指标数据结构分析
  3. 逐年时间趋势可视化
  4. 指标分组分析（经济、财政、产业、人口、教育、医疗等）
========================================================================
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
BASE = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾'
PANEL_CSV = os.path.join(BASE, '县域统计年鉴', '中国县域统计面板数据_最终版.csv')
OUT_DIR = os.path.join(BASE, 'figures', '高阳县分析')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COUNTY = '高阳县'

# 指标分组
INDICATOR_GROUPS = {
    '经济总量': ['gdp', 'primary_industry_va', 'secondary_industry_va', 'tertiary_industry_va',
                 'industry_gross_output', 'agriculture_va', 'animal_husbandry_va'],
    '工业': ['above_scale_ind_firms', 'above_scale_industry_output', 'industry_va'],
    '财政': ['local_fiscal_revenue', 'local_fiscal_expenditure', 'tax_revenue',
             'public_fiscal_revenue', 'public_fiscal_expenditure', 'fiscal_expenditure'],
    '金融': ['deposit_balance', 'loan_balance'],
    '投资消费': ['fixed_asset_invest', 'urban_fixed_invest', 'retail_sales',
                  'capital_construction_invest'],
    '农业': ['grain_output', 'cotton_output', 'oilseed_output', 'meat_output',
             'agri_machinery_power', 'grain_sowing_area', 'crop_sowing_area'],
    '人口': ['total_pop', 'rural_pop', 'registered_pop', 'total_households',
             'rural_households'],
    '就业': ['rural_employment', 'agri_employment', 'secondary_employment',
             'tertiary_employment'],
    '教育': ['primary_school_students', 'middle_school_students', 'voc_edu_students',
             'total_students'],
    '医疗': ['hospital_beds'],
    '行政': ['township_count', 'village_committee_count', 'area_km2'],
    '基础设施': ['fixed_phone', 'local_phone', 'rural_electricity'],
    '民生': ['welfare_institutions', 'welfare_beds'],
}


# ==================== 1. 加载数据 ====================
print("=" * 70)
print(f"  {TARGET_COUNTY} 数据分析")
print("=" * 70)

df = pd.read_csv(PANEL_CSV, encoding='utf-8-sig')
gy = df[df['county'] == TARGET_COUNTY].copy()

print(f"\n[数据概况]")
print(f"  年份: {sorted(gy['year'].astype(int).unique())}")
print(f"  省份: {gy['province'].unique()[0]}")

# 提取所有可用的指标
indicator_cols = [c for c in df.columns if c not in ['year', 'province', 'county']]
gy_data = {}
for col in indicator_cols:
    vals = gy[col].dropna()
    if len(vals) > 0:
        gy_data[col] = vals

print(f"  有数据的指标: {len(gy_data)} 个")


# ==================== 2. 基础描述性统计 ====================
print(f"\n{'='*70}")
print("  描述性统计")
print(f"{'='*70}")

desc_rows = []
for col, vals in sorted(gy_data.items()):
    years = gy.loc[vals.index, 'year'].astype(int)
    stats = {
        '指标': col,
        '年数': len(vals),
        '起始年': years.min(),
        '结束年': years.max(),
        '均值': vals.mean(),
        '标准差': vals.std(),
        '最小值': vals.min(),
        '最大值': vals.max(),
        '变异系数': f"{(vals.std()/vals.mean()*100):.1f}%" if vals.mean() != 0 else 'N/A',
        '极差': vals.max() - vals.min(),
    }
    desc_rows.append(stats)

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False, float_format='%.1f'))

# 保存描述性统计
desc_csv = os.path.join(OUT_DIR, f'{TARGET_COUNTY}_描述性统计.csv')
desc_df.to_csv(desc_csv, index=False, encoding='utf-8-sig')
print(f"\n  已保存: {desc_csv}")


# ==================== 3. 指标分组统计 ====================
print(f"\n{'='*70}")
print("  指标分组统计")
print(f"{'='*70}")

for group_name, indicators in INDICATOR_GROUPS.items():
    available = [i for i in indicators if i in gy_data]
    if available:
        print(f"\n  【{group_name}】({len(available)}个指标)")
        for ind in available:
            vals = gy_data[ind]
            years = gy.loc[vals.index, 'year'].astype(int)
            trend = '上升' if vals.iloc[-1] > vals.iloc[0] else '下降' if vals.iloc[-1] < vals.iloc[0] else '平稳'
            print(f"    {ind}: {vals.min():.1f} - {vals.max():.1f} ({years.min()}-{years.max()}), 趋势: {trend}")


# ==================== 4. 时间趋势可视化 ====================
print(f"\n{'='*70}")
print("  生成时间趋势图表...")
print(f"{'='*70}")

# 图1：所有有数据的指标时间趋势（分组）
for group_name, indicators in INDICATOR_GROUPS.items():
    available = [i for i in indicators if i in gy_data]
    if not available:
        continue
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ind in available:
        vals = gy_data[ind]
        years = gy.loc[vals.index, 'year'].astype(int).values
        ax.plot(years, vals.values, marker='o', linewidth=2, markersize=8, label=ind)
    
    ax.set_title(f'{TARGET_COUNTY} - {group_name} 时间趋势', fontsize=14, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(sorted(gy['year'].astype(int).unique()), rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(OUT_DIR, f'{TARGET_COUNTY}_{group_name}_时间趋势.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {group_name}: {save_path}")


# 图2：综合概览（核心指标）
core_indicators = ['gdp', 'secondary_industry_va', 'agriculture_va', 'local_fiscal_revenue', 'deposit_balance']
core_available = [i for i in core_indicators if i in gy_data]

if core_available:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ind in enumerate(core_available):
        vals = gy_data[ind]
        years = gy.loc[vals.index, 'year'].astype(int).values
        ax = axes[idx]
        ax.plot(years, vals.values, marker='o', linewidth=2, markersize=8, color='#2196F3')
        ax.set_title(f'{ind}', fontsize=12, fontweight='bold')
        ax.set_xlabel('年份', fontsize=10)
        ax.grid(True, alpha=0.3)
        # 标注数值
        for i, (y, v) in enumerate(zip(years, vals.values)):
            ax.annotate(f'{v:.0f}', (y, v), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=8)
    
    for idx in range(len(core_available), 6):
        axes[idx].axis('off')
    
    plt.suptitle(f'{TARGET_COUNTY} - 核心指标概览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f'{TARGET_COUNTY}_核心指标概览.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  核心概览: {save_path}")


# 图3：产业结构变化
fig, ax = plt.subplots(figsize=(10, 6))
industry_available = []
for ind in ['primary_industry_va', 'secondary_industry_va', 'tertiary_industry_va']:
    if ind in gy_data:
        industry_available.append(ind)

if len(industry_available) >= 2:
    # 堆叠面积图
    years_all = sorted(gy['year'].unique())
    data_dict = {}
    for ind in industry_available:
        vals = gy_data[ind]
        years = gy.loc[vals.index, 'year'].values
        data_dict[ind] = dict(zip(years, vals.values))
    
    plot_years = sorted(set().union(*[d.keys() for d in data_dict.values()]))
    
    for ind in industry_available:
        vals = [data_dict[ind].get(y, 0) for y in plot_years]
        ax.plot(plot_years, vals, marker='o', linewidth=2, markersize=8, label=ind)
    
    ax.set_title(f'{TARGET_COUNTY} - 产业结构变化', fontsize=14, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('增加值', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(plot_years, rotation=45)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f'{TARGET_COUNTY}_产业结构.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  产业结构: {save_path}")


# 图4：柱状图 - 各年份指标数量
fig, ax = plt.subplots(figsize=(8, 5))
year_indicator_count = {}
for y in sorted(gy['year'].unique()):
    gy_y = gy[gy['year'] == y]
    count = sum(1 for col in indicator_cols if gy_y[col].notna().any())
    year_indicator_count[int(y)] = count

ax.bar(year_indicator_count.keys(), year_indicator_count.values(), color='#4CAF50', edgecolor='white')
ax.set_title(f'{TARGET_COUNTY} - 各年份可用指标数', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('指标数', fontsize=12)
for x, y in zip(year_indicator_count.keys(), year_indicator_count.values()):
    ax.text(x, y, str(y), ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.xticks(sorted(year_indicator_count.keys()))
plt.tight_layout()
save_path = os.path.join(OUT_DIR, f'{TARGET_COUNTY}_各年指标数.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  各年指标数: {save_path}")

print(f"\n{'='*70}")
print(f"所有图表已保存到: {OUT_DIR}")
print(f"{'='*70}")
