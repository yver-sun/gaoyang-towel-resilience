"""
模块3b: 高阳县注册企业数据提取（从全国区县细分行业数据中提取）
功能：从883万行全国数据中提取高阳县专属的注册企业时间序列，用于产业韧性研究
输入：data/panel/区县细分行业-注册企业数据（1949-2024年）.csv
输出：
  - output/gaoyang_enterprise_registration.csv（高阳县年度汇总）
  - output/gaoyang_textile_registration.csv（高阳县纺织业年度汇总）
  - output/baoding_enterprise_registration.csv（保定市24区县汇总，供SDiD供体池）
依赖：pip install pandas
"""
import os
import pandas as pd

RAW_FILE = "data/panel/区县细分行业-注册企业数据（1949-2024年）.csv"
OUTPUT_DIR = "output"

def extract_enterprise_data():
    """从全国区县数据中提取高阳县和保定市的注册企业数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_FILE):
        print(f"错误：未找到 {RAW_FILE}")
        return

    print("加载全国区县注册企业数据（883万行）...")
    df = pd.read_csv(RAW_FILE, encoding='utf-8-sig')
    cols = [c.strip() for c in df.columns if c.strip()]
    df.columns = cols
    df = df[cols]

    # ==================== 高阳县提取 ====================
    print("\n提取高阳县数据...")
    gy = df[df['区县名称'].str.contains('高阳', na=False)].copy()
    print(f"  高阳县原始记录: {len(gy)} 行")
    print(f"  年份范围: {int(gy['注册年份'].min())} - {int(gy['注册年份'].max())}")

    # 按年份汇总（全部行业）
    gy_annual = gy.groupby('注册年份').agg(
        total_firms=('注册企业数', 'sum')
    ).reset_index()
    gy_annual.columns = ['Year', 'total_firms']
    gy_annual['County_Code'] = '130628'

    # 按年份+一级行业汇总
    gy_industry = gy.groupby(['注册年份', '一级行业'])['注册企业数'].sum().reset_index()
    gy_industry.columns = ['Year', 'industry_level1', 'firm_count']

    # 纺织业提取
    print("  提取纺织业数据...")
    textile = gy[gy['一级行业'] == '制造业'].copy()
    textile = textile[textile['二级行业'].str.contains('纺织', na=False)]
    print(f"  纺织业原始记录: {len(textile)} 行")
    print(f"  纺织业总企业数: {int(textile['注册企业数'].sum())}")

    textile_annual = textile.groupby('注册年份').agg(
        textile_firms=('注册企业数', 'sum')
    ).reset_index()
    textile_annual.columns = ['Year', 'textile_firms']

    # 纺织业细分：二级行业年度时间序列
    textile_sec = textile.groupby(['注册年份', '二级行业'])['注册企业数'].sum().reset_index()
    textile_sec.columns = ['Year', 'industry_level2', 'firm_count']

    # 纺织业细分：三级行业年度时间序列
    textile_ter = textile.groupby(['注册年份', '三级行业'])['注册企业数'].sum().reset_index()
    textile_ter.columns = ['Year', 'industry_level3', 'firm_count']

    # 合并高阳县年度面板
    gy_panel = gy_annual.merge(textile_annual, on='Year', how='left')
    gy_panel['textile_firms'] = gy_panel['textile_firms'].fillna(0).astype(int)

    # 计算纺织业占比
    gy_panel['textile_ratio'] = gy_panel['textile_firms'] / gy_panel['total_firms']

    # 新增注册数（当年新增 vs 累计）
    gy_panel['new_total_firms'] = gy_panel['total_firms']
    gy_panel['new_textile_firms'] = gy_panel['textile_firms']

    # 移动平均（3年）
    gy_panel['textile_firms_ma3'] = gy_panel['textile_firms'].rolling(3, min_periods=1).mean()

    gy_panel['Year'] = gy_panel['Year'].astype(int)

    gy_annual['Year'] = gy_annual['Year'].astype(int)
    textile_annual['Year'] = textile_annual['Year'].astype(int)

    textile_sec['Year'] = textile_sec['Year'].astype(int)

    textile_ter['Year'] = textile_ter['Year'].astype(int)

    # 保存高阳县年度面板
    gy_out = gy_panel[['County_Code', 'Year', 'total_firms', 'textile_firms',
                        'textile_ratio', 'new_total_firms', 'new_textile_firms',
                        'textile_firms_ma3']]
    gy_file = os.path.join(OUTPUT_DIR, "gaoyang_enterprise_registration.csv")
    gy_out.to_csv(gy_file, index=False, encoding='utf-8-sig')
    print(f"\n  高阳县年度面板已保存至 {gy_file}")
    print(f"  字段: {gy_out.columns.tolist()}")

    # 保存纺织业细分
    textile_file = os.path.join(OUTPUT_DIR, "gaoyang_textile_registration.csv")
    textile_sec.to_csv(textile_file, index=False, encoding='utf-8-sig')
    print(f"  纺织业二级行业时间序列已保存至 {textile_file}")

    # ==================== 保定市提取 ====================
    print("\n提取保定市24区县数据...")
    bd = df[df['所属城市'].str.contains('保定', na=False)].copy()
    print(f"  保定市原始记录: {len(bd)} 行")
    print(f"  区县数: {bd['区县名称'].nunique()}")

    # 按区县+年份汇总
    bd_annual = bd.groupby(['区县名称', '注册年份']).agg(
        total_firms=('注册企业数', 'sum')
    ).reset_index()
    bd_annual.columns = ['County_Name', 'Year', 'total_firms']

    # 纺织业提取
    bd_mfg = bd[bd['一级行业'] == '制造业'].copy()
    bd_textile = bd_mfg[bd_mfg['二级行业'].str.contains('纺织', na=False)]
    bd_textile_annual = bd_textile.groupby(['区县名称', '注册年份']).agg(
        textile_firms=('注册企业数', 'sum')
    ).reset_index()
    bd_textile_annual.columns = ['County_Name', 'Year', 'textile_firms']

    bd_panel = bd_annual.merge(bd_textile_annual, on=['County_Name', 'Year'], how='left')
    bd_panel['textile_firms'] = bd_panel['textile_firms'].fillna(0).astype(int)
    bd_panel['textile_ratio'] = bd_panel['textile_firms'] / bd_panel['total_firms']

    bd_file = os.path.join(OUTPUT_DIR, "baoding_enterprise_registration.csv")
    bd_panel.to_csv(bd_file, index=False, encoding='utf-8-sig')
    print(f"  保定市24区县年度面板已保存至 {bd_file}")

    # ==================== 打印摘要 ====================
    print("\n" + "=" * 60)
    print("【高阳县纺织业年度趋势】")
    print("=" * 60)
    recent = gy_panel[gy_panel['Year'] >= 2000][['Year', 'total_firms', 'textile_firms', 'textile_ratio']].to_string(index=False)
    print(recent)

    print("\n" + "=" * 60)
    print("【保定市纺织业区县排名】")
    print("=" * 60)
    bd_rank = bd_panel.groupby('County_Name')['textile_firms'].sum().sort_values(ascending=False)
    for county, count in bd_rank.items():
        print(f"  {county}: {int(count)}")

    print(f"\n数据提取完成！共生成 3 个输出文件。")

if __name__ == "__main__":
    extract_enterprise_data()
