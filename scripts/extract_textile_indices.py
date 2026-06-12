"""Pipeline 3: 纺织指数提取 + 年度化 + 构念效度验证"""
import pandas as pd
import os
import glob

INDICES_DIR = "data/textile_indices"
OUTPUT = "output/textile_indices_annual.csv"

def read_index_xlsx(filepath):
    """读取纺织指数Excel，返回季度/月度DataFrame"""
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except:
        return None
    if df.shape[1] < 2:
        return None
    df.columns = ['date', 'value']
    # 跳过基期行
    df = df[df['date'].astype(str).str.contains(r'\d{4}', na=False)].copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    return df

def to_annual_quarterly(df, name):
    """季度数据 → 年度均值"""
    df = df.copy()
    df['date_str'] = df['date'].astype(str)
    # 提取年份和季度
    df['year'] = df['date_str'].str.extract(r'(\d{4})').astype(int)
    df['quarter'] = df['date_str'].str.extract(r'[Qq](\d)').astype(int)
    annual = df.groupby('year')['value'].mean().reset_index()
    annual.columns = ['Year', name]
    return annual

def to_annual_monthly(df, name):
    """月度数据 → 年度均值"""
    df = df.copy()
    df['date_str'] = df['date'].astype(str)
    df['year'] = df['date_str'].str.extract(r'(\d{4})').astype(int)
    annual = df.groupby('year')['value'].mean().reset_index()
    annual.columns = ['Year', name]
    return annual

def main():
    files = glob.glob(os.path.join(INDICES_DIR, "*.xlsx"))
    print(f"找到 {len(files)} 个纺织指数文件")

    # 名称映射
    name_map = {
        '产业发展指数': 'index_industry_development',
        '产业景气指数': 'index_industry_prosperity',
        '产业竞争力指数': 'index_competitiveness',
        '产业规模指数': 'index_industry_scale',
        '产品价格指数': 'index_product_price',
        '人才打造指数': 'index_talent',
        '半成品价格指数': 'index_semifinished_price',
        '品牌运营指数': 'index_brand_operation',
        '成品价格指数': 'index_finished_price',
        '政策支持指数': 'index_policy_support',
        '科技创新指数': 'index_tech_innovation',
        '经济效益指数': 'index_economic_benefit',
        '营销能力指数': 'index_marketing',
        '转型发展指数': 'index_transformation',
        '集约经营指数': 'index_intensive_operation',
    }

    # 月度指数的文件名关键词
    monthly_indices = {'产品价格指数', '半成品价格指数', '成品价格指数'}

    annual_frames = []
    for fpath in files:
        fname = os.path.basename(fpath).replace('.xlsx', '')
        df = read_index_xlsx(fpath)
        if df is None or len(df) == 0:
            print(f"  [跳过] {fname}")
            continue

        # 匹配中文名
        matched_name = None
        for cn, en in name_map.items():
            if cn in fname:
                matched_name = en
                break
        if matched_name is None:
            matched_name = fname

        if any(mi in fname for mi in monthly_indices):
            annual = to_annual_monthly(df, matched_name)
            print(f"  {fname} → {matched_name} (月度, {len(annual)}年)")
        else:
            annual = to_annual_quarterly(df, matched_name)
            print(f"  {fname} → {matched_name} (季度, {len(annual)}年)")

        annual_frames.append(annual)

    # 合并所有指数
    result = annual_frames[0]
    for af in annual_frames[1:]:
        result = result.merge(af, on='Year', how='outer')
    result = result.sort_values('Year').reset_index(drop=True)

    os.makedirs("output", exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n合并后: {len(result)} 年 x {len(result.columns)} 列")
    print(f"年份范围: {int(result['Year'].min())}-{int(result['Year'].max())}")
    print(f"已保存至 {OUTPUT}")

    # === 构念效度验证: 官方政策支持指数 vs LLM评分 ===
    print("\n=== 构念效度验证 ===")
    if 'index_policy_support' in result.columns:
        ps = pd.read_csv("output/policy_scores_panel.csv", encoding='utf-8-sig')
        gy_ps = ps[ps['County_Code'].astype(str) == '130628'][['Year', 'policy_intensity_total']].copy()

        merged = result.merge(gy_ps, on='Year', how='inner')
        if len(merged) >= 3:
            corr = merged['index_policy_support'].corr(merged['policy_intensity_total'])
            print(f"  官方政策支持指数 vs LLM综合评分: r = {corr:.4f} (n={len(merged)})")
            if abs(corr) > 0.5:
                print(f"  [通过] 构念效度良好 (|r| > 0.5)")
            else:
                print(f"  [警告] 相关系数偏低，LLM评分可能与官方指数测量不同构念")
        else:
            print(f"  重叠年份不足 (n={len(merged)})")
    else:
        print("  未找到政策支持指数")

if __name__ == "__main__":
    main()
