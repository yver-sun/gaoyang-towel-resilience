"""Pipeline 2: 政策文件年度计数 + 分类计数"""
import pandas as pd
import os
import re

POLICY_CSV = "data/policies/高阳县毛巾产业政策文件_全量_20260517_195152.csv"
OUTPUT = "output/policy_document_counts.csv"

def parse_year(date_str):
    """从发文时间中提取年份"""
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    m = re.search(r'(\d{4})', s)
    return int(m.group(1)) if m else None

def main():
    df = pd.read_csv(POLICY_CSV, encoding='utf-8-sig')
    print(f"加载政策文件: {len(df)} 篇")

    df['year'] = df['发文时间'].apply(parse_year)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    print(f"有效年份记录: {len(df)} ({df['year'].min()}-{df['year'].max()})")

    # 年度总计数
    annual = df.groupby('year').size().reset_index(name='policy_doc_count')
    annual.columns = ['Year', 'policy_doc_count']

    # 分类计数：提取每个政策类别标签
    all_categories = set()
    for cats in df['政策类别'].dropna():
        for c in str(cats).replace('、', ',').split(','):
            c = c.strip()
            if c:
                all_categories.add(c)

    # 定义关注的类别组
    category_groups = {
        'env_policy_count': ['环保治理', '循环经济', '节能减排', '绿色发展'],
        'ecommerce_policy_count': ['电子商务', '数字经济', '跨境电商'],
        'brand_policy_count': ['品牌建设', '质量标准', '区域品牌'],
        'cluster_policy_count': ['产业集群', '产业升级', '园区建设'],
        'equipment_policy_count': ['设备升级', '技术改造', '数字化转型', '智能制造'],
    }

    for col_name, keywords in category_groups.items():
        annual[col_name] = 0
        for _, row in df.iterrows():
            y = row['year']
            cats_str = str(row['政策类别']) if pd.notna(row['政策类别']) else ''
            if any(kw in cats_str for kw in keywords):
                annual.loc[annual['Year'] == y, col_name] += 1

    # 完整年份范围
    full_years = pd.DataFrame({'Year': range(2000, 2027)})
    annual = full_years.merge(annual, on='Year', how='left').fillna(0)
    for col in category_groups.keys():
        annual[col] = annual[col].astype(int)
    annual['policy_doc_count'] = annual['policy_doc_count'].astype(int)

    os.makedirs("output", exist_ok=True)
    annual.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n年度政策计数:")
    print(annual.to_string(index=False))
    print(f"\n已保存至 {OUTPUT}")

if __name__ == "__main__":
    main()
