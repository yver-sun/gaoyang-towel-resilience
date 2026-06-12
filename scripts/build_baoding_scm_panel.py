"""Pipeline 4: 保定24区县对照面板 - 用于合成控制法(SCM)"""
import pandas as pd
import numpy as np
import os

BAODING_CSV = "output/baoding_enterprise_registration.csv"
OUTPUT = "output/baoding_county_panel.csv"

# 保定24区县名称→代码映射 (130600为保定市)
COUNTY_CODE_MAP = {
    '高阳县': '130628',
    '竞秀区': '130602',
    '莲池区': '130606',
    '满城区': '130607',
    '清苑区': '130608',
    '徐水区': '130609',
    '涞水县': '130623',
    '阜平县': '130624',
    '定兴县': '130626',
    '唐县': '130627',
    '容城县': '130629',
    '涞源县': '130630',
    '望都县': '130631',
    '安新县': '130632',
    '易县': '130633',
    '曲阳县': '130634',
    '蠡县': '130635',
    '顺平县': '130636',
    '博野县': '130637',
    '雄县': '130638',
    '涿州市': '130681',
    '定州市': '130682',
    '安国市': '130683',
    '高碑店市': '130684',
}


def main():
    df = pd.read_csv(BAODING_CSV, encoding='utf-8-sig')
    df['Year'] = df['Year'].astype(int)

    # 映射County_Code
    df['County_Code'] = df['County_Name'].map(COUNTY_CODE_MAP)
    unmatched = df[df['County_Code'].isna()]['County_Name'].unique()
    if len(unmatched) > 0:
        print(f"未匹配区县: {list(unmatched)}")

    # 过滤2000-2024（完整覆盖期）
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)].copy()
    print(f"2000-2024年数据: {len(df)} 行, {df['County_Code'].nunique()} 区县")

    # === 构建平衡面板 ===
    counties = sorted(df['County_Code'].unique())
    years = list(range(2000, 2025))
    full_index = pd.MultiIndex.from_product(
        [counties, years], names=['County_Code', 'Year']
    )
    panel = pd.DataFrame(index=full_index).reset_index()

    panel = panel.merge(
        df[['County_Code', 'Year', 'County_Name', 'total_firms', 'textile_firms', 'textile_ratio']],
        on=['County_Code', 'Year'], how='left'
    )

    # 填充County_Name
    name_lookup = df.groupby('County_Code')['County_Name'].first().to_dict()
    panel['County_Name'] = panel['County_Code'].map(name_lookup)

    # 用0填充缺失的纺织企业数（表示该年无注册）
    panel['total_firms'] = panel['total_firms'].fillna(0)
    panel['textile_firms'] = panel['textile_firms'].fillna(0)
    panel['textile_ratio'] = panel['textile_ratio'].fillna(0)

    # === 衍生变量 ===
    panel = panel.sort_values(['County_Code', 'Year'])

    # 纺织企业3年移动平均
    panel['textile_firms_ma3'] = panel.groupby('County_Code')['textile_firms'].transform(
        lambda x: x.rolling(3, min_periods=1, center=True).mean()
    )

    # 纺织企业占比3年移动平均
    panel['textile_ratio_ma3'] = panel.groupby('County_Code')['textile_ratio'].transform(
        lambda x: x.rolling(3, min_periods=1, center=True).mean()
    )

    # 纺织企业增速 (YoY)
    panel['textile_firms_growth'] = panel.groupby('County_Code')['textile_firms'].pct_change()
    panel['textile_firms_growth'] = panel['textile_firms_growth'].replace([np.inf, -np.inf], np.nan)

    # 总企业增速
    panel['total_firms_growth'] = panel.groupby('County_Code')['total_firms'].pct_change()
    panel['total_firms_growth'] = panel['total_firms_growth'].replace([np.inf, -np.inf], np.nan)

    # 标记高阳县（处理组）和其他县（对照组）
    panel['is_treated'] = (panel['County_Code'] == '130628').astype(int)
    panel['post_2017'] = (panel['Year'] >= 2017).astype(int)

    # === 输出 ===
    os.makedirs("output", exist_ok=True)
    panel.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

    # === 摘要统计 ===
    gy = panel[panel['County_Code'] == '130628']
    others = panel[panel['County_Code'] != '130628']

    print(f"\n高阳县(处理组): {len(gy)}年, 纺织企业均值={gy['textile_firms'].mean():.1f}")
    print(f"对照县: {others['County_Code'].nunique()}个, 纺织企业均值={others['textile_firms'].mean():.1f}")

    # pre-2017趋势对比
    pre = panel[panel['Year'] < 2017]
    gy_pre = pre[pre['County_Code'] == '130628']['textile_firms'].mean()
    other_pre = pre[pre['County_Code'] != '130628']['textile_firms'].mean()
    print(f"Pre-2017 高阳纺织企业均值: {gy_pre:.1f}, 对照县均值: {other_pre:.1f}")

    post = panel[panel['Year'] >= 2017]
    gy_post = post[post['County_Code'] == '130628']['textile_firms'].mean()
    other_post = post[post['County_Code'] != '130628']['textile_firms'].mean()
    print(f"Post-2017 高阳纺织企业均值: {gy_post:.1f}, 对照县均值: {other_post:.1f}")

    print(f"\n已保存至 {OUTPUT}")
    print(f"面板: {len(panel)}行 x {len(panel.columns)}列")


if __name__ == "__main__":
    main()
