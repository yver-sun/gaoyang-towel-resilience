"""
模块2: 宏观面板组装 - 县域统计年鉴解密与提取
功能：从加密的《中国县域统计年鉴》XLS中提取河北省所有县的宏观指标
输入：data/yearbooks/ 下的加密XLS文件（县域统计年鉴 2000-2024）
输出：output/hebei_counties_macro_panel.csv
依赖：pip install pandas msoffcrypto-tool openpyxl
"""
import os
import re
import io
import pandas as pd
import msoffcrypto

# ================= 配置区 =================
YEARBOOK_DIR = "data/yearbooks"
OUTPUT_FILE = "output/hebei_counties_macro_panel.csv"

# 需要提取的核心指标（列名关键词映射）
INDICATOR_KEYWORDS = {
    "gdp_total": ["地区生产总值", "GDP", "生产总值"],
    "industry_above_scale": ["规模以上工业", "规上工业", "规模以上工业总产值", "规上工业企业数", "规模以上工业企业"],
    "resident_population": ["常住人口", "年末常住人口", "户籍人口"],
    "import_export": ["进出口", "货物进出口", "进出口总额"],
    "fiscal_revenue": ["一般公共预算收入", "地方财政收入", "一般公共预算"],
    "fiscal_expenditure": ["一般公共预算支出", "地方财政支出"],
    "finance_deposit": ["金融机构存款", "金融机构各项存款", "本外币存款"],
    "finance_loan": ["金融机构贷款", "金融机构各项贷款", "本外币贷款"],
    "fixed_asset_invest": ["固定资产投资", "全社会固定资产投资"],
    "retail_sales": ["社会消费品零售", "消费品零售总额"],
    "primary_industry": ["第一产业", "农林牧渔"],
    "secondary_industry": ["第二产业", "工业"],
    "tertiary_industry": ["第三产业", "服务业"],
    "urban_income": ["城镇居民人均可支配收入", "城镇居民收入"],
    "rural_income": ["农村居民人均可支配收入", "农民人均纯收入"],
}

# 年鉴密码（常见密码，可根据实际情况修改）
PASSWORDS = ["", "123456", "csy", "county", "county2024"]

# ================= 函数定义 =================
def decrypt_excel(file_path, passwords):
    """尝试用多个密码解密加密的XLS文件"""
    with open(file_path, 'rb') as f:
        for pwd in passwords:
            try:
                f.seek(0)
                crypto = msoffcrypto.OfficeFile(f)
                if pwd:
                    crypto.load_key(password=pwd)
                decrypted = io.BytesIO()
                crypto.decrypt(decrypted)
                decrypted.seek(0)
                return pd.read_excel(decrypted, engine='openpyxl')
            except Exception:
                continue
    return None

def find_column_by_keywords(df, keywords):
    """通过关键词在DataFrame中查找匹配的列"""
    for col in df.columns:
        col_str = str(col).strip()
        for kw in keywords:
            if kw in col_str:
                return col
    return None

def extract_county_name_from_df(df):
    """从DataFrame中提取县名列"""
    for col in df.columns:
        sample = df[col].dropna().head(5).astype(str)
        if sample.str.contains('县|区|市').sum() >= 2:
            return col
    return None

def process_yearbook(file_path):
    """处理单个年鉴文件，提取指标数据"""
    df = decrypt_excel(file_path, PASSWORDS)
    if df is None:
        print(f"  [失败] 无法解密: {file_path}")
        return None, None
    
    county_col = extract_county_name_from_df(df)
    if county_col is None:
        print(f"  [失败] 未找到县名列: {file_path}")
        return None, None
    
    extracted = {ind: find_column_by_keywords(df, kws) for ind, kws in INDICATOR_KEYWORDS.items()}
    
    output_cols = {ind: col for ind, col in extracted.items() if col is not None}
    
    if not output_cols:
        print(f"  [失败] 未找到任何指标列: {file_path}")
        return None, None
    
    selected_cols = [county_col] + list(output_cols.values())
    result = df[[county_col] + list(output_cols.values())].copy()
    result.columns = ['County_Name'] + list(output_cols.keys())
    
    year_match = re.search(r'(\d{4})', os.path.basename(file_path))
    if year_match:
        result['Year'] = int(year_match.group(1))
    
    return result, output_cols

def process_all_yearbooks():
    """批量处理所有年鉴文件"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(YEARBOOK_DIR):
        os.makedirs(YEARBOOK_DIR)
        print(f"已创建 {YEARBOOK_DIR} 目录")
        print("请将《中国县域统计年鉴》XLS文件放入该目录后重新运行。")
        return

    files = sorted([f for f in os.listdir(YEARBOOK_DIR) if f.endswith(('.xls', '.xlsx'))])
    if not files:
        print(f"{YEARBOOK_DIR} 目录中没有找到 XLS 文件。")
        return

    print(f"找到 {len(files)} 个年鉴文件")
    
    all_data = []
    year_columns = {}
    
    for file in files:
        file_path = os.path.join(YEARBOOK_DIR, file)
        print(f"处理: {file}")
        result, cols = process_yearbook(file_path)
        if result is not None:
            all_data.append(result)
            year_match = re.search(r'(\d{4})', file)
            if year_match:
                year_columns[int(year_match.group(1))] = cols

    if not all_data:
        print("未能从任何年鉴文件中提取数据。")
        return

    df_combined = pd.concat(all_data, ignore_index=True)
    df_hebei = df_combined[df_combined['County_Name'].str.contains('河北|高阳|保定', na=False)].copy()
    
    df_hebei = df_hebei.sort_values(['County_Name', 'Year']).reset_index(drop=True)
    
    county_name_to_code = {}
    for idx, row in df_hebei.iterrows():
        name = row['County_Name']
        if name not in county_name_to_code:
            if '高阳' in str(name):
                county_name_to_code[name] = '130628'
            else:
                county_name_to_code[name] = f'13{idx:06d}'
    
    df_hebei['County_Code'] = df_hebei['County_Name'].map(county_name_to_code)
    
    for col in df_hebei.select_dtypes(include=['object']).columns:
        if col not in ['County_Name', 'County_Code']:
            df_hebei[col] = pd.to_numeric(df_hebei[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    
    df_hebei.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n处理完成！")
    print(f"共提取 {len(df_hebei)} 条记录")
    print(f"县数: {df_hebei['County_Name'].nunique()}")
    print(f"年份范围: {df_hebei['Year'].min()} - {df_hebei['Year'].max()}")
    print(f"结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_yearbooks()
