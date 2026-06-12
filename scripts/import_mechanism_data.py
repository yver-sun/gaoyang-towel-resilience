"""
模块3: 微观机制变量 - 高阳专属指标导入
功能：将调研获取的底层台账数据转化为标准时间序列CSV
输入：data/micro_data/ 下的调研台账（Excel/CSV格式）
输出：output/gaoyang_micro_mechanism.csv
依赖：pip install pandas openpyxl
"""
import os
import pandas as pd

MICRO_DATA_DIR = "data/micro_data"
OUTPUT_FILE = "output/gaoyang_micro_mechanism.csv"

REQUIRED_COLUMNS = [
    "Year",
    "express_delivery_volume",       # 快递年发件量（万件）
    "cotton_yarn_throughput",         # 棉纱交易吞吐量（万吨）
    "centralized_steam_volume",       # 集中供汽量（万吨）
    "eco_park_area",                  # 生态印染园区面积（万平方米）
    "sewage_treatment_capacity",      # 污水集中处理能力（万吨/日）
    "brand_authorized_firms",         # 区域品牌授权企业数（家）
    "textile_enterprises_count",      # 纺织企业总数（家）
    "above_scale_textile_firms",      # 规上纺织企业数（家）
    "export_volume_textile",          # 毛巾出口额（万美元）
    "ecommerce_shipments",            # 电商发件量（万件）
]

def create_template():
    """创建空白模板供手动填入调研数据"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    template_path = os.path.join(MICRO_DATA_DIR, "gaoyang_micro_data_template.xlsx")
    os.makedirs(MICRO_DATA_DIR, exist_ok=True)
    template.to_excel(template_path, index=False, engine='openpyxl')
    print(f"模板已创建: {template_path}")
    print("请按照以下字段填入调研数据：")
    for col in REQUIRED_COLUMNS:
        print(f"  - {col}")

def load_and_validate():
    """从Excel/CSV加载数据并验证"""
    if not os.path.exists(MICRO_DATA_DIR):
        print(f"未找到 {MICRO_DATA_DIR} 目录")
        return None
    
    files = [f for f in os.listdir(MICRO_DATA_DIR) if f.endswith(('.xlsx', '.xls', '.csv')) and not f.startswith('gaoyang_micro_data_template')]
    if not files:
        print(f"{MICRO_DATA_DIR} 中没有找到数据文件（排除模板）。")
        return None
    
    df_list = []
    for file in files:
        file_path = os.path.join(MICRO_DATA_DIR, file)
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            df_list.append(df)
        except Exception as e:
            print(f"  [错误] 无法读取 {file}: {e}")
    
    if not df_list:
        return None
    
    df = pd.concat(df_list, ignore_index=True)
    
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"  [警告] 缺少字段: {col}，将填充为NaN")
            df[col] = None
    
    df = df[REQUIRED_COLUMNS]
    
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
        df = df.sort_values('Year').reset_index(drop=True)
    
    return df

def import_mechanism_data():
    """主函数：导入并保存微观机制变量"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df = load_and_validate()
    if df is None:
        print("\n没有找到数据文件。是否创建空白模板？")
        create_template()
        return
    
    df['County_Code'] = '130628'
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n处理完成！")
    print(f"共 {len(df)} 年微观机制数据")
    print(f"字段: {df.columns.tolist()}")
    print(f"结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    import_mechanism_data()
