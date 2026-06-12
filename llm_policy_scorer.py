"""
模块1: NLP文本量化 - 政策强度评分流水线 (断点续传版)
功能：使用本地Ollama LLM对政府报告/政策文本进行多维度评分，生成面板数据
特点：
  - 每处理完一份报告立即保存进度
  - 支持中断后从断点继续
  - 输出日志到文件，方便追踪
  - 七维度评分体系聚焦高阳毛巾产业（设备/环保/电商/品牌/集群/金融+教育安慰剂）
输出：output/policy_scores_panel.csv（含County_Code + Year双主键、滞后项、安慰剂校验、复合变量）
依赖：pip install ollama pandas
"""
import os
import re
import json
import time
import pandas as pd
import numpy as np
import ollama

# ================= 配置区 =================
REPORT_DIR = "data/government_reports"
OUTPUT_FILE = "output/policy_scores_panel.csv"
PLACEBO_REPORT = "output/placebo_validation.csv"
PROGRESS_FILE = "output/llm_progress.json"
LOG_FILE = "output/llm_scorer.log"

FILTER_MODEL = "qwen2.5:1.5b"
SCORE_MODEL = "qwen2.5:14b"

MAX_FILES_TO_PROCESS = None  # 设为None处理全部，或设为数字如3表示只处理前3份

# 七维度定义（6个产业维度 + 1个安慰剂维度）
DIMENSIONS = [
    "Equipment",      # 设备升级与技改
    "Environment",    # 环保治理与减排
    "Ecommerce",      # 电商渠道与营销
    "BrandQuality",   # 品牌建设与质量标准
    "Cluster",        # 园区建设与集群发展
    "Finance",        # 金融支持与要素保障
    "Education"       # 教育与人力（安慰剂维度）
]

# ================= 关键词体系设计（三阶段过滤第1层） =================
# 核心词：毛巾产业直接相关
TEXTILE_KEYWORDS_CORE = [
    '毛巾', '浴巾', '方巾', '童巾', '枕巾',  # 毛巾产品
    '纺织', '纺纱', '织造', '印染', '织布',    # 纺织工艺
    '布料', '纱线', '棉纱', '化纤', '坯布',    # 纺织原料
]

# 扩展词：纺织产业链上下游
TEXTILE_KEYWORDS_EXTEND = [
    '织机', '无梭织机', '津田驹', '1515', '1575',  # 纺织设备
    '喷气织机', '剑杆织机', '毛巾织机',            # 设备类型
    '纺织企业', '纺织产业', '纺织园区',            # 产业组织
    '生态印染', '集中印染', '印染废水',            # 环保治理
    '家纺', '家用纺织', '床上用品',                # 下游产品
    '棉花', '棉纺织', '棉纺',                      # 上游原料
    '制毯', '毛毯', '地毯',                        # 相关产品
    '梭织', '针织', '经编',                        # 织造工艺
    '漂染', '印花', '染色',                        # 印染工艺
]

# 区域词：高阳县专属
TEXTILE_KEYWORDS_REGION = [
    '高阳毛巾', '高阳县纺织', '高阳纺织',          # 区域品牌
    '三利', '邦辰', '锦华',                       # 本地龙头企业
    '高阳经济开发区', '高阳纺织商贸城',            # 区域载体
]

# 配套层1：工业园区/产业集群（影响毛巾产业的空间载体）
KEYWORDS_PARK = [
    '开发区', '工业园区', '产业集聚', '产业集群',
    '集中入园', '企业搬迁', '标准厂房', '公共服务平台',
    '创业基地', '孵化园', '众创空间', '科技园',
    '专业园区', '生态园区', '循环经济', '配套完善',
]

# 配套层2：环保治理（对印染/纺织企业有直接影响）
KEYWORDS_ENV = [
    '污水处理', '集中供热', '集中供汽', '散乱污',
    '燃煤锅炉', '达标排放', '环保督察', '关停取缔',
    '废气治理', '废水治理', '固废处理', '节能减排',
    '排污许可证', '环评审批', '环保设施', '清洁生产',
    '水资源税', '用能权', '碳排放', '绿色工厂',
]

# 配套层3：电子商务/物流快递（毛巾销售渠道）
KEYWORDS_ECOMMERCE = [
    '电子商务', '电商', '网店', '直播带货', '跨境电商',
    '快递', '物流', '仓储', '快递发件', '发件量',
    '网上销售', '线上交易', '阿里巴巴', '京东', '淘宝',
    '电商园区', '电商创业', '数字经济', '产业互联网',
]

# 配套层4：金融支持/要素保障（企业融资/技改资金）
KEYWORDS_FINANCE = [
    '融资', '贷款', '信贷', '担保', '政银企',
    '专项资金', '技改资金', '税收优惠', '减税降费',
    '挂牌上市', '股权融资', '融资租赁', '供应链金融',
    '金融生态环境', '不良贷款', '融资担保', '小额贷款',
    '用地保障', '土地供应', '要素保障', '用工保障',
]

# 配套层5：区域品牌/质量标准/展会推介
KEYWORDS_BRAND = [
    '区域品牌', '公共品牌', '地理标志', '集体商标',
    '质量标准', '标准化', '质量监管', '名优产品',
    '著名商标', '知名品牌', '品牌建设', '品牌推广',
    '展会', '推介会', '交易会', '博览会', '广交会',
    '知识产权', '专利', '标准制定', '行业规范',
]

# 配套层6：农业（棉花种植-毛巾原料源头）
KEYWORDS_AGRICULTURE = [
    '棉花', '棉田', '棉花种植', '优质棉基地',
    '棉花产量', '经济作物', '农业结构调整',
    '种植结构', '特色农业', '订单农业',
]

# 配套层7：交通物流（原材料运入/成品运出）
KEYWORDS_TRANSPORT = [
    '高速公路', '国道', '省道', '县乡道路', '交通网络',
    '物流园区', '配送中心', '货运', '交通运输',
    '路网', '交通枢纽', '物流通道', '冷链物流',
    '道路建设', '公路改造', '交通基础设施',
]

# 配套层8：水利/能源（印染高耗水/织机用电）
KEYWORDS_WATER_ENERGY = [
    '工业用水', '供水工程', '水资源', '南水北调',
    '水库', '引水', '水利设施', '供水管网',
    '工业用电', '电网改造', '电力保障', '变电站',
    '能源保障', '天然气管网', '热力管网',
]

# 配套层9：城镇化/用工（劳动密集型产业用工需求）
KEYWORDS_LABOR = [
    '城镇化', '农民工', '劳动力转移', '农村劳动力',
    '就业培训', '技能人才', '用工保障', '招工',
    '人才引进', '校企合作', '订单培养', '职业技术',
    '就业服务', '劳务输出', '返乡创业',
]

# 合并所有关键词
ALL_TEXTILE_KEYWORDS = (
    TEXTILE_KEYWORDS_CORE + 
    TEXTILE_KEYWORDS_EXTEND + 
    TEXTILE_KEYWORDS_REGION +
    KEYWORDS_PARK + 
    KEYWORDS_ENV + 
    KEYWORDS_ECOMMERCE + 
    KEYWORDS_FINANCE + 
    KEYWORDS_BRAND +
    KEYWORDS_AGRICULTURE +
    KEYWORDS_TRANSPORT +
    KEYWORDS_WATER_ENERGY +
    KEYWORDS_LABOR
)

def keyword_filter(chunk):
    """第1层过滤：关键词粗筛，极速（0.01ms/chunk）
    
    匹配逻辑：包含任一关键词即通过
    覆盖范围：毛巾产业核心+配套产业链+宏观产业环境
    """
    return any(kw in chunk for kw in ALL_TEXTILE_KEYWORDS)

def analyze_keyword_layers(chunk):
    """分析chunk匹配的关键词层级（用于统计和调试）"""
    layers = {
        'core': any(kw in chunk for kw in TEXTILE_KEYWORDS_CORE),
        'extend': any(kw in chunk for kw in TEXTILE_KEYWORDS_EXTEND),
        'region': any(kw in chunk for kw in TEXTILE_KEYWORDS_REGION),
        'park': any(kw in chunk for kw in KEYWORDS_PARK),
        'env': any(kw in chunk for kw in KEYWORDS_ENV),
        'ecommerce': any(kw in chunk for kw in KEYWORDS_ECOMMERCE),
        'finance': any(kw in chunk for kw in KEYWORDS_FINANCE),
        'brand': any(kw in chunk for kw in KEYWORDS_BRAND),
        'agriculture': any(kw in chunk for kw in KEYWORDS_AGRICULTURE),
        'transport': any(kw in chunk for kw in KEYWORDS_TRANSPORT),
        'water_energy': any(kw in chunk for kw in KEYWORDS_WATER_ENERGY),
        'labor': any(kw in chunk for kw in KEYWORDS_LABOR),
    }
    return layers

# ================= Prompt 定义 =================
FILTER_PROMPT = """判断以下文本片段是否与以下任一主题相关：
- 毛巾/纺织产业链（纺纱/织造/印染/家纺/棉纱/坯布）
- 纺织设备与工艺（织机/无梭织机/印染设备/淘汰落后产能）
- 棉花种植与农业（棉田/棉花产量/优质棉基地/经济作物）
- 交通物流（高速公路/快递发件/物流园区/货运/交通运输）
- 水利/能源（工业用水/供水工程/电网改造/电力保障/能源保障）
- 工业园区/产业集群（开发区/企业入园/标准厂房/配套完善）
- 环保治理（污水处理/集中供热供汽/散乱污关停/印染园区/节能减排）
- 电子商务/数字经济（网店/直播带货/跨境电商/阿里巴巴/产业互联网）
- 金融支持/要素保障（融资贷款/技改资金/税收优惠/政银企/用地保障）
- 区域品牌/质量标准（地理标志/集体商标/展会推介/著名商标/知识产权）
- 城镇化/用工保障（农民工/劳动力转移/就业培训/校企合作/技能人才）
- 高阳县及保定市产业经济政策

与产业经济完全无关的内容（如纯教育/纯医疗/党建会议/城市绿化/老旧小区改造等）请输出0。
只需输出 1（相关）或 0（不相关），不要输出其他内容：
[{chunk}]"""

SYSTEM_PROMPT = """你是一位客观严谨的区域经济与产业政策分析师，专门研究中国县域传统制造业集群（特别是毛巾纺织产业）的转型升级政策。

请阅读输入的政策文本片段，针对以下七个维度进行 0-5 分的独立量化打分。

【七维度定义与评分标准】

1. Equipment（设备升级与技术改造）
   关注：无梭织机/先进织机引进、淘汰落后设备（1515/1575型）、自动化改造、智能分拣系统、ERP/MES系统升级、技改资金投入、设备更新补贴等。
   - 0分：完全未提及
   - 1分：口号式（如"推进技术改造"）
   - 2分：方向性（如"引导企业更新设备"）
   - 3分：一般举措（如"鼓励引进先进无梭织机，淘汰落后产能"）
   - 4分：具体项目（如"新增先进织机1000台，淘汰1515老旧织机2000台"）
   - 5分：量化+资金（如"投入1.5亿技改资金，新增津田驹织机800台，淘汰落后织机5000台"）

2. Environment（环保治理与减排）
   关注：污水处理厂建设与扩容、集中供热/供汽、散乱污企业关停、生态印染园区、循环经济示范区、污水管网改造、达标排放监管、燃煤锅炉淘汰、水资源税等。
   - 0分：完全未提及
   - 1分：口号式（如"加强环境保护"）
   - 2分：方向性（如"推进污染治理，改善环境质量"）
   - 3分：一般举措（如"加快污水处理厂建设，推进集中供热"）
   - 4分：具体项目（如"投资5.2亿建设污水处理厂三期，集中供热锅炉改造完工"）
   - 5分：量化+目标（如"投资8亿建成生态印染园区，日处理污水10万吨，关停散乱污企业120家"）

3. Ecommerce（电商渠道与营销）
   关注：电子商务发展、直播带货、跨境电商、线上交易平台（如"好农机商城"）、网店培育、电商园区/众创空间、外贸出口企业培育、阿里巴巴/京东合作等。
   - 0分：完全未提及
   - 1分：口号式（如"发展电子商务"）
   - 2分：方向性（如"支持企业开展网上销售"）
   - 3分：一般举措（如"大力发展电商，培育电商创业主体"）
   - 4分：具体项目（如"邦辰众创空间建成，300家企业上网，获评跨境电商25佳县"）
   - 5分：量化+成果（如"电商交易额突破50亿，新增网店2000家，直播带货销售额增长300%"）

4. BrandQuality（品牌建设与质量标准）
   关注：区域公共品牌（如"高阳毛巾"）、地理标志证明商标、集体商标注册、行业/国家标准制定、质量监管整治、国内外展会推介（广交会/家纺展）、著名商标/名优产品培育、知识产权保护等。
   - 0分：完全未提及
   - 1分：口号式（如"加强品牌建设"）
   - 2分：方向性（如"提升产品质量，培育知名品牌"）
   - 3分：一般举措（如"支持企业申报商标，参加展会推介"）
   - 4分：具体项目（如"申报高阳毛巾地理标志，组织30家企业参加进出口商品交易会"）
   - 5分：量化+成果（如"新增著名商标15个，制定行业标准3项，品牌授权企业达100家"）

5. Cluster（园区建设与集群发展）
   关注：经济开发区/工业园区建设、企业集中入园、上下游配套、公共服务平台、创业辅导基地、产业集群培育、园区基础设施（道路/管网）、循环经济示范区、专业市场（纺织商贸城/农机配件城）等。
   - 0分：完全未提及
   - 1分：口号式（如"推进园区建设"）
   - 2分：方向性（如"加快产业集聚，打造产业集群"）
   - 3分：一般举措（如"完善开发区基础设施，引导企业入园"）
   - 4分：具体项目（如"高阳经济开发区入驻企业达155家，循环经济示范区道路建成"）
   - 5分：量化+成果（如"开发区扩容5平方公里，新入驻企业80家，产业集群产值突破200亿"）

6. Finance（金融支持与要素保障）
   关注：银行贷款投放、融资担保体系、技改专项资金、税收优惠/减免、企业挂牌上市（天交所等）、金融生态环境、投融资平台公司、用地保障、政银企对接等。
   - 0分：完全未提及
   - 1分：口号式（如"加强金融支持"）
   - 2分：方向性（如"拓宽企业融资渠道"）
   - 3分：一般举措（如"加大信贷投放，支持实体经济发展"）
   - 4分：具体项目（如"设立5000万技改专项资金，5家企业在天交所挂牌"）
   - 5分：量化+成果（如"发放纺织企业贷款15亿，融资担保覆盖率80%，税收减免3000万"）

7. Education（教育与人力）【安慰剂维度】
   关注：学校建设、教师招聘、义务教育均衡、学前教育、高中教育、职业教育（非产业类）、教育资金投入、教育设施改善等。
   - 0分：完全未提及
   - 1分：口号式（如"发展教育事业"）
   - 2分：方向性（如"推进城乡教育均衡发展"）
   - 3分：一般举措（如"加快学校建设，加强教师队伍建设"）
   - 4分：具体项目（如"实施40所项目校建设，招聘中小学教师500人"）
   - 5分：量化+成果（如"投资2亿新建学校15所，新增学位8000个，教师达标率98%"）

【重要评分原则】
- 七个维度必须完全独立评分，不得因某项高分而给其他维度也打高分（严禁"和稀泥"）。
- 如果文本完全不涉及某个维度，必须给0分，不得给1分作为"安慰"。
- Education维度仅在与教育/人力直接相关时才打分，涉及产业经济的内容与Education完全无关。
- Equipment与Environment虽有交叉（环保设备），但应分别判断：设备升级侧重产能/效率，环保治理侧重污染治理/减排。
- 有具体数据（金额/数量/百分比/项目名称）的才能打4-5分，模糊表述最高3分。

请仅按以下 JSON 格式输出结果，不要输出任何推理过程或其他文本：
{"Equipment": 0, "Environment": 0, "Ecommerce": 0, "BrandQuality": 0, "Cluster": 0, "Finance": 0, "Education": 0}"""

# ================= 日志 =================
def log(msg, level='INFO'):
    prefix = {'DEBUG': '[DEBUG]', 'INFO': '[INFO]', 'WARN': '[WARN]', 'ERROR': '[ERROR]'}.get(level, '[INFO]')
    ts = time.strftime('%H:%M:%S')
    line = f"{ts} {prefix} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ================= 进度管理 =================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed_files': [], 'annual_data': {}}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def save_results(annual_data):
    """保存最终结果"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    final_rows = []
    for key, data in sorted(annual_data.items()):
        county_code, year = key
        total = data['total_chunks'] if data['total_chunks'] > 0 else 1
        
        # 数据可靠性标记：total_chunks < 10 视为不可靠样本（用1/0而非True/False）
        sample_reliable = 1 if data['total_chunks'] >= 10 else 0
        
        row = {
            'County_Code': county_code,
            'Year': year,
            'total_chunks': data['total_chunks'],
            'sample_reliable': sample_reliable,
            # 新增：过滤漏斗统计
            'keyword_passed': data.get('keyword_passed', 0),
            'filter_passed': data.get('filter_passed', 0),
            'scored': data.get('scored', 0),
        }
        
        for dim in DIMENSIONS:
            dim_lower = dim.lower()
            row[f'{dim_lower}_index'] = round((data.get(dim, 0) / total) * 100, 2)
        
        final_rows.append(row)
    
    df = pd.DataFrame(final_rows)
    df = df.sort_values(['County_Code', 'Year']).reset_index(drop=True)
    
    # 生成滞后项
    for dim in DIMENSIONS:
        dim_lower = dim.lower()
        col = f'{dim_lower}_index'
        df[f'L1_{col}'] = df.groupby('County_Code')[col].shift(1)
        df[f'L2_{col}'] = df.groupby('County_Code')[col].shift(2)
    
    # 计算复合变量（只使用已存在的列）
    production_dims = ['equipment', 'environment', 'cluster']
    market_dims = ['ecommerce', 'brandquality', 'finance']
    all_industry_dims = production_dims + market_dims
    
    # 检查需要哪些列
    needed_l1_cols = [f'L1_{d}_index' for d in all_industry_dims]
    existing_l1_cols = [c for c in needed_l1_cols if c in df.columns]
    
    if len(existing_l1_cols) > 0:
        df['policy_intensity_total'] = sum(df[c].fillna(0) for c in existing_l1_cols)
        df['policy_intensity_production'] = sum(df[f'L1_{d}_index'].fillna(0) for d in production_dims if f'L1_{d}_index' in df.columns)
        df['policy_intensity_market'] = sum(df[f'L1_{d}_index'].fillna(0) for d in market_dims if f'L1_{d}_index' in df.columns)
    else:
        log("警告: 没有L1滞后项可用于计算复合变量", 'WARN')
        df['policy_intensity_total'] = np.nan
        df['policy_intensity_production'] = np.nan
        df['policy_intensity_market'] = np.nan
    
    # 修复1：交互项替代比值（避免除零爆炸）
    # policy_mix_equipment_env: 设备升级与环保治理的协同效应
    # 使用乘法：两个维度都高时协同效应才显著
    if 'L1_equipment_index' in df.columns and 'L1_environment_index' in df.columns:
        df['policy_mix_equipment_env'] = (
            df['L1_equipment_index'].fillna(0) * df['L1_environment_index'].fillna(0)
        ) / 100  # 归一化到合理范围
    
    # policy_mix_production_market: 供给侧与需求侧政策的结构性平衡
    # 使用比值但添加平滑处理（加1避免除零，取log压缩极端值）
    if 'policy_intensity_production' in df.columns and 'policy_intensity_market' in df.columns:
        prod = df['policy_intensity_production'].fillna(0) + 1
        mkt = df['policy_intensity_market'].fillna(0) + 1
        df['policy_mix_production_market'] = np.log(prod / mkt)  # log比值，对称分布
    
    log(f"复合变量生成完成: policy_intensity_total均值={df['policy_intensity_total'].mean():.2f}")
    log(f"复合变量生成完成: policy_mix_equipment_env均值={df['policy_mix_equipment_env'].mean():.4f}")
    log(f"复合变量生成完成: policy_mix_production_market均值={df['policy_mix_production_market'].mean():.4f}")
    
    # 统计不可靠样本数量
    unreliable_count = (~df['sample_reliable']).sum()
    if unreliable_count > 0:
        log(f"警告: {unreliable_count} 个样本 total_chunks<10，标记为不可靠", 'WARN')
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    log(f"结果已保存至 {OUTPUT_FILE} ({len(df)} 行)")
    
    return df

# ================= 文本处理 =================
def parse_filename(filename):
    basename = filename.replace('.txt', '')
    parts = basename.split('_')
    if len(parts) >= 2:
        county_name = parts[0]
        county_code_map = {'高阳县': '130628', '保定市': '130600', '河北省': '130000'}
        county_code = county_code_map.get(county_name, '999999')
        try:
            year = int(parts[1])
        except ValueError:
            return None, None
        return county_code, year
    return None, None

def chunk_text(text, sentences_per_chunk=4, max_chars_per_chunk=500):
    """
    多重策略文本分块：
    1. 清理元数据行（标题、发文单位、来源等，通常在前6行）
    2. 优先按段落拆分（双换行符）
    3. 段落内部按句子拆分（句号/叹号/分号）
    4. 合并为sentences_per_chunk一组，但不超过max_chars_per_chunk
    5. 兜底：如果分块<3，按固定长度强行拆分
    
    改进：更精确的元数据识别，避免误删正文
    """
    lines = text.split('\n')
    
    # 改进的元数据清理逻辑
    body_start = 0
    metadata_keywords = ['来源', '作者', '发布时间', '更新日期', '点击', '浏览', '下载', 
                         '字号', '视力保护色', '分享到', '打印', '关闭']
    
    for i, line in enumerate(lines[:15]):  # 检查前15行
        stripped = line.strip()
        
        # 跳过明显的元数据行
        if any(kw in stripped for kw in metadata_keywords):
            body_start = i + 1
            continue
        
        # 如果找到正文标志（人民代表大会/人民政府/各位代表），认为是正文开始
        if any(kw in stripped for kw in ['人民代表大会', '人民政府', '各位代表']):
            body_start = i
            break
        
        # 跳过短行（<30字符）但仅在前6行
        if len(stripped) < 30 and i < 6:
            body_start = i + 1
        else:
            # 遇到长行，认为是正文开始
            break
    
    clean_text = '\n'.join(lines[body_start:])
    
    # 按段落拆分（双换行/空行分隔）
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', clean_text) if len(p.strip()) > 20]
    
    # 段落内部按句子拆分
    all_sentences = []
    for para in paragraphs:
        # 清理不可打印字符（乱码）
        para = re.sub(r'[^\x20-\x7E\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]', '', para)
        sentences = re.split(r'[。！；!?]', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        all_sentences.extend(sentences)
    
    # 合并为chunks
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in all_sentences:
        sent_with_punct = sent + '。'
        if current_len + len(sent_with_punct) > max_chars_per_chunk and len(current_chunk) > 0:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent_with_punct)
        current_len += len(sent_with_punct)
        if len(current_chunk) >= sentences_per_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # 兜底策略——如果chunks太少，按固定长度强行拆分
    if len(chunks) < 3:
        fallback_chunks = []
        for i in range(0, len(all_sentences), sentences_per_chunk):
            batch = all_sentences[i:i+sentences_per_chunk]
            if batch:
                fallback_chunks.append('。'.join(batch) + '。')
        if fallback_chunks:
            chunks = fallback_chunks
    
    return chunks

# ================= LLM 调用 =================
def ollama_ping(timeout=5):
    """检测Ollama服务是否可用"""
    try:
        ollama.list()
        return True
    except:
        return False

def run_filter(chunk, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=FILTER_MODEL,
                messages=[{'role': 'user', 'content': FILTER_PROMPT.format(chunk=chunk)}],
                options={"num_predict": 3, "temperature": 0.0}
            )
            result = response['message']['content'].strip()
            return 1 if '1' in result else 0
        except Exception as e:
            wait = 2 ** attempt  # 指数退避: 2s, 4s, 8s
            log(f"  Filter重试 {attempt+1}/{max_retries} (等待{wait}s): {e}", 'WARN')
            time.sleep(wait)
            
            # 第2次重试失败后检测服务
            if attempt == 1 and not ollama_ping():
                log("  Ollama服务不可用，等待10秒...", 'ERROR')
                time.sleep(10)
    return 0

def run_score(chunk, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=SCORE_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f"请对以下政策文本片段进行打分：\n\n{chunk}"}
                ],
                format='json',
                options={"num_predict": 50, "temperature": 0.1}
            )
            result = response['message']['content'].strip()
            scores = json.loads(result)
            
            # 验证返回的JSON是否包含所有维度
            valid_score = True
            for dim in DIMENSIONS:
                if dim not in scores:
                    log(f"  Score缺失维度: {dim}", 'WARN')
                    valid_score = False
                    break
            
            if not valid_score:
                raise ValueError(f"Score返回缺少维度: {[d for d in DIMENSIONS if d not in scores]}")
            
            return {
                dim: int(scores.get(dim, 0)) for dim in DIMENSIONS
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log(f"  Score格式错误重试 {attempt+1}/{max_retries}: {e}", 'WARN')
            time.sleep(1)
        except Exception as e:
            wait = 3 * (2 ** attempt)  # 指数退避: 3s, 6s, 12s
            log(f"  Score网络错误重试 {attempt+1}/{max_retries} (等待{wait}s): {e}", 'WARN')
            time.sleep(wait)
            
            # 第2次重试失败后检测服务
            if attempt == 1 and not ollama_ping():
                log("  Ollama服务不可用，等待15秒...", 'ERROR')
                time.sleep(15)
    return {dim: 0 for dim in DIMENSIONS}

# ================= 主流程 =================
def read_file_auto_encoding(fpath):
    """自动检测文件编码并读取"""
    # 先尝试UTF-8
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        garbage_ratio = text.count('\ufffd') / max(len(text), 1)
        if garbage_ratio < 0.01:  # 乱码比例<1%
            return text, 'utf-8'
    except:
        pass
    
    # UTF-8失败或乱码太多，尝试GBK
    try:
        with open(fpath, 'r', encoding='gbk') as f:
            text = f.read()
        return text, 'gbk'
    except:
        pass
    
    # 兜底
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(), 'utf-8-ignored'

def process_single_file(fpath, county_code, year):
    """处理单个报告文件（三阶段过滤：关键词→LLM Filter→14B打分）"""
    text, enc = read_file_auto_encoding(fpath)
    
    chunks = chunk_text(text)
    log(f"  文本 {len(text)} 字符（编码:{enc}），分块 {len(chunks)} 个")
    
    data = {"total_chunks": len(chunks)}
    for dim in DIMENSIONS:
        data[dim] = 0
    
    # 三阶段过滤统计
    keyword_passed = 0  # 第1层通过数
    filter_passed = 0   # 第2层通过数
    scored = 0          # 第3层打分数
    
    for i, chunk in enumerate(chunks):
        # 第1层：关键词粗筛（0.01ms/chunk）
        if not keyword_filter(chunk):
            continue
        keyword_passed += 1
        
        # 第2层：LLM轻量Filter（1.5B模型，~0.5s/chunk）
        if run_filter(chunk) != 1:
            continue
        filter_passed += 1
        
        # 第3层：LLM精细打分（14B模型，~2-3s/chunk）
        scores = run_score(chunk)
        scored += 1
        for key in DIMENSIONS:
            data[key] += scores[key]
        
        if (i + 1) % 50 == 0:
            log(f"  进度: {i+1}/{len(chunks)} chunks, keyword={keyword_passed}, filter={filter_passed}, scored={scored}")
    
    # 保存关键统计信息（用于数据质量分析）
    data['keyword_passed'] = keyword_passed
    data['filter_passed'] = filter_passed
    data['scored'] = scored
    
    log(f"  完成: keyword={keyword_passed}/{len(chunks)}, filter={filter_passed}, scored={scored}")
    score_summary = ", ".join([f"{dim}={data[dim]}" for dim in DIMENSIONS])
    log(f"  得分: {score_summary}")
    
    return data

def process_corpus():
    """主函数：处理语料库并生成面板数据"""
    os.makedirs("output", exist_ok=True)
    
    log("=" * 60)
    log("模块1: LLM政策评分 - 启动")
    log("=" * 60)
    
    if not os.path.exists(REPORT_DIR):
        log(f"错误：{REPORT_DIR} 目录不存在", 'ERROR')
        return
    
    files = sorted([f for f in os.listdir(REPORT_DIR) if f.endswith('.txt')])
    log(f"找到 {len(files)} 份报告文件")
    
    if MAX_FILES_TO_PROCESS:
        files = files[:MAX_FILES_TO_PROCESS]
        log(f"限制处理前 {MAX_FILES_TO_PROCESS} 份")
    
    # 加载进度
    progress = load_progress()
    completed = progress['completed_files']
    annual_data = progress['annual_data']
    log(f"已完成的文件: {len(completed)} 份")
    
    remaining = [f for f in files if f not in completed]
    log(f"剩余待处理: {len(remaining)} 份")
    
    if not remaining:
        log("所有文件已处理完毕，跳过")
    else:
        total = len(remaining)
        for idx, fname in enumerate(remaining):
            fpath = os.path.join(REPORT_DIR, fname)
            county_code, year = parse_filename(fname)
            if county_code is None or year is None:
                log(f"跳过无法解析的文件: {fname}", 'WARN')
                # ❌ 不标记为完成，保留下次重试机会
                continue
            
            key = f"{county_code}_{year}"
            log(f"\n[{idx+1}/{total}] 处理: {fname} -> {key}")
            
            t_start = time.time()
            try:
                data = process_single_file(fpath, county_code, year)
                annual_data[key] = data
                elapsed = time.time() - t_start
                log(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
                
                progress['completed_files'].append(fname)
                progress['annual_data'] = annual_data
                save_progress(progress)
                log(f"  进度已保存")
            except Exception as e:
                log(f"  处理失败: {e}", 'ERROR')
                continue
    
    # 生成最终结果
    log("\n" + "=" * 60)
    log("生成最终面板数据...")
    log("=" * 60)
    
    formatted_data = {}
    for key, data in annual_data.items():
        parts = key.split('_')
        county_code = parts[0]
        year = int(parts[1])
        formatted_data[(county_code, year)] = data
    
    df = save_results(formatted_data)
    
    log(f"\n最终统计:")
    log(f"  处理县数: {df['County_Code'].nunique()}")
    log(f"  年份范围: {df['Year'].min()} - {df['Year'].max()}")
    log(f"  记录数: {len(df)}")
    
    log("\n模块1: LLM政策评分 - 完成!")

if __name__ == "__main__":
    process_corpus()
