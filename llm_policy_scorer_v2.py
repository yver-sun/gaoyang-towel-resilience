"""
模块1 v2: LLM政策文本量化 — 修复版
修复清单:
  Bug#1: policy_intensity_total 改用当前年维度得分(非L1滞后)
  Bug#2: 新增 _raw(绝对力度) 变量, 不除以total_chunks
  Bug#3: Prompt增加纺织/毛巾实体锚定约束
  Bug#4: 分块改为自然段落边界(max 800 chars)
  Bug#5: 移除1.5B小模型初筛(14B直接处理全部关键词命中chunk)
  Bug#6: 关键词精简为纺织专属, 移除通用基础设施词
  Bug#7: 移除"好农机商城"等误导性示例
  Bug#8: 标注Education为财政充裕度代理变量(非安慰剂)
  Bug#9: System Prompt增加5/3/1分锚定示例

输出: output/policy_scores_panel_v2.csv
"""
import os
import re
import json
import time
import pandas as pd
import numpy as np
import ollama

REPORT_DIR = "data/government_reports"
OUTPUT_FILE = "output/policy_scores_panel_v2.csv"
PROGRESS_FILE = "output/llm_progress_v2.json"
LOG_FILE = "output/llm_scorer_v2.log"

SCORE_MODEL = "qwen2.5:14b"
MAX_FILES = None

DIMENSIONS = [
    "Equipment",
    "Environment",
    "Ecommerce",
    "BrandQuality",
    "Cluster",
    "Finance",
    "Education"
]

# === Bug#6 修复: 精简为纺织专属关键词 ===
TEXTILE_KEYWORDS = [
    # 核心产品
    '毛巾', '浴巾', '方巾', '童巾', '枕巾', '巾被',
    # 核心产业
    '纺织', '纺纱', '织造', '印染', '织布', '漂染', '印花', '染色',
    # 原料
    '棉纱', '坯布', '化纤', '棉花', '棉纺织', '棉纺',
    # 设备
    '织机', '无梭织机', '喷气织机', '剑杆织机', '毛巾织机',
    '津田驹', '1515', '1575', '梭织', '针织', '经编',
    # 产品
    '家纺', '家用纺织', '床上用品', '布料', '纱线', '毛毯', '制毯',
    # 产业组织
    '纺织企业', '纺织产业', '纺织园区', '纺织商贸城',
    # 环保(纺织专属)
    '生态印染', '集中印染', '印染废水', '印染园区',
    # 品牌/区域
    '高阳毛巾', '高阳纺织', '中国纺织之乡', '中国毛巾之乡',
    '三利', '邦辰', '锦华',
    # 纺织专用载体
    '高阳经济开发区', '高阳纺织商贸城',
    # 纺织语境下的环保/能源（在Gaoyang几乎总与印染纺织相关）
    '循环经济', '散乱污', '集中供热', '集中供汽',
    # 技改（在Gaoyang主导产业为纺织的语境下）
    '技改资金', '技改专项', '技术改造',
]

def keyword_filter(chunk):
    return any(kw in chunk for kw in TEXTILE_KEYWORDS)

# === Bug#3+7+9 修复: System Prompt ===
SYSTEM_PROMPT = """你是一位客观严谨的产业政策分析师，专门研究中国县域传统制造业集群的转型升级政策。

【最重要约束 — Bug#3修复】你的评分【仅针对纺织/毛巾/印染产业链】。在打分前，你必须首先判断文本主体是否直接涉及纺织产业：

规则A：以下情况必须打0分 —
- 文本涉及农业（养猪/种地/棉花种植除外）、农机、建筑、旅游、餐饮
- 文本仅涉及通用教育（学校建设/教师招聘）、通用医疗（医院/卫生院）
- 文本仅涉及通用基础设施（道路/供水/供电）且未提及纺织产业
- 文本中提到的"企业/产业"明确指向非纺织行业

规则B：可以合理打分的场景 —
- 文本明确提及纺织/毛巾/印染/织造/家纺
- 文本提及"工业企业""产业园区""产业转型升级"且高阳县以纺织为主导产业
- 文本提及印染废水/集中供热供汽/散乱污关停且位于纺织产业聚集区
- 文本提及棉花种植、棉田（纺织产业链上游原料）

规则C：评分锚定示例（Bug#9修复 — 跨年标尺校准）—
设备升级维度：
- 5分："投资1.5亿元技改资金，引进津田驹喷气织机800台，淘汰1515型老旧织机5000台"
- 3分："鼓励企业引进先进无梭织机，加快设备更新换代"
- 1分："推进技术改造"
- 0分：未提及设备相关内容，或提及的是非纺织设备

环保治理维度：
- 5分："投资8亿元建成生态印染园区，日处理印染污水10万吨，关停散乱污企业120家"
- 3分："加快污水处理厂建设，推进集中供热，加强散乱污企业整治"
- 1分："加强环境保护"
- 0分：未提及环保相关内容，或提及的是非工业环保

【七维度定义与评分标准】

1. Equipment（设备升级与技术改造）
   关注：无梭织机/先进织机引进、淘汰落后设备（1515/1575型）、自动化改造、智能分拣系统、ERP/MES系统升级、技改资金投入、设备更新补贴等（仅限纺织）。
   - 0分：完全未提及或非纺织设备
   - 1分：口号式（如"推进技术改造"）
   - 2分：方向性（如"引导企业更新设备"）
   - 3分：一般举措（如"鼓励引进先进无梭织机，淘汰落后产能"）
   - 4分：具体项目（如"新增先进织机1000台，淘汰1515老旧织机2000台"）
   - 5分：量化+资金（如"投入1.5亿技改资金，新增津田驹织机800台，淘汰落后织机5000台"）

2. Environment（环保治理与减排）
   关注：印染污水处理厂建设与扩容、集中供热/供汽、散乱污企业关停（纺织相关）、生态印染园区、循环经济示范区（纺织）、污水管网改造、达标排放监管、燃煤锅炉淘汰等。
   - 0分：完全未提及或非工业环保
   - 1分：口号式（如"加强环境保护"）
   - 2分：方向性（如"推进污染治理，改善环境质量"）
   - 3分：一般举措（如"加快污水处理厂建设，推进集中供热"）
   - 4分：具体项目（如"投资5.2亿建设污水处理厂三期，集中供热锅炉改造完工"）
   - 5分：量化+目标（如"投资8亿建成生态印染园区，日处理污水10万吨，关停散乱污企业120家"）

3. Ecommerce（电商渠道与营销）
   关注：纺织品电子商务发展、直播带货、跨境电商、纺织品线上交易平台、网店培育、电商园区/众创空间（纺织相关）、外贸出口企业培育（纺织）等。
   Bug#7修复：不再提示"好农机商城"（与纺织无关）。
   - 0分：完全未提及或非纺织电商
   - 1分：口号式（如"发展电子商务"）
   - 2分：方向性（如"支持企业开展网上销售"）
   - 3分：一般举措（如"大力发展电商，培育电商创业主体"）
   - 4分：具体项目（如"邦辰众创空间建成，300家纺织企业上网，获评跨境电商25佳县"）
   - 5分：量化+成果（如"纺织品电商交易额突破50亿，新增网店2000家"）

4. BrandQuality（品牌建设与质量标准）
   关注：区域公共品牌（如"高阳毛巾"）、地理标志证明商标、集体商标注册、纺织行业/国家标准制定、纺织品质量监管整治、纺织展会推介（广交会/家纺展）、著名商标/名优产品培育等。
   - 0分：完全未提及或非纺织品牌
   - 1分：口号式（如"加强品牌建设"）
   - 2分：方向性（如"提升产品质量，培育知名品牌"）
   - 3分：一般举措（如"支持企业申报商标，参加展会推介"）
   - 4分：具体项目（如"申报高阳毛巾地理标志，组织30家企业参加进出口商品交易会"）
   - 5分：量化+成果（如"新增著名商标15个，制定行业标准3项，品牌授权企业达100家"）

5. Cluster（园区建设与集群发展）
   关注：纺织经济开发区/纺织工业园区建设、纺织企业集中入园、纺织上下游配套、公共服务平台（纺织检测/设计）、创业辅导基地（纺织）、纺织产业集群培育、纺织专业市场（纺织商贸城）等。
   - 0分：完全未提及或非纺织园区
   - 1分：口号式（如"推进园区建设"）
   - 2分：方向性（如"加快产业集聚，打造产业集群"）
   - 3分：一般举措（如"完善开发区基础设施，引导企业入园"）
   - 4分：具体项目（如"高阳经济开发区入驻纺织企业达155家"）
   - 5分：量化+成果（如"开发区扩容5平方公里，新入驻纺织企业80家，集群产值突破200亿"）

6. Finance（金融支持与要素保障）
   关注：纺织企业银行贷款投放、融资担保体系、技改专项资金（纺织）、税收优惠/减免（纺织）、企业挂牌上市（纺织）、政银企对接（纺织）、纺织企业用地保障等。
   - 0分：完全未提及或非纺织金融
   - 1分：口号式（如"加强金融支持"）
   - 2分：方向性（如"拓宽企业融资渠道"）
   - 3分：一般举措（如"加大信贷投放，支持实体经济发展"）
   - 4分：具体项目（如"设立5000万技改专项资金，5家企业在天交所挂牌"）
   - 5分：量化+成果（如"发放纺织企业贷款15亿，融资担保覆盖率80%，税收减免3000万"）

7. Education（教育与人力）【注意：非安慰剂 — Bug#8修复】
   此维度测量的是文本中教育/医疗/民生投入的叙事强度。由于在中国县域治理中，产业补贴与教育医疗投入都依赖于地方财政充裕度，此维度实际反映的是"地方政府整体行政活跃度与财力"，而非纯安慰剂。
   - 0分：完全未提及
   - 1分：口号式（如"发展教育事业"）
   - 2分：方向性（如"推进城乡教育均衡发展"）
   - 3分：一般举措（如"加快学校建设，加强教师队伍建设"）
   - 4分：具体项目（如"实施40所项目校建设，招聘中小学教师500人"）
   - 5分：量化+成果（如"投资2亿新建学校15所，新增学位8000个，教师达标率98%"）

【评分原则】
- 七个维度独立评分，不得因某项高分而给其他维度也打高分
- 文本完全不涉及某维度 → 必须给0分
- 有具体数据（金额/数量/百分比/项目名称）才能打4-5分，模糊表述最高3分
- 仅按以下JSON格式输出结果，不要输出任何推理过程：
{"Equipment": 0, "Environment": 0, "Ecommerce": 0, "BrandQuality": 0, "Cluster": 0, "Finance": 0, "Education": 0}"""

def log(msg, level='INFO'):
    prefix = {'DEBUG': '[D]', 'INFO': '[I]', 'WARN': '[W]', 'ERROR': '[E]'}.get(level, '[I]')
    ts = time.strftime('%H:%M:%S')
    line = f"{ts} {prefix} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed_files': [], 'annual_data': {}}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def parse_filename(filename):
    basename = filename.replace('.txt', '')
    parts = basename.split('_')
    if len(parts) >= 2:
        county_code_map = {'高阳县': '130628', '保定市': '130600', '河北省': '130000'}
        county_code = county_code_map.get(parts[0], '999999')
        try:
            year = int(parts[1])
        except ValueError:
            return None, None
        return county_code, year
    return None, None

def chunk_text_v2(text, max_chars=800):
    """
    Bug#4修复: 按自然段落边界分块，保持语义完整性
    - 优先按双换行符(段落)拆分
    - 超长段落按句子拆分
    - max_chars提高到800(原500)，减少截断
    """
    lines = text.split('\n')

    # 元数据清理
    body_start = 0
    metadata_keywords = ['来源', '作者', '发布时间', '更新日期', '点击', '浏览',
                         '下载', '字号', '视力保护色', '分享到', '打印', '关闭']
    for i, line in enumerate(lines[:15]):
        stripped = line.strip()
        if any(kw in stripped for kw in metadata_keywords):
            body_start = i + 1
            continue
        if any(kw in stripped for kw in ['人民代表大会', '人民政府', '各位代表']):
            body_start = i
            break
        if len(stripped) < 30 and i < 6:
            body_start = i + 1
        else:
            break

    clean_text = '\n'.join(lines[body_start:])

    # 按段落拆分
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', clean_text) if len(p.strip()) > 10]

    chunks = []
    for para in paragraphs:
        para = re.sub(r'[^\x20-\x7E一-鿿　-〿＀-￯\s]', '', para)

        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 超长段落：按句子拆分后组合
            sentences = re.split(r'(?<=[。！；!?])', para)
            current = []
            current_len = 0
            for sent in sentences:
                s = sent.strip()
                if len(s) < 5:
                    continue
                if current_len + len(s) > max_chars and current:
                    chunks.append(''.join(current))
                    current = []
                    current_len = 0
                current.append(s)
                current_len += len(s)
            if current:
                chunks.append(''.join(current))

    return chunks

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
                options={"num_predict": 80, "temperature": 0.1}
            )
            result = response['message']['content'].strip()
            scores = json.loads(result)
            for dim in DIMENSIONS:
                if dim not in scores:
                    raise ValueError(f"Missing dimension: {dim}")
            return {dim: int(scores.get(dim, 0)) for dim in DIMENSIONS}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log(f"  Score format retry {attempt+1}/{max_retries}: {e}", 'WARN')
            time.sleep(2)
        except Exception as e:
            wait = 3 * (2 ** attempt)
            log(f"  Score network retry {attempt+1}/{max_retries} (wait {wait}s): {e}", 'WARN')
            time.sleep(wait)
    return {dim: 0 for dim in DIMENSIONS}

def read_file_auto_encoding(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        if text.count('�') / max(len(text), 1) < 0.01:
            return text, 'utf-8'
    except:
        pass
    try:
        with open(fpath, 'r', encoding='gbk') as f:
            return f.read(), 'gbk'
    except:
        pass
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(), 'utf-8-ignored'

def process_single_file(fpath, county_code, year):
    """Bug#5修复: 两阶段过滤（关键词→14B直接打分，移除1.5B初筛）"""
    text, enc = read_file_auto_encoding(fpath)
    chunks = chunk_text_v2(text)
    log(f"  {len(text)} chars (enc:{enc}), {len(chunks)} chunks")

    data = {"total_chunks": len(chunks)}
    for dim in DIMENSIONS:
        data[dim] = 0

    keyword_passed = 0
    scored = 0

    for i, chunk in enumerate(chunks):
        # Stage 1: 纺织关键词粗筛
        if not keyword_filter(chunk):
            continue
        keyword_passed += 1

        # Stage 2: 14B直接打分 (移除1.5B初筛——Bug#5修复)
        scores = run_score(chunk)
        scored += 1
        for key in DIMENSIONS:
            data[key] += scores[key]

        if (scored + 1) % 30 == 0:
            log(f"  progress: {i+1}/{len(chunks)} chunks, kw={keyword_passed}, scored={scored}")

    data['keyword_passed'] = keyword_passed
    data['scored'] = scored
    log(f"  done: kw={keyword_passed}/{len(chunks)}, scored={scored}")
    log(f"  scores: " + ", ".join([f"{d}={data[d]}" for d in DIMENSIONS]))
    return data

def save_results_v2(annual_data):
    """Bug#1修复: policy_intensity_total使用当前年维度得分(非L1)"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    final_rows = []
    for key, data in sorted(annual_data.items()):
        county_code, year = key
        total = max(data['total_chunks'], 1)

        # 报告可靠性标记
        sample_reliable = 1 if data['total_chunks'] >= 10 else 0

        row = {
            'County_Code': county_code,
            'Year': year,
            'total_chunks': data['total_chunks'],
            'sample_reliable': sample_reliable,
            'keyword_passed': data.get('keyword_passed', 0),
            'scored': data.get('scored', 0),
        }

        # === Bug#2修复: 同时生成两种度量 ===
        # _index: 浓度度量（sum/total_chunks*100）—— 叙事密度
        # _raw: 绝对度量（sum）—— 政策叙事总量
        for dim in DIMENSIONS:
            dim_lower = dim.lower()
            raw_sum = data.get(dim, 0)
            row[f'{dim_lower}_sum'] = raw_sum  # 绝对力度
            row[f'{dim_lower}_index'] = round((raw_sum / total) * 100, 2)  # 浓度

        final_rows.append(row)

    df = pd.DataFrame(final_rows)
    df = df.sort_values(['County_Code', 'Year']).reset_index(drop=True)

    # 生成滞后项(L1, L2)
    for dim in DIMENSIONS:
        dim_lower = dim.lower()
        for suffix in ['_sum', '_index']:
            col = f'{dim_lower}{suffix}'
            df[f'L1_{col}'] = df.groupby('County_Code')[col].shift(1)
            df[f'L2_{col}'] = df.groupby('County_Code')[col].shift(2)

    # === Bug#1修复: policy_intensity_total = 当前年6维度_sum的总和 ===
    industry_dims = ['equipment', 'environment', 'ecommerce', 'brandquality', 'cluster', 'finance']

    # 绝对力度（推荐使用）
    df['policy_intensity_total'] = sum(
        df[f'{d}_sum'].fillna(0) for d in industry_dims
    )
    # 浓度度量（用于对比）
    df['policy_intensity_concentration'] = sum(
        df[f'{d}_index'].fillna(0) for d in industry_dims
    )

    # 生产端 vs 市场端
    prod_dims = ['equipment', 'environment', 'cluster']
    mkt_dims = ['ecommerce', 'brandquality', 'finance']

    df['policy_intensity_production'] = sum(df[f'{d}_sum'].fillna(0) for d in prod_dims)
    df['policy_intensity_market'] = sum(df[f'{d}_sum'].fillna(0) for d in mkt_dims)

    # 交互项 (改用 _index 避免量纲问题)
    df['policy_mix_equipment_env'] = (
        df['equipment_index'].fillna(0) * df['environment_index'].fillna(0)
    ) / 100
    prod_raw = df['policy_intensity_production'].fillna(0) + 1
    mkt_raw = df['policy_intensity_market'].fillna(0) + 1
    df['policy_mix_production_market'] = np.log(prod_raw / mkt_raw)

    log(f"Composite: policy_intensity_total mean={df['policy_intensity_total'].mean():.1f}")
    log(f"Composite: policy_intensity_concentration mean={df['policy_intensity_concentration'].mean():.1f}")
    unreliable = (~df['sample_reliable'].astype(bool)).sum()
    if unreliable > 0:
        log(f"WARNING: {unreliable} samples total_chunks<10 (unreliable)")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    log(f"Saved to {OUTPUT_FILE} ({len(df)} rows)")
    return df

def process_corpus():
    os.makedirs("output", exist_ok=True)
    log("=" * 60)
    log("LLM Policy Scorer v2 — All Bugs Fixed")
    log("=" * 60)

    if not os.path.exists(REPORT_DIR):
        log(f"ERROR: {REPORT_DIR} not found", 'ERROR')
        return

    files = sorted([f for f in os.listdir(REPORT_DIR) if f.endswith('.txt')])
    log(f"Found {len(files)} report files")

    if MAX_FILES:
        files = files[:MAX_FILES]
        log(f"Limited to first {MAX_FILES}")

    progress = load_progress()
    completed = progress['completed_files']
    annual_data = progress['annual_data']
    log(f"Already completed: {len(completed)} files")

    remaining = [f for f in files if f not in completed]
    log(f"Remaining: {len(remaining)} files")

    if not remaining:
        log("All files processed, skipping")
    else:
        for idx, fname in enumerate(remaining):
            fpath = os.path.join(REPORT_DIR, fname)
            county_code, year = parse_filename(fname)
            if county_code is None:
                log(f"Skip unparseable: {fname}", 'WARN')
                continue

            key = f"{county_code}_{year}"
            log(f"\n[{idx+1}/{len(remaining)}] {fname} -> {key}")

            t0 = time.time()
            try:
                data = process_single_file(fpath, county_code, year)
                annual_data[key] = data
                elapsed = time.time() - t0
                log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

                progress['completed_files'].append(fname)
                progress['annual_data'] = annual_data
                save_progress(progress)
                log(f"  Progress saved")
            except Exception as e:
                log(f"  FAILED: {e}", 'ERROR')
                continue

    log("\n" + "=" * 60)
    log("Generating final panel...")
    formatted_data = {}
    for key, data in annual_data.items():
        parts = key.split('_')
        formatted_data[(parts[0], int(parts[1]))] = data

    df = save_results_v2(formatted_data)
    log(f"\nFinal: {df['County_Code'].nunique()} counties, "
        f"{df['Year'].min()}-{df['Year'].max()}, {len(df)} rows")
    log("DONE!")

if __name__ == "__main__":
    process_corpus()
