import os
import re
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置路径
DATA_DIR = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports"
OUTPUT_DIR = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾\output\bertopic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 停用词设置（增强版）
STOPWORDS = set([
    '的', '了', '和', '与', '在', '是', '我', '有', '也', '就', '不', '人', '都', '一', '一个', '上',
    '我们', '要', '到', '以', '这', '为', '中', '大', '会', '可', '等', '其', '他', '她', '它',
    '发展', '建设', '推进', '工作', '加强', '促进', '提高', '实现', '全面', '深化', '坚持',
    '高阳县', '高阳', '保定市', '河北省', '全县', '全市', '全省', '政府', '国家', '社会', '经济',
    '重点', '重大', '重要', '基本', '不断', '持续', '进一步', '新', '好', '去', '过', '能', '可以',
    '年', '月', '日', '万', '亿', '百分之', '同比增长', '增长'
])

def preprocess_text(text):
    """文本清洗与分词"""
    # 移除HTML标签和特殊字符
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t\u3000]', '', text)
    text = re.sub(r'[^\w\s]', '', text) # 移除标点符号
    
    # jieba分词
    words = jieba.lcut(text)
    
    # 过滤停用词和单字
    words = [w for w in words if w not in STOPWORDS and len(w) > 1 and not w.isnumeric()]
    
    return " ".join(words)

def load_and_preprocess_data():
    """加载高阳县政府工作报告并进行预处理"""
    documents = []
    timestamps = []
    
    print("Loading and preprocessing documents...")
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('高阳县') and f.endswith('.txt')]
    
    for filename in tqdm(files):
        # 提取年份
        match = re.search(r'高阳县_(\d{4})_report\.txt', filename)
        if not match:
            continue
        year = int(match.group(1))
        
        # 读取文件
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='gbk') as f:
                text = f.read()
                
        # 由于政府报告通常很长，按段落拆分以增加文档数量，提高主题模型效果
        paragraphs = text.split('\n')
        for para in paragraphs:
            if len(para.strip()) < 20: # 忽略太短的段落
                continue
                
            processed_para = preprocess_text(para)
            if len(processed_para.split()) >= 5: # 确保段落包含足够的有效词汇
                documents.append(processed_para)
                timestamps.append(year)
                
    return documents, timestamps

def run_bertopic_analysis():
    # 1. 准备数据
    docs, timestamps = load_and_preprocess_data()
    print(f"Total documents (paragraphs) loaded: {len(docs)}")
    
    if len(docs) < 50:
        print("Not enough documents for meaningful topic modeling. Check data.")
        return
        
    # 2. 初始化 CountVectorizer (帮助BERTopic更好地处理中文分词结果)
    vectorizer_model = CountVectorizer(stop_words=list(STOPWORDS))
    
    # 3. 训练 BERTopic 模型
    print("Training BERTopic model...")
    topic_model = BERTopic(
        language="multilingual", # 支持中文的多语言模型
        vectorizer_model=vectorizer_model,
        nr_topics="auto",        # 自动决定主题数量
        min_topic_size=10,       # 每个主题最少的文档数
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
    # 4. 获取主题信息
    topic_info = topic_model.get_topic_info()
    print("\nTop Topics Found:")
    print(topic_info.head(10))
    topic_info.to_csv(os.path.join(OUTPUT_DIR, 'topic_info.csv'), index=False, encoding='utf-8-sig')
    
    # 5. 动态主题建模 (Dynamic Topic Modeling)
    print("\nGenerating Dynamic Topics over time...")
    # BERTopic需要时间戳列表与文档列表长度一致
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    
    # 保存动态主题数据
    topics_over_time.to_csv(os.path.join(OUTPUT_DIR, 'topics_over_time.csv'), index=False, encoding='utf-8-sig')
    
    # 6. 可视化
    print("\nGenerating visualizations...")
    
    # 主题条形图 (Top words for each topic)
    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=8)
        fig_barchart.write_html(os.path.join(OUTPUT_DIR, 'topic_barchart.html'))
    except Exception as e:
        print(f"Could not generate barchart: {e}")

    # 动态主题演化图 (Topics over time)
    try:
        # 选取最热门的前6个真实主题（排除-1离群点）
        top_topics = topic_info[topic_info['Topic'] != -1].head(6)['Topic'].tolist()
        fig_over_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=6)
        fig_over_time.write_html(os.path.join(OUTPUT_DIR, 'topics_over_time.html'))
    except Exception as e:
        print(f"Could not generate topics over time plot: {e}")
        
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_bertopic_analysis()
