"""
高阳县毛巾产业政策文件批量抓取脚本
从搜索结果URL批量获取完整正文内容，生成Excel
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
BASE_DIR = r'c:\Users\Yver\Desktop\史岩林\高阳毛巾'
OUTPUT_DIR = os.path.join(BASE_DIR, '高阳县政策文件')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# 从搜索结果整理的URL列表（基于搜索获取的真实URL）
URL_LIST = [
    {
        'url': 'https://gxt.hebei.gov.cn/hbgyhxxht/xwzx32/stdt65/2025041919133291914/index.html',
        'title': '河北省特色产业集群发展经验交流 | 数字化引领高阳纺织产业集群的转型之路',
        'source': '河北省工业和信息化厅',
        'date': '2025-01-09',
    },
    {
        'url': 'https://gaoyang.gov.cn/info_show.asp?infoid=11022',
        'title': '经济强省 美丽河北·一线观察丨保定高阳：小毛巾"织"出大产业',
        'source': '高阳县政府',
        'date': '2024-12-12',
    },
    {
        'url': 'https://gaoyang.gov.cn/info_show.asp?infoid=10382',
        'title': '高阳县毛巾纺织入选全国中小企业特色产业集群',
        'source': '高阳县政府',
        'date': '2023-11-26',
    },
    {
        'url': 'https://www.spb.gov.cn/gjyzj/c100196/202412/56ca66d70b7d4b278b3233c8165a8508.shtml',
        'title': '"高阳优品""纺"遍全国',
        'source': '国家邮政局',
        'date': '2024-12-04',
    },
    {
        'url': 'https://gaoyang.gov.cn/info_show.asp?infoid=11003',
        'title': '保定高阳：小毛巾"织"出大产业',
        'source': '高阳县政府',
        'date': '2024-12-03',
    },
    {
        'url': 'https://gxt.hebei.gov.cn/hbgyhxxht/xwzx32/dfgz28/2026032009422397104/index.html',
        'title': '高阳共享智造赋能产业绿色转型',
        'source': '河北省工业和信息化厅',
        'date': '2026-03-20',
    },
    {
        'url': 'http://he.people.com.cn/BIG5/n2/2025/0305/c192235-41154616.html',
        'title': '"高阳毛巾"成为中国消费名品区域品牌',
        'source': '人民网-河北频道',
        'date': '2025-03-05',
    },
    {
        'url': 'https://gaoyang.gov.cn/info_show.asp?infoid=5509',
        'title': '高阳毛巾"变形记"（三）：创新驱动 让传统产业"新"起来',
        'source': '高阳县政府',
        'date': '2018-12-12',
    },
    {
        'url': 'http://www.gaoyang.gov.cn/com_show.asp?infoid=109',
        'title': '高阳毛巾紧抓机遇加速崛起',
        'source': '高阳县政府',
        'date': '2015-07-27',
    },
    {
        'url': 'http://www.gaoyang.gov.cn/info_show.asp?infoid=5495',
        'title': '高阳毛巾"变形记" ——高阳县建设循环经济示范区提升传统纺织产业的调查',
        'source': '高阳县政府',
        'date': '2018-11-20',
    },
    {
        'url': 'https://gaoyang.gov.cn/info_show.asp?infoid=10503',
        'title': '河北唯一！高阳县成为国家循环经济示范县',
        'source': '高阳县政府',
        'date': '2024-02-06',
    },
    {
        'url': 'http://www.gaoyang.gov.cn/info_show.asp?infoid=5490',
        'title': '高阳毛巾"变形记"（一）：集聚效应 让纺织产业"活"起来',
        'source': '高阳县政府',
        'date': '2018-12-06',
    },
    {
        'url': 'http://www.gaoyang.gov.cn/info_show.asp?infoid=8879',
        'title': '中共高阳县委关于巡视整改进展情况的通报',
        'source': '高阳县政府',
        'date': '2021-11-15',
    },
    {
        'url': 'https://bd.hebccw.cn/system/2025/04/22/102055296.shtml',
        'title': '保定高阳县推动传统纺织产业高端化智能化绿色化发展',
        'source': '长城网',
        'date': '2025-04-23',
    },
]

def extract_content_from_page(url_info):
    """从网页提取完整正文"""
    url = url_info['url']
    print(f"抓取: {url_info['title'][:40]}...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试从页面提取标题
        title = url_info.get('title', '')
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag and len(title_tag.get_text(strip=True)) > 5:
            title = title_tag.get_text(strip=True)
        
        # 尝试提取发文单位/来源
        publish_dept = url_info.get('source', '')
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            if any(kw in name for kw in ['source', 'author', 'publisher', '来源']):
                publish_dept = publish_dept or content
        
        # 从页面文本中提取来源
        for span in soup.find_all(['span', 'div', 'p']):
            text = span.get_text(strip=True)
            if re.search(r'来源[:：]|发布单位[:：]|作者[:：]', text):
                match = re.search(r'来源[:：]\s*(.+?)(?:\s{2,}|$)', text)
                if match:
                    publish_dept = publish_dept or match.group(1).strip()
        
        # 尝试提取时间
        publish_time = url_info.get('date', '')
        
        # 从页面文本中提取时间
        time_patterns = [
            r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
        ]
        for text in soup.stripped_strings:
            for pattern in time_patterns:
                match = re.search(pattern, text)
                if match and len(match.group(1)) >= 8:
                    publish_time = publish_time or match.group(1)
                    break
        
        # 提取正文（尝试多种常见的正文容器）
        body = ''
        
        # 策略1: 查找常见的正文容器
        content_selectors = [
            {'class_': 'TRS_Editor'},
            {'class_': re.compile(r'content|article|detail', re.I)},
            {'id': re.compile(r'content|article|Zoom', re.I)},
            {'class_': 'page-content'},
            {'class_': 'main-content'},
        ]
        
        content_div = None
        for selector in content_selectors:
            content_div = soup.find('div', **selector)
            if content_div and len(content_div.get_text()) > 200:
                break
        
        if content_div:
            # 清理无关元素
            for tag in content_div.find_all(['script', 'style', 'iframe', 'noscript']):
                tag.decompose()
            body = content_div.get_text('\n', strip=True)
        else:
            # 策略2: 提取所有p标签
            paragraphs = soup.find_all('p')
            body_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20:  # 过滤掉太短的文本
                    body_parts.append(text)
            body = '\n\n'.join(body_parts)
        
        if not body or len(body) < 100:
            # 策略3: 提取所有有意义的div
            divs = soup.find_all('div')
            max_text = ''
            for div in divs:
                text = div.get_text(strip=True)
                if len(text) > len(max_text):
                    max_text = text
            body = max_text
        
        # 清理正文
        body = re.sub(r'\n{3,}', '\n\n', body)
        body = re.sub(r'[ \t]+', ' ', body)
        
        return {
            'title': title,
            'publish_dept': publish_dept,
            'publish_time': publish_time,
            'doc_number': '',
            'body': body,
            'source_url': url,
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        print(f"  抓取失败: {e}")
        return {
            'title': url_info.get('title', ''),
            'publish_dept': url_info.get('source', ''),
            'publish_time': url_info.get('date', ''),
            'doc_number': '',
            'body': f'抓取失败: {str(e)}',
            'source_url': url,
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

def save_to_excel(docs):
    """保存为Excel文件"""
    if not docs:
        print("无数据可保存")
        return
    
    df = pd.DataFrame(docs)
    
    # 提取政策类别
    def categorize(body, title):
        text = (body or '') + ' ' + (title or '')
        categories = []
        if '毛巾' in text or '浴巾' in text:
            categories.append('毛巾产业')
        if '纺织' in text or '印染' in text or '织造' in text:
            categories.append('纺织行业')
        if '产业集群' in text or '产业园区' in text or '共享工厂' in text:
            categories.append('产业集群')
        if '数字化' in text or '智能化' in text or '产业大脑' in text:
            categories.append('数字化转型')
        if '环保' in text or '污染治理' in text or '绿色' in text or '循环经济' in text:
            categories.append('环保治理')
        if '品牌' in text or '商标' in text or '高阳优品' in text:
            categories.append('品牌建设')
        if '电商' in text or '直播' in text or '跨境' in text:
            categories.append('电子商务')
        if '循环经济' in text or '示范县' in text:
            categories.append('循环经济')
        return '、'.join(categories) if categories else '综合'
    
    df['category'] = df.apply(lambda x: categorize(x.get('body', ''), x.get('title', '')), axis=1)
    
    # 按时间排序
    df['sort_year'] = df['publish_time'].apply(lambda x: re.search(r'(\d{4})', x).group(1) if re.search(r'(\d{4})', x) else '0000')
    df = df.sort_values('sort_year', ascending=False)
    df = df.drop('sort_year', axis=1)
    
    # 重新排列列顺序
    df = df[['title', 'publish_dept', 'publish_time', 'doc_number', 'category', 'body', 'source_url', 'crawl_time']]
    df.columns = ['标题', '发文单位', '发文时间', '文号', '政策类别', '正文', '来源URL', '爬取时间']
    
    # 添加序号
    df.insert(0, '序号', range(1, len(df) + 1))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'高阳县毛巾产业政策文件_{timestamp}.xlsx')
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='政策文件', index=False)
        
        # 设置列宽
        worksheet = writer.sheets['政策文件']
        worksheet.set_column('A:A', 6)   # 序号
        worksheet.set_column('B:B', 40)  # 标题
        worksheet.set_column('C:C', 20)  # 发文单位
        worksheet.set_column('D:D', 15)  # 发文时间
        worksheet.set_column('E:E', 12)  # 文号
        worksheet.set_column('F:F', 25)  # 政策类别
        worksheet.set_column('G:G', 80)  # 正文
        worksheet.set_column('H:H', 60)  # 来源URL
        worksheet.set_column('I:I', 20)  # 爬取时间
    
    # 同时保存CSV
    csv_file = os.path.join(OUTPUT_DIR, f'高阳县毛巾产业政策文件_{timestamp}.csv')
    df.to_csv(csv_file, encoding='utf-8-sig', index=False)
    
    print(f"\n保存成功: {output_file}")
    print(f"文件数: {len(df)}")
    
    return output_file

# 主流程
def main():
    print("=" * 60)
    print("高阳县毛巾产业政策文件批量抓取")
    print("=" * 60)
    print(f"目标URL数: {len(URL_LIST)}")
    print("=" * 60)
    
    all_docs = []
    
    # 顺序抓取（避免并发请求被封）
    for url_info in URL_LIST:
        doc = extract_content_from_page(url_info)
        if doc:
            all_docs.append(doc)
        time.sleep(2)  # 礼貌延迟
    
    print(f"\n\n{'='*60}")
    print(f"抓取完成!")
    print(f"总文档数: {len(all_docs)}")
    print(f"{'='*60}")
    
    if all_docs:
        save_to_excel(all_docs)
    else:
        print("未获取到有效数据")

if __name__ == '__main__':
    main()
