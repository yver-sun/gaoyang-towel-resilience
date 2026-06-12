import requests
import os
import time
from bs4 import BeautifulSoup

OUTPUT_DIR = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports"

reports_to_fetch = [
    # 河北省 - complete URLs from hebei.gov.cn
    {
        "filename": "河北省_2000_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/aed868b1-af84-497b-af67-9d4c5814c73a.html",
        "title": "河北省2000年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2000-01-20",
            "报告人": "河北省人民政府省长 钮茂生"
        }
    },
    {
        "filename": "河北省_2001_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/8ad5256a-c625-4a22-81ab-77715bf3b346.html",
        "title": "河北省2001年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2001-01-08",
            "报告人": "河北省人民政府省长 钮茂生"
        }
    },
    {
        "filename": "河北省_2002_report.txt",
        "url": "https://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/96458b1a-4e65-4874-b558-efff398d4e44.html",
        "title": "河北省2002年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2002-01-24",
            "报告人": "河北省人民政府省长 钮茂生"
        }
    },
    {
        "filename": "河北省_2003_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/e7981539-3eb3-4994-88b0-c0ffd851d69f.html",
        "title": "河北省2003年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2003-01-13",
            "报告人": "河北省人民政府代省长 季允石"
        }
    },
    {
        "filename": "河北省_2004_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202309/12/75142eba-cd81-4f0f-baf4-c0b4ead257ad.html",
        "title": "河北省2004年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2004-01-12",
            "报告人": "河北省人民政府省长 季允石"
        }
    },
    {
        "filename": "河北省_2005_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/77c8e8d5-36f6-49a3-92c1-2b85b3e7c2d5.html",
        "title": "河北省2005年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2005-01-12",
            "报告人": "河北省人民政府省长 季允石"
        }
    },
    {
        "filename": "河北省_2006_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/dbbbe8d5-887e-4987-ad80-4d3744eaf8ca.html",
        "title": "河北省2006年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2006-02-18",
            "报告人": "河北省人民政府省长 季允石"
        }
    },
    {
        "filename": "河北省_2007_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/21708abe-5ba2-4ea2-9b77-f5f699f2afd3.html",
        "title": "河北省2007年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2007-04-05",
            "报告人": "河北省人民政府代省长 郭庚茂"
        }
    },
    {
        "filename": "河北省_2008_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/8f5c7d3e-4b2a-49c1-8a6d-7e3f2b1c9d8a.html",
        "title": "河北省2008年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2008-01-22",
            "报告人": "河北省人民政府省长 郭庚茂"
        }
    },
    {
        "filename": "河北省_2009_report.txt",
        "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/6855a930-1b4d-4041-a81c-071462accb34.html",
        "title": "河北省2009年政府工作报告",
        "meta": {
            "发文单位": "河北省人民政府",
            "发文时间": "2009-02-13",
            "报告人": "河北省人民政府代省长 胡春华"
        }
    },
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

success_count = 0
fail_count = 0

for report in reports_to_fetch:
    filename = report["filename"]
    url = report["url"]
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        print(f"正在获取: {filename}")
        resp = requests.get(url, headers=headers, timeout=30)
        resp.encoding = resp.apparent_encoding
        
        if resp.status_code != 200:
            print(f"  失败: HTTP {resp.status_code}")
            fail_count += 1
            continue
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Try to find the main content
        content_div = soup.find('div', class_='TRS_Editor') or soup.find('div', id='content') or soup.find('div', class_='content')
        
        if content_div:
            text = content_div.get_text('\n', strip=True)
        else:
            # Fallback: get all text from the main area
            text = soup.get_text('\n', strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n\n'.join(lines)
        
        if len(clean_text) > 500:
            # Write the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"标题：{report['title']}\n")
                f.write(f"发文单位：{report['meta']['发文单位']}\n")
                f.write(f"发文时间：{report['meta']['发文时间']}\n")
                f.write(f"报告人：{report['meta']['报告人']}\n")
                f.write(f"来源URL：{url}\n\n")
                f.write("正文：\n\n")
                f.write(clean_text)
            
            file_size = os.path.getsize(filepath)
            print(f"  成功: {file_size} bytes")
            success_count += 1
        else:
            print(f"  失败: 内容过少 ({len(clean_text)} chars)")
            fail_count += 1
        
        time.sleep(1)
        
    except Exception as e:
        print(f"  失败: {str(e)}")
        fail_count += 1

print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")
