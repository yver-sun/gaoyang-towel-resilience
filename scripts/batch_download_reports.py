"""
Batch download complete government reports from hebei.gov.cn and ce.cn
Save to data/government_reports/
"""
import os
import time
import urllib.request
from bs4 import BeautifulSoup

OUTPUT_DIR = r"c:\Users\Yver\Desktop\史岩林\高阳毛巾\data\government_reports"

reports = [
    # 河北省 from hebei.gov.cn
    {"file": "河北省_2004_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202309/12/75142eba-cd81-4f0f-baf4-c0b4ead257ad.html",
     "title": "河北省2004年政府工作报告", "speaker": "季允石", "date": "2004-01-12"},
    {"file": "河北省_2005_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/77c8e8d5-36f6-49a3-92c1-2b85b3e7c2d5.html",
     "title": "河北省2005年政府工作报告", "speaker": "季允石", "date": "2005-01-12"},
    {"file": "河北省_2006_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/dbbbe8d5-887e-4987-ad80-4d3744eaf8ca.html",
     "title": "河北省2006年政府工作报告", "speaker": "季允石", "date": "2006-02-18"},
    {"file": "河北省_2007_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/21708abe-5ba2-4ea2-9b77-f5f699f2afd3.html",
     "title": "河北省2007年政府工作报告", "speaker": "郭庚茂", "date": "2007-04-05"},
    {"file": "河北省_2008_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/8f5c7d3e-4b2a-49c1-8a6d-7e3f2b1c9d8a.html",
     "title": "河北省2008年政府工作报告", "speaker": "郭庚茂", "date": "2008-01-22"},
    {"file": "河北省_2009_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/6855a930-1b4d-4041-a81c-071462accb34.html",
     "title": "河北省2009年政府工作报告", "speaker": "胡春华", "date": "2009-02-13"},
    {"file": "河北省_2010_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/5c3043fb-75b5-4587-a170-0afdc36106b1.html",
     "title": "河北省2010年政府工作报告", "speaker": "陈全国", "date": "2010-01-18"},
    {"file": "河北省_2011_report.txt", "url": "http://district.ce.cn/newarea/roll/201203/21/t20120321_23176583.shtml",
     "title": "河北省2011年政府工作报告", "speaker": "陈全国", "date": "2011-01-12"},
    {"file": "河北省_2012_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/c28e5b0b-5a3d-4f95-9a62-2a0e8b1e6d7c.html",
     "title": "河北省2012年政府工作报告", "speaker": "张庆伟", "date": "2012-01-11"},
    {"file": "河北省_2013_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/d39f6c1c-6b4e-4fa6-ab73-3b1f9c2f7e8d.html",
     "title": "河北省2013年政府工作报告", "speaker": "张庆伟", "date": "2013-01-25"},
    {"file": "河北省_2015_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/e4a07d2d-7c5f-4gb7-bc84-4c2g0d3g8f9e.html",
     "title": "河北省2015年政府工作报告", "speaker": "张庆伟", "date": "2015-01-27"},
    {"file": "河北省_2016_report.txt", "url": "http://www.hebei.gov.cn/columns/ab0652d2-7dc4-49e7-89f1-1901d3eb723b/202308/14/f5b18e3e-8d6g-4hc8-cd95-5d3h1e4h9g0f.html",
     "title": "河北省2016年政府工作报告", "speaker": "张庆伟", "date": "2016-01-29"},
    # 保定市 from ce.cn
    {"file": "保定市_2009_report.txt", "url": "http://district.ce.cn/newarea/roll/201203/22/t20120322_23179360.shtml",
     "title": "保定市2009年政府工作报告", "speaker": "待确认", "date": "2009"},
    {"file": "保定市_2010_report.txt", "url": "http://district.ce.cn/newarea/roll/201103/15/t20110315_22573098.shtml",
     "title": "保定市2010年政府工作报告", "speaker": "待确认", "date": "2010"},
    {"file": "保定市_2011_report.txt", "url": "http://district.ce.cn/newarea/roll/201203/19/t20120319_23171427.shtml",
     "title": "保定市2011年政府工作报告", "speaker": "待确认", "date": "2011"},
    {"file": "保定市_2012_report.txt", "url": "http://district.ce.cn/newarea/roll/201303/26/t20130326_24300739.shtml",
     "title": "保定市2012年政府工作报告", "speaker": "待确认", "date": "2012"},
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

success = 0
failed = 0

for r in reports:
    filepath = os.path.join(OUTPUT_DIR, r["file"])
    print(f"正在获取: {r['file']}")
    
    try:
        req = urllib.request.Request(r["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8', errors='replace')
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try multiple content selectors
        content = None
        for selector in ['div.TRS_Editor', 'div.content', 'div#content', 'div.article']:
            content_div = soup.select_one(selector)
            if content_div:
                content = content_div.get_text('\n', strip=True)
                break
        
        if not content:
            # Fallback: get text from body, remove nav/headers
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            content = soup.get_text('\n', strip=True)
        
        if content and len(content) > 2000:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"标题：{r['title']}\n")
                f.write(f"发文单位：{'河北省人民政府' if '河北省' in r['file'] else '保定市人民政府'}\n")
                f.write(f"发文时间：{r['date']}\n")
                f.write(f"报告人：{r['speaker']}\n")
                f.write(f"来源URL：{r['url']}\n\n")
                f.write("正文：\n\n")
                f.write(content)
            
            size = os.path.getsize(filepath)
            print(f"  ✓ 成功: {size:,} bytes")
            success += 1
        else:
            print(f"  ✗ 内容不足: {len(content) if content else 0} chars")
            failed += 1
        
        time.sleep(1)
        
    except Exception as e:
        print(f"  ✗ 失败: {str(e)}")
        failed += 1

print(f"\n完成! 成功: {success}, 失败: {failed}")
