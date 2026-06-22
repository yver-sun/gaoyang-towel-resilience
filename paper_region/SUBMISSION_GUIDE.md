# REGION 投稿操作指南

## 投稿网址
https://openjournals.wu.ac.at/ojs/index.php/region

---

## 一、文件清单

### 提交到 OJS 的文件

| 文件 | 用途 | 路径 |
|------|------|------|
| **盲审稿 PDF** | 主稿件（双盲：无作者信息） | `paper_region/main.pdf`（13页，713KB） |
| **盲审稿 LaTeX** | 源文件（可选上传） | `paper_region/main.tex` |
| **参考文献** | BibTeX | `paper_region/references.bib` |
| **投稿信** | 粘贴到 OJS 文本框 | `paper_region/Cover_Letter.md` |

### 自己留存的文件

| 文件 | 用途 | 路径 |
|------|------|------|
| **最终版 PDF**（含作者） | 自己存档 | `paper_region/main_final.pdf`（11页，694KB） |
| **最终版 LaTeX** | 录用后提交 | `paper_region/main_final.tex` |

---

## 二、OJS 提交步骤

### Step 1: 注册/登录
访问 https://openjournals.wu.ac.at/ojs/index.php/region/login
- 通讯作者（Yanlin Shi）注册账号
- ORCID 建议绑定

### Step 2: New Submission
点击 "New Submission" → 进入 5 步提交流程

### Step 3: Upload File（上传文件）
- **Article Component**: 选择 "Article Text"
- **Upload File**: 上传 `main.pdf`
- 可选：再上传 `main.tex`（Article Component → "Other"，注明 "LaTeX source"）

### Step 4: Enter Metadata（元数据）

**Title**（粘贴）:
```
Policy Attention versus Institutional Supply: How Local Governments Foster Cluster Resilience---A Mixed-Methods Case Study of China's Gaoyang Textile Industry
```

**Abstract**（粘贴以下内容）:
```
Traditional labor-intensive industrial clusters in developing economies face existential threats from compounding environmental, trade, and technological shocks, yet the micro-level institutional mechanisms through which local governments foster cluster adaptability remain undertheorized. This study advances a conceptual distinction between policy attention (discursive emphasis in official documents) and institutional supply (actual resource allocation), and traces two sequential micro-mechanisms---compliance cost socialization through collective environmental infrastructure, and transaction cost reduction through e-commerce and regional branding---through a 25-year (2000--2024) mixed-methods case study of China's largest towel production cluster in Gaoyang County. We combine BERTopic dynamic topic modeling of government work reports with descriptive quantitative patterns from Synthetic Control Method and Interrupted Time Series analysis. The case reveals a two-stage resilience trajectory: environmental infrastructure investment secured baseline resilience during the 2017 regulatory shock, while subsequent digital and branding policies fostered value resilience. Quantitative patterns are directionally consistent with the theoretical framework but are properly interpreted as mechanism-illustrative rather than causal-confirmatory, given single-case constraints and limited statistical power. The paper contributes a portable conceptual framework for distinguishing policy signals from institutional actions, a microeconomic formalization of compliance cost and transaction cost channels, and a transparent template for mixed-methods policy evaluation under archival data constraints.
```

**Authors**（按顺序添加，Starred(*) = Corresponding Author）:
1. Yiwen Sun — School of Mathematics and Statistics, Beijing Technology and Business University, Beijing, China — yver_sun@163.com
2. Yanlin Shi* — School of Languages and Communication, Beijing Technology and Business University, Beijing, China — shiyanlin@th.btbu.edu.cn
3. Jiarui Liang — School of Mathematics and Statistics, Beijing Technology and Business University, Beijing, China — 2151460699@qq.com

**Keywords**:
```
Regional Economic Resilience, Policy Attention, Institutional Supply, Industrial Clusters, BERTopic, Synthetic Control Method, China, Textile Industry
```

**JEL Classification**:
```
R11, R58, Q58, O25, C21
```

### Step 5: Cover Letter（粘贴以下内容到 "Comments for the Editor" 框）

直接粘贴 [Cover_Letter.md](Cover_Letter.md) 的内容。

### Step 6: Suggest Reviewers（建议审稿人）

在 OJS 系统中填入（如果有此选项）：

1. **Prof. Ron Boschma** — Utrecht University — r.a.boschma@uu.nl
2. **Prof. Robert Hassink** — Kiel University — hassink@geographie.uni-kiel.de
3. **Prof. Markku Sotarauta** — Tampere University — markku.sotarauta@tuni.fi
4. **Prof. Xiaohui Hu** — Nanjing Normal University — huxh@njnu.edu.cn

### Step 7: Confirm & Submit
- 勾选所有声明（原创性、无多投、全体作者同意等）
- 点击 "Submit"

---

## 三、注意事项

- **双盲**：OJS 会自动剥离 PDF 元数据中的作者信息，但我们的盲审稿已手动匿名
- **格式要求**：REGION 无严格格式要求（录用后排版），12pt/双倍行距非必需
- **审稿周期**：中位数 3.6 个月，初审约 10 周
- **APC**：免费（Diamond OA）
- **录用后**：需提交 `main_final.tex`（含作者信息版本）+ 所有高清图片源文件
