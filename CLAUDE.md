# 高阳县毛巾产业政策强度与集群韧性研究

## 项目目标

量化地方政府制度供给（政策强度）对传统劳动密集型产业集群韧性的因果托举作用。

**核心问题**：2017年环保规制冲击下，高阳县毛巾产业如何跨越"死亡谷"？政策干预扮演了什么角色？

## 当前阶段

研究设计已批准（见 `~/.claude/plans/humming-wobbling-avalanche.md`），待执行 Phase A 五条数据 Pipeline。

## 数据资产速查

| 数据 | 路径 | 状态 |
|------|------|------|
| LLM政策评分面板(81行×33列) | `output/policy_scores_panel.csv` | 已完成 |
| 企业注册-高阳(44年) | `output/gaoyang_enterprise_registration.csv` | 已提取 |
| 企业注册-保定24区县 | `output/baoding_enterprise_registration.csv` | 已提取 |
| 企业注册-纺织细分 | `output/gaoyang_textile_registration.csv` | 已提取 |
| 政策原文(92篇) | `data/policies/高阳县毛巾产业政策文件_全量_*.csv` | 未使用 |
| 纺织指数(15个Excel) | `data/textile_indices/*.xlsx` | 未使用 |
| 全国县域面板 | `data/panel/中国县域统计面板数据_最终版.csv` | 未使用 |
| 高阳专属面板 | `data/panel/高阳县面板数据.csv` | 未使用 |
| 保定区县报告(168行) | `data/policies/保定市各县政府工作报告.xlsx` | 未使用 |

## 技术环境

- Windows 11 + Git Bash
- Python（科学计算栈：pandas, numpy, scipy, statsmodels）
- Ollama 本地部署（qwen2.5:1.5b + qwen2.5:14b，已完成81份报告评分）
- 无 GPU 依赖

## 工作约定

- 代码默认不加注释（除非 WHY 不显而易见的）
- 学术严谨优先：每个分析步骤需要验证假设条件
- 输出到 `output/` 和 `analysis/` 目录
