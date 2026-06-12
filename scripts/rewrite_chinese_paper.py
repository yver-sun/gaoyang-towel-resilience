# -*- coding: utf-8 -*-
import re

with open('C:/Users/Yver/Desktop/史岩林/高阳毛巾/paper_latex/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace abstract
old_abstract_start = text.find(r'\begin{abstract}')
old_abstract_end = text.find(r'\end{abstract}', old_abstract_start)
new_abstract = r'''\begin{abstract}
随着全球经济步入以逆全球化、环保规制和结构性市场变迁为特征的转型期，传统劳动密集型产业集群面临外部冲击带来的适应性挑战。本文以中国高阳县毛巾产业集群（2000--2024）为研究对象，采用混合方法纵向单案例研究设计，探讨地方政府政策干预与产业集群韧性的协同演化机制。首先，利用BERTopic模型对25年的地方政府工作报告进行全文本建模，刻画政策焦点的时序变迁，并引入多重安慰剂主题检验确保NLP指标的区分效度；其次，运用合成控制法（SCM）和中断时间序列分析（ITS），基于高阳县及保定市24个区县的面板数据，对环保基建冲击的企业存活效应进行多源三角验证。研究发现，高阳集群的韧性演化经历了从"规模扩张"到"底盘韧性"、最终向"价值韧性"的递进过程，地方政府的敏捷政策注意力转移在其中发挥了关键的支撑作用。本文通过量化政策注意力（而非制度支出）并引入微观成本视角，对区域经济韧性文献进行了实证补充。

\vspace{0.5cm}
\noindent\textbf{关键词：} 区域经济韧性；产业集聚；BERTopic；合成控制法；中断时间序列；制度供给
\end{abstract}'''

# Replace keywords section
assert old_abstract_start > 0
text = text[:old_abstract_start] + new_abstract + text[old_abstract_end + len(r'\end{abstract}'):]

print('Abstract replaced.')

# Now fix the methods section - replace SDID with SCM+ITS
# Find the methods section
methods_start = text.find(r'\section{研究方法与数据设计}')
methods_end = text.find(r'\section{', methods_start + 1)

new_methods = r'''\section{研究方法与数据设计}
\label{sec:method}

为克服单案例研究在外部有效性上的不足，以及纯定量研究在过程解释上的黑箱问题，本文采用严格的"定性过程追踪 + 定量三角验证"的混合方法（Mixed-Methods）设计。

\subsection{数据来源与变量说明}
本研究的数据涵盖了宏观经济面板、微观企业注册特征以及本地专属的产业指数。
\begin{enumerate}
    \item \textbf{政策文本语料库：} 收集高阳县2000年至2024年连续25份政府工作报告（约20万字），文本来源于高阳县人民政府官方网站及地方志档案。
    \item \textbf{企业注册与存活数据：} 提取自国家市场监督管理总局全国企业信用信息公示系统，匹配高阳县及保定市24个区县的面板数据（2000-2024）。主要因变量为\textbf{新增纺织企业注册数对数}，用于衡量集群在危机中的底层生存与活力。
    \item \textbf{机制与经济绩效变量：} 选取\textbf{电商企业注册数对数}作为机制变量。此外，提取"河北·高阳纺织指数"中的\textbf{产品价格指数}（基期=100，仅2020-2026年可用）。
    \item \textbf{控制变量：} 包括地区生产总值（GDP）对数、常住人口对数等，用于控制地区经济规模与人口禀赋的异质性。
\end{enumerate}

\subsection{基于BERTopic的政策文本演化分析}
为精准捕捉政策焦点的时序变迁，本文采用\textbf{BERTopic模型} \cite{Grootendorst2022}。与传统LDA不同，BERTopic利用预训练Transformer嵌入来提取高连贯性的主题，并使用基于类别的TF-IDF程序进行主题表示。

在效度验证方面，本文进行了两项检验：
\begin{enumerate}
    \item \textbf{多重安慰剂主题检验：} 除教育主题外，增加文化旅游、医疗卫生等至少三个与纺织产业正交的主题。如果这些正交主题的评分与产业指标的相关均不显著，则表明模型提取的政策注意力指标具有良好的区分效度。
    \item \textbf{主题一致性报告：} 报告各主题的CV一致性分数（Coherence Score），并对UMAP和HDBSCAN的超参数选择依据进行说明。
\end{enumerate}
数学上，文档使用Sentence-BERT嵌入到稠密向量空间，通过UMAP降维，使用HDBSCAN聚类，最后通过c-TF-IDF生成主题表示：
\begin{equation}
    W_{t,c} = tf_{t,c} \times \log\left(1 + \frac{A}{tf_t}\right)
\end{equation}

\subsection{因果推断：合成控制法与中断时间序列}
\label{sec:causal}
为三角验证NLP发现，本文综合运用两种互补的因果推断方法。

\textbf{合成控制法（SCM）：} 采用Abadie et al. \cite{Abadie2010}提出的合成控制法，以保定市其他23个区县为供体池（Donor Pool），为高阳县合成一个反事实对照。合成权重通过最小化处理前（2000-2016年）结果变量的均方预测误差（RMSPE）估算。处理效应定义为2017年后高阳县与合成对照之间的结果差异。安慰剂检验通过将处理状态随机分配给供体池区县（1000次迭代）进行。

\textbf{中断时间序列分析（ITS）：} 采用Bernal et al. \cite{Bernal2017}推荐的ITS方法，对高阳县单县25年时间序列数据进行分段回归：
\begin{equation}
    Y_t = \beta_0 + \beta_1 T_t + \beta_2 Post_t + \beta_3 (T_t \times Post_t) + \gamma P_t + \varepsilon_t
\end{equation}
其中$Y_t$为年度纺织企业新增注册对数，$T_t$为时间趋势，$Post_t$为2017年后的处理指示变量，$P_t$为NLP提取的政策注意力评分向量。标准误使用Newey-West HAC估计控制序列自相关。

两种方法各有侧重：SCM利用跨县域变异构造反事实，控制不可观测的时间恒定混杂因素；ITS利用单县时间序列变异，直接控制可观测的随时间变化混杂因素。二者的交叉验证提供了比单一方法更强的多维证据。

\subsection{机制检验}
为验证H2（价值韧性），建立如下时间序列回归模型：
\begin{equation}
    M_t = \alpha + \beta \times PolicyAttention^{Digital}_t + \gamma X_t + \varepsilon_t
\end{equation}
其中$M_t$为机制变量向量（电商企业注册对数、产品价格指数），$PolicyAttention^{Digital}_t$为NLP提取的数字与品牌政策注意力得分，$X_t$为控制变量。

\textbf{需要强调的是}，本研究中的政策注意力评分仅覆盖高阳县（以及作为参考基准的保定市级和北京市级），因此无法实施需要多处理组-对照组的传统双重差分（DID）或合成双重差分（SDID）分析。SCM虽利用了保定市24个区县的跨县面板，但政策注意力评分本身仅针对高阳县构造，各对照县仅提供结果变量（纺织企业注册数）信息，不包含政策内容差异。这是本研究最重要的数据约束，在结果解读中需充分认识。'''

methods_end = text.find(r'\section{', methods_start + len(r'\section{研究方法与数据设计}'))
if methods_end < 0:
    methods_end = text.find(r'\section{分析结果', methods_start)

if methods_end > 0:
    text = text[:methods_start] + new_methods + text[methods_end:]
    print('Methods section replaced.')
else:
    print('WARNING: Could not find methods end boundary')

with open('C:/Users/Yver/Desktop/史岩林/高阳毛巾/paper_latex/main.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Final file size: {len(text)} bytes')
