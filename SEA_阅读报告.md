# SEA 阅读报告

**SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs**  
（面向多模态大模型词元级视觉—文本整合的监督嵌入对齐）

> **出处**：EMNLP 2025 主会，pp. 1058–1070  
> **代码**：<https://github.com/YuanyangYin/SEA>

**核心概括：** SEA 面向多模态大模型预训练阶段的视觉—文本表征错位问题，提出一种 token 级监督嵌入对齐方法。它先用 CLIP 类视觉—文本模型为每个视觉 patch 抽取连续的语义标签，再把这些标签映射到 LLM 自身的词嵌入空间，用双向对比损失约束 adapter 输出的 visual token，使视觉表征直接落在 LLM 能够解释的位置。整个过程不额外构造训练数据，也不增加推理开销，在 2B 到 13B 的多个骨干上一致提升，小模型受益最明显（Gemma-2B 平均 +7.61%）。

## 一、文章解决了什么下游任务？

**改善多模态大模型预训练阶段的视觉—文本对齐质量。**

多模态大模型（MLLM）通常由三段组成：视觉编码器把图像切成 patch 并提取特征，adapter 把这些 patch 特征转换成 LLM 可读的 visual token，再与文本 token 拼接后送入 LLM。所谓对齐，就是让这些 visual token 落在 LLM 已经理解的语义位置上——既保留图像原本的语义，又不要求 LLM 额外花容量去"翻译"一批陌生向量。

问题的典型表现有两个层面。表征层面，把某个 visual token 拿去词表里检索最近词，蓝莓所在 patch 可能对应到 bluejeans（牛仔裤），说明这个向量虽然能撑起一句通顺的描述，却没有忠实保存"蓝莓"的语义。行为层面，论文给出的例子是一张红色插座的图片，LLaVA-1.5 回答成白色，而经 SEA 训练的模型回答红色。问题不在于语言不通顺，而在于视觉证据在进入 LLM 之前就已经被扭曲了。

因此这项任务的目标不是提升生成流畅度，而是让每个 visual token **准确承载对应图像区域的语义**，并且**落在 LLM 原生嵌入空间**。后续的 VQA、描述生成、细粒度识别与 POPE 幻觉检测都依赖这个底层质量。从阶段上看，它属于**预训练期的表征对齐**；这与 Re-Align 那类后训练偏好对齐处在不同环节，两者可以叠加而非互斥。

## 二、传统做法都是怎么解决的？

传统 MLLM 沿用 LLaVA 式两阶段流程：第一阶段冻结视觉编码器与 LLM，只训练 adapter 做模态对齐；第二阶段做指令微调。给定图文对 $(X_{\text{image}},X_{\text{text}})$，前向过程为：

$$
X_v=g_\theta(f(X_{\text{image}})),\qquad X_t=\Psi(X_{\text{text}}),\qquad X_{\text{input}}=[x^v_0,\dots,x^v_m,x^t_0,\dots,x^t_n],
$$

这里 $f$ 是视觉编码器，$g_\theta$ 是可训练的 adapter，$\Psi$ 是 LLM 自身的词嵌入层，$X_v\in\mathbb{R}^{m\times d}$ 是 $m$ 个 visual token，$X_t$ 是文本 token。第一阶段只有 adapter 的梯度在更新，监督信号是标准自回归语言建模损失，也就是让模型依据图像与前文预测下一个文本 token。

问题在于这种监督是**图像级**的：一整张图只对应一整段 caption，约束的是"整句说得对不对"，并不约束"第 37 个 visual patch 到底表示什么"。后续工作尝试过更细的粒度，例如用最优传输做 token 分配（Park et al., 2024）、用回归约束特征（Shang et al., 2024），也有 AlignGPT 这类基于相似度的 token 指派、以及用分割与 OCR 信息增强 visual token 的做法。它们或停留在图像级、区域级，或改变的是输入侧的 token 构造，始终没有直接把监督目标定义在 LLM 真正工作的那个嵌入空间里，adapter 本身的对齐职责也没有被真正约束。

## 三、本文发现了什么此前解决不好的问题？

本文指出的新问题不是"模型完全看不懂图"，而是**图像级监督没有约束单个 visual token 是否落在正确的语义位置**。作者用两个可量化的诊断把这一点坐实。

第一，**语义信息失真**。对每个 visual token，在预定义词表 $W$ 中检索余弦相似度最高的词：

$$
w_j=\arg\max_{w\in W}\ \mathrm{sim}(x^v_j,\Psi(w)).
$$

结果显示传统 adapter 频繁把 visual token 映射到语义无关的词上（蓝莓对应 bluejeans），说明向量内部已经丢失了原区域语义。LLM 随后只能靠语言先验去"猜"，猜错就产生幻觉。

第二，**模态表征鸿沟**。作者用 t-SNE 对比了三组向量：传统 adapter 输出的 visual token、SEA 输出的 visual token、LLM 原生词嵌入，前两者与第三者之间存在明显距离。这个距离可以量化：

$$
D=\frac{1}{|X_v|}\sum_{x^v_j\in X_v}\ \min_{w\in W}\left\lVert x^v_j-\Psi(w)\right\rVert_2 .
$$

$D$ 越大，说明 visual token 离 LLM 熟悉的语义锚点越远，LLM 必须分配额外容量去解读错位输入，而不是调用预训练知识。论文用 MMLU 曲线佐证：指令微调过程中，没有预训练对齐的模型语言能力明显下降，图像级对齐只能小幅缓解；这个效应在小模型上尤其严重，因为容量本就紧张，视觉感知与语言能力之间的挤占更尖锐。

因此缺少的是一种方案：既能在 **token 级**提供细粒度监督，又把目标直接定义在 **LLM 原生嵌入空间**，同时不引入额外数据与推理成本。

## 四、本文具体怎么做？

核心思想可以概括为：**让 CLIP 给每个 patch 报几个语义候选词，把这些词变成 LLM 空间里的目标点，再用对比损失把 adapter 的输出拉过去。** 整条流程只在预训练第一阶段生效，第二阶段指令微调照常进行。

### 第一步：为每个视觉 patch 抽取语义标签

视觉编码器 $f$ 与文本编码器 $h$ 来自同一个对齐好的 CLIP 类模型，词表 $W$ 含 $q$ 个词。先分别编码：

$$
V=f(X_{\text{image}})\in\mathbb{R}^{m\times d},\qquad T=h(W)\in\mathbb{R}^{q\times d},
$$

再对每个 patch 取相似度最高的 $n$ 个词作为候选标签：

$$
(w_i,s_i)=\mathrm{TopN}_{j}\left\{-\cos(v_i,t_j)\right\}.
$$

其中 $v_i$ 是第 $i$ 个 patch 的视觉特征，$t_j$ 是第 $j$ 个词的文本嵌入，$w_i$ 与 $s_i$ 分别是入选词的索引与得分。这里取负余弦，是因为在 CLIP 嵌入空间中检索需要先取负（论文沿用 CLIP surgery 的约定）。同时把相似度小于 0 的词剔除，保证只保留正相关的标签。

保留多个标签而不是硬选一个，是为了表达视觉语义的连续性：一个 patch 往往同时是"蓝莓""水果""蓝色"，压成单个离散词会丢失信息；而多个标签的加权混合可以逼近这个连续语义，也避免视觉、文本编码器配对训练带来的语义漂移。词表规模方面，作者从预训练语料的句法分析中抽取实体与属性词，再并入 2of12 词表的常用词，最终约 400 万词。

（蓝莓 patch → blueberry、fruit、blue，而不是只硬选一个。）

### 第二步：按相似度加权采样

把候选标签的得分归一化成采样概率：

$$
S^{\text{norm}}_i=\frac{S_i}{\mathrm{sum}(S_i)},
$$

再按该分布为每个 patch 抽出一个标签。得分越高的词越常被选中，训练过程中等价于用多个相关词向量的加权混合去逼近该 patch 的真实语义；同时保留随机性，避免模型只记住词表里那个"最像"的词。

（blueberry 比 fruit 更相关，被抽中的概率也更高。）

### 第三步：局部化采样，避免对比学习退化

对比学习需要负样本，但相邻 patch 往往语义高度相似。若把它们互相当作负例，本应靠近的向量会被错误推远。为此作者做了两层限制：对每张图在 $k\times k$ 窗口内只采一个 patch，一张含 $N$ 个 patch 的图实际只有 $N/(k\times k)$ 个进入对比损失；同一 batch 内共享同一标签的 patch 只随机保留一个。

（一片草地里相邻的两个 patch 都表示 grass，不应被当作彼此的负例。）

### 第四步：把标签映射到 LLM 空间，并做双向对比对齐

采样得到的词 $w_i$ 分词后有 $M$ 个 token，送入 **LLM 自身的嵌入层** $\Psi$ 并取平均，得到目标向量：

$$
t_i=\frac{1}{M}\sum_{k=1}^{M}\Psi(w_i^k).
$$

这一步是全篇的关键设计：目标点定义在 LLM 原生嵌入空间，而不是 CLIP 文本空间。若目标取自 CLIP，adapter 只是从"一种错位"挪到"另一种错位"，鸿沟依旧存在。随后用双向 InfoNCE 拉近 adapter 输出 $x_v^i$ 与对应目标 $t_i$：

$$
\mathcal{L}_a=-\frac{1}{2N}\sum_{i=1}^{N}\left[
\log\frac{\exp(\phi(x_v^i,t_i)/\tau)}{\sum_{j=1}^{N}\exp(\phi(x_v^i,t_j)/\tau)}
+\log\frac{\exp(\phi(t_i,x_v^i)/\tau)}{\sum_{j=1}^{N}\exp(\phi(t_i,x_v^j)/\tau)}
\right],
$$

其中 $\phi(x,t)=\frac{x}{\lVert x\rVert_2}\cdot\frac{t}{\lVert t\rVert_2}$ 是余弦相似度，$\tau$ 是温度。第一项要求 visual token 在自己的词向量集合中找回对应项；第二项要求词向量在 visual token 集合中找回对应项。双向的意义在于防止塌缩——若只做单向，最优解可能是所有 patch 都靠近同一个高频词，反过来约束可以抑制这种退化解。

（蓝莓 patch 的 visual token 应靠近 blueberry 与 fruit，而不是 bluejeans。）

### 第五步：与自回归生成损失联合训练

生成侧保持原样，仍是最小化下一个 token 的负对数似然：

$$
\mathcal{L}_g=-\frac{1}{B}\sum_{i=1}^{B}\log p_\theta\left(x^{(i)}\mid V_i,P,x_{\lt i}\right),
$$

$B$ 是 batch size，$P$ 是指令提示，$x_{\lt i}$ 是前序 token。最终目标为两者加权和：

$$
\mathcal{L}=\mathcal{L}_g+\lambda\mathcal{L}_a .
$$

$\lambda$ 平衡两项权重。两种损失的分工可以简明地区分为：

| 损失 | 约束对象 | 监督信号来源 | 学习目的 |
|---|---|---|---|
| $\mathcal{L}_g$ | 整张图与指令 → 下一个文本 token | 图文对自带的 caption | 保持语言生成能力 |
| $\mathcal{L}_a$ | 单个 visual token ↔ 目标词嵌入 | CLIP 文本编码器 + LLM 嵌入层 | 让 visual token 落到正确语义位置 |

关键配置：候选标签数 $n=10$，采样窗口 $2\times2$，词表约 400 万词，训练 1 epoch、AdamW、余弦退火，与 LLaVA 超参一致。阶段一额外开销只有 10–20 分钟。需要留意的是论文内部有一处表述不一致：第 4.1 节写"zero temperature（$\tau=0$）"，而式（11）下方又称 $\tau$ 是可学习温度参数，复现时应以代码为准；$\lambda$ 的具体取值正文也未给出。

## 五、小结

SEA 解决：传统 adapter 只接受图像级 caption 的弱监督，导致 visual token 语义失真、与 LLM 嵌入空间之间存在表征鸿沟，小模型上还会挤占语言能力。

本文先用"**CLIP 语义标签 + 相似度加权采样 + 局部化采样**"得到 token 级监督信号，再用"**LLM 原生嵌入目标 + 双向对比损失 + 生成损失联合**"训练 adapter；只改预训练第一阶段的损失，不动数据、不改结构、不增推理开销。

效果上，在 LLaVA-1.5 同数据同配置的对照中，VQAv2 78.8 → 79.1、GQA 62.0 → 63.2、POPE 86.5 → 87.6、VizWiz 45.7 → 48.8；细粒度任务提升更明显（CapsBench +2.7%、Stanford Dogs +3.9%、OCRBench +5.3%、MMMU +11.4%）；Gemma-2B 上平均提升 7.61%，说明它能补上小模型最缺的那部分对齐。两个附加结论也值得记：第二阶段解冻视觉编码器不但不破坏已建立的对齐，反而继续涨点；把 CLIP 抽出的语义标签直接迁移到 DINOv2 上训练，同样有效，说明这套监督不绑定特定视觉编码器。

从研究脉络看，SEA 属于**预训练期的表征对齐**，与 Re-Align 的**后训练偏好对齐**恰好构成前后两个阶段的对照：一个让视觉 token 先落到正确位置，一个让模型在这个位置上更愿意说符合图像的话。
