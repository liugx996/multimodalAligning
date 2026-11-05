# MoD 阅读报告

**Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models**

（注意力启发的大型视觉语言模型自适应混合解码）

> **出处**：Findings of ACL 2025
>
> **代码**：<https://github.com/xlchen0205/MoD>

**核心概括**：MoD 面向 LVLM 的跨模态幻觉问题，提出一种训练无关的自适应解码方法。它先判断模型对图像 token 的注意力是否正确，再决定"放大关键视觉信息"还是"抑制误导性视觉信息"，避免传统解码方法对所有样本采用同一种处理。

## 一、文章解决了什么下游任务？

**在推理阶段减少 LVLM 的物体与属性幻觉。**&#8203;

MoD 不改变模型参数，而在每个解码步判断当前视觉注意力是否可靠，并据此调整输出分布。

## 二、传统做法都是怎么解决的？

传统对比解码方法通常固定地抑制某一类偏差：VCD、M3ID 对比原始与扰动图像以削弱语言先验；AvisC 一概降低高注意力视觉 token 的影响。它们的问题是**不判断注意力是否正确**：有些高注意力 token 正是正确证据，统一抑制会损失有效信息；有些错误注意力又需要更强的对比修正。

## 三、本文发现了什么此前解决不好的问题？

注意力存在两种状态：

- **注意力正确**：原始视觉 token 与模型实际关注的图像区域所支持的回答一致，此时应放大关键视觉信息。
- **注意力错误**：两者输出分布明显不一致，说明模型关注了错误区域，此时应抑制由错误注意力产生的分布。

此前方法缺少一个**轻量**的"注意力正确性判据"，因此无法自适应选择互补解码或对比解码。

## 四、本文具体怎么做？

### 第一步：找出模型真正关注的图像 token

对所有 Transformer 层、所有注意力头取平均，使用输入序列最后一个 token 对图像 token 的注意力 $A^{I}$；再取注意力最高的 top- $lambda$ 比例，得到被关注图像 token 索引 $\mathrm{IDX}^{I}_{\mathrm{att}}$ .把其余图像 token 置零，得到 $v_{\mathrm{att}}$ .（问"沙发上的动物"时，只保留猫所在 patch，遮住窗帘。）

### 第二步：计算两路输出分布的一致性

分别用完整视觉 token $v$ 和被关注 token $v_{\mathrm{att}}$ 生成第 $t$ 步输出分布，并用 JS 散度衡量差异：

$$
d(v, v_{\mathrm{att}}) = D_{\mathrm{JS}}( p_\theta(y_t \mid v, x, y_{\lt t}) \parallel p_\theta(y_t \mid v_{\mathrm{att}}, x, y_{\lt t}) )
$$

$d$ 小表示"看全部图"和"只看关注区域"结论一致，注意力可靠； $d$ 大表示模型关注区域不能支持原回答，注意力可能错误。

### 第三步：按阈值自适应选择解码策略

设一致性阈值 $\gamma$。当 $d \le \gamma$ 时采用互补解码，放大正确视觉信息；当 $d \gt \gamma$ 时采用对比解码，抑制错误注意力：

$$
y_t \sim
\begin{cases}
\mathrm{softmax}[\, \mathrm{logit}(v) + \alpha_1 \mathrm{logit}(v_{\mathrm{att}}) \,], & d \le \gamma, \\
\mathrm{softmax}[\, (1+\alpha_2)\mathrm{logit}(v) - \alpha_2 \mathrm{logit}(v_{\mathrm{att}}) \,], & d \gt \gamma.
\end{cases}
$$

$\alpha_1$ 是互补强度， $\alpha_2$ 是对比强度。含义：看对了就把两路证据相加；看错了就从原始分布中减去错误关注分支。

### 第四步：限制不合理候选词

仅保留概率达到最大概率 $\beta$ 倍以上的 token（这个 token 的概率 ≥ β × 当前最大概率 token 的概率）：

$$
V_{\mathrm{head}}(y_{\lt t}) = \lbrace\ y_t \in V : p_\theta(y_t \mid v, x, y_{\lt t}) \ge \beta \max_w p_\theta(w \mid v, x, y_{\lt t}) \,\rbrace
$$

这防止对比或互补放大后输出低概率、不合理的词。

| 状态 | 判据 | 使用策略 | 学习目的 | 例子 |
|---|---|---|---|---|
| 注意力正确 | $d \le \gamma$ | 互补解码 | 放大关键视觉证据 | 关注猫区域，放大"cat" |
| 注意力错误 | $d \gt \gamma$ | 对比解码 | 抑制误导信息 | 关注窗帘却答"dog"，减去该分支 |

## 五、小结

MoD 解决：传统对比解码不区分注意力是否正确，统一抑制可能误伤有效视觉信息。

本文先用"**跨层注意力平均 + top- $\lambda$ 掩码**"得到被关注图像 token，再用 **JS 散度**判断注意力一致性，最后自适应选择"互补放大"或"对比抑制"；整个过程不训练模型、不涉及 CoT。
