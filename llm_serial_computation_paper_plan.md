# LLM 串行计算结构的行为识别：论文架构与实验设计方案

> 暂定题目：**Disentangling Computational Work and Serial Depth in Large Language Models**
>
> 备选题目：**Probing the Effective Serial Computation of Large Language Models**
>
> 核心定位：将 LLM 视为一个未知计算系统，通过受控计算任务、关键路径干预与可选的内部表征分析，研究模型如何处理“可并行工作量（work）”与“不可并行串行深度（serial depth）”。

---

# 1. 论文核心问题

本文不把自然语言 Chain-of-Thought（CoT）视为模型内部推理过程的忠实记录，也不以“CoT token 越多模型越强”为主要研究对象。

真正的问题是：

> **当两个任务具有相同的总计算工作量，但不同的串行依赖深度时，LLM 的表现是否系统不同？如果人为缩短任务的关键依赖路径，模型性能是否会因果性恢复？**

换言之，我们研究的是：

$$
\text{Task computational structure}
\rightarrow
\text{LLM behavior}
$$

并尝试通过外部行为和干预反推出模型的有效计算组织。

---

# 2. 核心动机

Transformer 在一次前向传播中具有高度并行的计算结构，但自回归解码天然提供额外的串行计算轮次。

因此，一个自然问题是：

> **LLM 是否能够在一次前向传播中吸收大量可并行计算，但在面对更长的不可并行依赖链时出现系统性瓶颈？**

现有研究已经表明：

- CoT 可以增强 Transformer 的串行计算能力；
- 推理预算和任务深度之间存在关系；
- 某些推理任务具有非平凡的 token complexity。

但很多已有实验中，任务规模、总工作量和串行深度往往同时增长，例如：

$$
W \uparrow,\qquad D \uparrow
$$

因此难以回答：

> 模型失败究竟是因为“需要做更多计算”，还是因为“必须做更深的串行计算”？

本文的核心贡献是将这两个变量**正交解耦（orthogonal disentanglement）**。

---

# 3. 计算任务形式化

将一个任务表示为计算有向无环图：

$$
G=(V,E)
$$

其中：

- $V$：primitive operations；
- $E$：操作之间的依赖关系。

定义三个量。

## 3.1 总工作量 Work

$$
W(G)=|V|
$$

表示完成任务所需的 primitive operations 数量。

---

## 3.2 串行深度 Serial Depth

$$
D(G)=\max_{p\in Paths(G)} |p|
$$

即计算图中的最长依赖路径，也可理解为 critical-path length。

$D$ 描述的是无法被并行化的串行计算需求。

---

## 3.3 工作记忆需求 Memory

记：

$$
M(G)
$$

为计算过程中最大同时活跃的中间状态数量。

第一篇论文中不重点研究 $M$，尽量将其控制在相近范围，避免把 memory requirement 与 serial depth 混在一起。

---

# 4. 研究问题（Research Questions）

## RQ1：总工作量与串行深度哪个更影响 LLM 的正确率？

固定 $W$，改变 $D$：

$$
W=\text{constant},\qquad D\uparrow
$$

观察：

$$
Acc(W,D)
$$

是否显著下降。

再固定 $D$，改变 $W$：

$$
D=\text{constant},\qquad W\uparrow
$$

比较二者对模型性能的影响。

---

## RQ2：串行深度是否具有独立于总工作量的预测作用？

拟合：

$$
Acc = f(W,D)
$$

或者对错误概率建模：

$$
\text{logit}(P(\text{error}))
=
\beta_0+\beta_W W+\beta_D D+\beta_{WD}WD
$$

核心观察：

$$
|\beta_D| \overset{?}{>} |\beta_W|
$$

更重要的是检查：

$$
\beta_D \neq 0
$$

即在控制总工作量后，串行深度是否仍显著解释模型失败。

---

## RQ3：人为缩短关键依赖路径，是否会因果性恢复性能？

对于原始链：

$$
s_0\rightarrow s_1\rightarrow\dots\rightarrow s_D
$$

向模型直接提供真实中间状态 $s_k$：

$$
s_k \rightarrow s_{k+1}\rightarrow\dots\rightarrow s_D
$$

使剩余深度变为：

$$
D_{\text{remain}}=D-k
$$

观察：

$$
Acc(D_{\text{remain}})
$$

是否随剩余关键路径缩短而系统提升。

这是全文最重要的**因果干预实验**。

---

## RQ4：额外外部迭代计算是否主要帮助高串行深度任务？

比较：

1. Direct Answer；
2. Structured Scratchpad；
3. Free-form CoT（仅作为对照）。

如果外部迭代计算主要改善高 $D$ 而不是高 $W$ 任务，则支持：

> 模型受限于串行计算，而额外迭代步骤可以缓解该瓶颈。

注意：

**本文不将 CoT 文本内容解释为模型内部真实推理轨迹。**

---

# 5. 核心假设

## H1：Serial-Depth Bottleneck

在总工作量相同的情况下：

$$
D_1 < D_2
\Rightarrow
Acc(W,D_1)>Acc(W,D_2)
$$

尤其在 $D$ 超过某一阈值后，准确率会明显下降。

---

## H2：Parallel-Work Absorption

固定串行深度：

$$
D=\text{constant}
$$

增加总工作量 $W$ 时，模型性能下降幅度小于增加 $D$ 时的下降幅度。

即：

$$
\left|
\frac{\partial Acc}{\partial D}
\right|
>
\left|
\frac{\partial Acc}{\partial W}
\right|
$$

至少在一部分任务区域成立。

---

## H3：Critical-Path Causality

对于同一个原始任务，如果直接提供正确的中间状态、缩短剩余 critical path，则性能应恢复：

$$
D_{\text{remain}}\downarrow
\Rightarrow
Acc\uparrow
$$

并且该恢复不应仅由 prompt 长度或额外信息量解释。

---

## H4：External-Compute Rescue

对高 $D$ 任务提供结构化外部迭代计算后：

$$
Acc_{\text{iterative}}
>
Acc_{\text{direct}}
$$

而对低 $D$ 任务提升较小。

---

# 6. 主实验任务设计

第一篇论文建议只使用 **2 个 synthetic task families**，确保控制干净。

---

# 6.1 Task Family A：Random State Transition

构造随机有限状态系统：

$$
f:S\rightarrow S
$$

其中：

$$
S=\{A,B,C,\dots\}
$$

随机生成映射，例如：

```text
A -> F
B -> H
C -> A
D -> G
E -> C
F -> B
...
```

给定初始状态：

$$
s_0
$$

要求计算：

$$
s_D=f^{(D)}(s_0)
$$

## 优点

- 中间状态完全可知；
- 无预训练记忆；
- 不需要复杂数学；
- 每一步只依赖当前状态；
- memory requirement 接近 $O(1)$；
- 非常适合做 checkpoint intervention。

## 主要操纵变量

串行深度：

$$
D\in\{2,4,8,16,24,32\}
$$

总映射表规模：

$$
W_{\text{input}}
$$

可以独立控制。

---

# 6.2 Task Family B：Random Computation DAG

生成随机 primitive operator：

$$
g(a,b)
$$

使用随机查表定义，避免结合律、交换律等 shortcut。

例如：

```text
g(A,B)=F
g(A,C)=D
g(B,D)=H
...
```

构建计算 DAG。

目标：在保持 $W$ 相同的情况下，生成不同 critical-path depth 的 DAG。

例如固定：

$$
W=31
$$

生成：

- Balanced tree：$D\approx5$
- Semi-balanced tree：$D\approx8$
- Irregular DAG：$D\approx12$
- Chain-like DAG：$D\approx16$
- Near-chain：$D\approx24$
- Chain：$D=31$

## 关键点

总 operation 数：

$$
W=31
$$

保持完全一致。

只系统改变：

$$
D
$$

这是论文最重要的实验控制。

---

# 7. 实验矩阵

## 7.1 第一阶段 Pilot

模型：

- Qwen3.5-9B

推理方式：

- Direct Answer Only

每个条件样本数：

$$
N=100
$$

先跑：

$$
W=31
$$

以及：

$$
D\in\{5,8,12,16,24,31\}
$$

总样本：

$$
600
$$

### Pilot 成功标准

如果观察到明显趋势：

$$
D\uparrow
\Rightarrow
Acc\downarrow
$$

则进入正式实验。

如果完全没有结构性趋势，优先检查：

1. 任务是否存在 shortcut；
2. 模型是否根本不会该任务；
3. 模型是否在最低深度就失败；
4. tokenization 是否引入额外难度；
5. task generator 是否有 bug。

---

# 7.2 正式 Work–Depth Grid

建议：

$$
W\in\{15,31,63\}
$$

$$
D\in\{3,5,8,12,16,24,31\}
$$

但只保留满足：

$$
D\le W
$$

的合法组合。

每格：

$$
N=200
$$

如果算力紧张，可以先用：

$$
N=100
$$

做完整 grid，再对关键区域补到 300。

---

# 8. Prompt 设计

必须严格避免 prompt 本身暗示算法步骤。

## Direct Answer Prompt

```text
You are given a state-transition system.

Transitions:
A -> F
B -> H
C -> A
...

Starting state: C
Apply the transition rule exactly 8 times.

Return only the final state.
```

不要求解释过程。

---

# 9. 关键干预实验：Checkpoint Intervention

这是论文第二核心实验。

对于：

$$
s_0\rightarrow s_1\rightarrow\dots\rightarrow s_{32}
$$

分别构造：

- 无 checkpoint；
- 提供 $s_8$；
- 提供 $s_{16}$；
- 提供 $s_{24}$。

例如：

```text
The process has already been executed for 16 steps.

The correct state after step 16 is: G

Continue from G for the remaining 16 steps.

Return only the final state.
```

于是：

$$
D_{\text{remain}}
\in
\{32,24,16,8\}
$$

## 预期结果

如果串行深度确实是瓶颈：

$$
Acc(32)
<
Acc(24)
<
Acc(16)
<
Acc(8)
$$

形成清晰的 dose-response curve。

---

# 10. 关键控制：错误 Checkpoint

对一部分样本故意提供错误中间状态。

例如真实：

$$
s_{16}=G
$$

但告诉模型：

$$
s_{16}=H
$$

观察最终输出。

目的不是主结果，而是验证：

> checkpoint 是否真正改变了后续计算轨迹。

可记录：

1. Follow wrong checkpoint；
2. Correct checkpoint autonomously；
3. Fail / inconsistent。

这可以作为附加分析。

---

# 11. External Compute 实验

不应只使用 free-form CoT。

建议三个条件。

## 11.1 Direct

只输出最终答案。

---

## 11.2 Structured Scratchpad

要求每一步只输出状态：

```text
Step 1: F
Step 2: B
Step 3: D
...
Final: H
```

优点：

- 中间状态机器可验证；
- 避免自然语言冗余；
- 不依赖 CoT faithfulness。

---

## 11.3 Free-form CoT

只作为参考条件。

目的：

> 比较结构化外部迭代计算与自然语言 CoT 是否都能缓解高 $D$ 瓶颈。

不要分析 CoT 文本为“内部思维”。

---

# 12. 可选内部表征实验

这一部分不是首轮必须完成。

如果行为实验和干预实验结果明显，再做。

---

## 12.1 Layer-wise Linear Probe

对于每个 ground-truth intermediate state：

$$
s_1,s_2,\dots,s_D
$$

提取：

$$
h^{(1)},h^{(2)},\dots,h^{(L)}
$$

训练简单 linear probe：

$$
P_l(s_k\mid h^{(l)})
$$

研究：

> 第 $k$ 个算法状态在第几层开始可解码？

---

## 12.2 期望现象

可能出现：

$$
s_1 \rightarrow \text{early layers}
$$

$$
s_2 \rightarrow \text{middle layers}
$$

$$
s_3 \rightarrow \text{later layers}
$$

若存在这种推进，说明 layer axis 可能与 computation progression 有关系。

注意只能声称：

> intermediate state information becomes decodable

不能直接声称模型使用了该状态。

---

## 12.3 Activation Patching

如果 probe 显示明显结构，可以进一步做：

- Run A；
- Run B；
- 在 layer $l$ 替换隐藏表示；
- 看最终输出是否沿另一条 computation trajectory 改变。

这才是更强的内部因果证据。

第一篇小论文可以把这一部分放 Supplement 或 Future Work。

---

# 13. 核心指标

## 13.1 Exact Match Accuracy

$$
Acc(W,D)
$$

synthetic task 答案空间离散，直接 exact match。

---

## 13.2 Error Rate

$$
Err=1-Acc
$$

用于 logistic regression。

---

## 13.3 Critical Depth

定义：

$$
D_{90}
=
\max\{D:Acc(D)\ge0.9\}
$$

也可以定义：

$$
D_{50}
$$

作为模型在某任务族上的行为性“有效隐式串行容量”指标。

注意：

这不是模型真实 architecture depth。

---

## 13.4 Intervention Gain

定义：

$$
\Delta Acc(k)
=
Acc(\text{checkpoint at }k)
-
Acc(\text{no checkpoint})
$$

研究：

$$
\Delta Acc
$$

是否随被移除的 serial depth 增长。

---

## 13.5 External Compute Gain

$$
\Delta_{\text{ext}}
=
Acc_{\text{structured}}
-
Acc_{\text{direct}}
$$

重点考察：

$$
\Delta_{\text{ext}}(D)
$$

是否随 $D$ 增长。

---

# 14. 统计分析

推荐使用 logistic regression：

$$
\text{logit}(P(\text{correct}))
=
\beta_0
+\beta_W\log W
+\beta_D D
+\beta_{WD}\log W\cdot D
$$

如果不同模型共同分析，可以加入 model fixed effects。

或者：

$$
\text{logit}(P(\text{correct}))
=
\beta_0
+\beta_DD
+\beta_WW
+\beta_M\text{Model}
$$

主要报告：

- coefficient；
- confidence interval；
- odds ratio；
- likelihood-ratio test；
- bootstrap confidence intervals。

不要只依赖简单 Pearson correlation。

---

# 15. 必须做的控制实验

## 15.1 Prompt Length Control

高 $D$ 条件不能天然拥有更长 prompt。

尽量保持：

$$
|\text{prompt tokens}|
$$

相同或作为 covariate 控制。

---

## 15.2 Answer Space Control

所有条件保持相同状态空间大小：

$$
|S|=\text{constant}
$$

---

## 15.3 Operator Distribution Control

确保不同 $D$ 条件使用相同 primitive operator 分布。

---

## 15.4 Randomization

每个样本随机：

- state names；
- transition tables；
- graph structure；
- initial values。

避免模型利用表面模式。

---

## 15.5 Shortcut Test

设计 adversarial instances，确保：

- 无结合律；
- 无交换律；
- 无固定常数；
- 无简单周期；
- 无位置模式。

---

# 16. 模型选择

第一阶段：

- Qwen3.5-9B

第二阶段至少加入：

- 一个同规模非 reasoning-oriented 模型；
- 一个不同模型族的 7B–14B 模型。

目标不是做规模 benchmark，而是验证现象是否跨模型存在。

建议最终控制在：

$$
2\sim3
$$

个模型。

---

# 17. 算力规划

当前单张 RTX 4090 足够完成第一阶段。

## Pilot

约：

$$
600\sim1500
$$

次 inference。

完全可行。

## 正式实验

约：

$$
5000\sim15000
$$

次 inference。

由于答案很短、synthetic prompt 较小，成本仍明显低于训练实验。

不需要：

- CPT；
- SFT；
- RL；
- Megatron；
- 多卡训练。

---

# 18. 论文结构

# 1 Introduction

核心叙事：

1. LLM 推理通常被用“题目难度”和“更多 reasoning tokens”描述；
2. 但计算任务存在两个不同属性：
   - total work；
   - critical-path depth；
3. 现有研究常将二者混合；
4. 本文正交控制二者；
5. 通过路径干预研究 serial-depth bottleneck 是否具有因果性。

## Introduction 最后一段贡献

建议写成三点：

1. 提出一个将 task work 与 serial depth 解耦的受控 benchmark；
2. 证明 serial depth 在控制 total work 后仍系统预测 LLM failure；
3. 通过 intermediate-state intervention，证明缩短 critical path 可以因果性恢复模型性能。

如果 External Compute 结果强，再加第四点。

---

# 2 Related Work

## 2.1 Transformer Computational Expressivity

- Transformer computational power；
- circuit complexity；
- serial vs parallel computation。

## 2.2 Chain-of-Thought as Additional Computation

只作为背景：

- CoT 增加外部计算轮次；
- 不假设 CoT faithful。

## 2.3 Reasoning Depth and Token Complexity

讨论：

- logical depth；
- serial depth；
- token lower bounds。

## 2.4 Behavioral and Mechanistic Interpretability

定位本文：

> 不直接寻找单个 neuron/head，而通过受控任务和干预识别模型的计算组织。

---

# 3 Problem Setup

形式化：

$$
G=(V,E)
$$

定义：

$$
W(G),\quad D(G),\quad M(G)
$$

给出核心假设。

---

# 4 Controlled Computational Tasks

介绍：

- State Transition；
- Random Computation DAG。

重点解释如何保证：

$$
W\perp D
$$

以及如何消除 shortcut。

---

# 5 Experiments

## 5.1 Work–Depth Disentanglement

主图：

$$
Acc(W,D)
$$

heatmap。

---

## 5.2 Critical-Path Intervention

主图：

$$
Acc
\text{ vs. }
D_{\text{remaining}}
$$

---

## 5.3 External Compute Rescue

比较：

- direct；
- structured scratchpad；
- free-form CoT。

---

## 5.4 Cross-model Analysis

验证普适性。

---

# 6 Analysis

讨论：

- 是否存在 effective serial-depth threshold；
- work 与 depth 哪个解释力更强；
- external iterative compute 如何补偿；
- 对 Transformer computation 的含义。

避免过度声称：

不能说：

> 模型内部真的执行了 $D$ 步算法。

只能说：

> 模型行为对 task serial depth 表现出系统性、可干预的敏感性。

---

# 7 Limitations

必须明确：

1. Task-level depth 不等于 model-internal computational depth；
2. Synthetic task 与真实 reasoning task 之间存在 gap；
3. Behavioral evidence 不等于 mechanistic proof；
4. Structured scratchpad 同时提供 compute 与 external memory；
5. Transformer 内部可能存在 shortcut algorithm。

这些 limitation 主动写出来反而会使论文更可信。

---

# 8 Conclusion

核心结论：

> LLM failure cannot be explained solely by total computational work. The serial organization of computation independently matters, and causally shortening the task's critical path restores performance.

---

# 19. 核心 Figure 规划

## Figure 1：概念图

左：

```text
Same Work, Shallow
```

右：

```text
Same Work, Deep
```

强调：

$$
W_1=W_2,\qquad D_1\ll D_2
$$

---

## Figure 2：Work–Depth Heatmap

横轴：

$$
D
$$

纵轴：

$$
W
$$

颜色：

$$
Accuracy
$$

这是全文最重要的行为图。

---

## Figure 3：Depth vs Work Effect

两条曲线：

- fixed $W$, vary $D$；
- fixed $D$, vary $W$。

直观看出：

$$
D
$$

和

$$
W
$$

的影响差异。

---

## Figure 4：Critical-Path Intervention

横轴：

$$
D_{\text{remaining}}
$$

纵轴：

$$
Accuracy
$$

如果出现平滑 dose-response，非常漂亮。

---

## Figure 5：External Compute Rescue

不同：

- direct；
- structured；
- CoT；

在不同 $D$ 下的 accuracy。

---

# 20. 最小可行论文（MVP）

如果时间紧，只完成：

### Task

Random State Transition + Random DAG

### Models

2 个模型

### Experiments

1. $W/D$ disentanglement；
2. checkpoint intervention；
3. structured scratchpad rescue。

### Figures

4 张。

完全可以形成一篇完整 short paper / workshop paper，并具备继续扩展到 conference full paper 的空间。

---

# 21. 项目停止标准

这是非常重要的。

## Stop Criterion 1

如果在合理难度区域：

$$
D\uparrow
$$

不产生任何稳定性能变化，

则停止。

---

## Stop Criterion 2

如果高 $D$ 与低 $D$ 的差异完全能被：

- prompt length；
- token count；
- state-space size；

解释，

停止。

---

## Stop Criterion 3

如果 checkpoint intervention 不恢复性能：

$$
D_{\text{remain}}\downarrow
\not\Rightarrow
Acc\uparrow
$$

则“serial-depth bottleneck”的因果叙事非常弱，需要重新考虑项目。

---

## Stop Criterion 4

如果只在单一模型上出现，换模型后完全消失，则不应立刻写成普遍结论。

---

# 22. 推荐代码结构

```text
serial-depth-probe/
├── configs/
│   ├── pilot.yaml
│   ├── main_grid.yaml
│   └── intervention.yaml
│
├── src/
│   ├── tasks/
│   │   ├── state_transition.py
│   │   ├── random_dag.py
│   │   └── validators.py
│   │
│   ├── generation/
│   │   ├── generate_dataset.py
│   │   └── graph_utils.py
│   │
│   ├── inference/
│   │   ├── vllm_runner.py
│   │   ├── prompts.py
│   │   └── parsers.py
│   │
│   ├── interventions/
│   │   ├── checkpoint.py
│   │   └── wrong_checkpoint.py
│   │
│   ├── analysis/
│   │   ├── metrics.py
│   │   ├── regression.py
│   │   └── plotting.py
│   │
│   └── utils/
│
├── data/
│   ├── pilot/
│   ├── main/
│   └── intervention/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── scripts/
│   ├── run_pilot.sh
│   ├── run_grid.sh
│   └── run_intervention.sh
│
└── README.md
```

---

# 23. 推荐实施顺序

## Phase 1：一天内完成任务生成器

实现：

- random state transition；
- exact ground-truth executor；
- 自动 validator。

要求：

$$
100\%
$$

程序可验证。

---

## Phase 2：最小 Pilot

只跑：

$$
W=31
$$

$$
D=\{5,8,12,16,24,31\}
$$

模型：

Qwen3.5-9B。

若没有趋势，先停。

---

## Phase 3：Work–Depth Grid

跑二维实验。

得到：

$$
Acc(W,D)
$$

heatmap。

---

## Phase 4：Checkpoint Intervention

如果 Phase 3 有明显 serial-depth effect，立即做。

这是决定论文是否成立的最重要实验。

---

## Phase 5：第二模型复现

只有前面成立，再扩模型。

---

## Phase 6：External Compute

structured scratchpad 优先。

CoT 仅作为对照。

---

## Phase 7：Internal Probe（可选）

如果前六步结果很强，再做。

避免一开始陷入 mechanistic interpretability 的巨大工程量。

---

# 24. 最终论文核心叙事

整篇论文只讲一件事：

> **总计算量相同，并不意味着 LLM 面临相同难度。计算依赖结构本身会影响模型性能。**

进一步：

> **当关键依赖路径更长时，LLM 更容易失败；而人为切断或缩短这条路径，可以系统性恢复性能。**

因此我们得到的不是：

> “模型说了多少 reasoning tokens。”

而是：

> **LLM 对任务计算结构表现出可测量、可干预的行为响应。**

这使本文具有一种“计算可解释性（computational interpretability）”的味道：

$$
\text{controlled computational stimulus}
\rightarrow
\text{behavior}
\rightarrow
\text{intervention}
\rightarrow
\text{inference about computation}
$$

---

# 25. 一句话版论文贡献

> **We disentangle total computational work from serial dependency depth and show that LLM performance is specifically sensitive to the latter; causally shortening the task's critical path systematically restores performance.**

中文：

> **本文将总计算工作量与串行依赖深度解耦，发现 LLM 的性能对后者存在独立而系统的敏感性；进一步通过关键路径干预证明，直接缩短任务的串行依赖链可以因果性恢复模型性能。**

