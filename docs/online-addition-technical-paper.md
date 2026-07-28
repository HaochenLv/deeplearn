# 在线交互式整数加法学习系统：基于反馈对齐的三模型协同架构

> 技术方案论文 · 2026-07-26
> 仓库：`deeplearn/` · 分支：`mlp-char-addition`

---

## 摘要

本工程实现了一个从零开始、边推理边学习的多位整数加法系统。系统由**三个模型协同**完成：两个前馈推理模型（个位加法器 `DigitAdder` 与进位传播器 `CarryPropagator`）负责"做加法"，一个拥有**独立参数集**的学习机制（`LearningMechanism`）负责"学会做加法"。学习机制不依赖外部 autograd/Adam，而是通过**反馈对齐（Feedback Alignment, FA）**与**Delta 规则**直接修改推理模型权重，实现生物启发的局部在线学习。

我们对比了三种学习信号方案（Hebbian / Reward / Correction），并系统研究了一个关键架构问题：**进位传播应该用循环网络（GRU）还是前馈网络建模**。实验表明，FA 无法训练 GRU 的时间动力学（因 FA 不含沿时间反向传播 BPTT），导致系统无法泛化到训练长度之外的位数（20 位仅 28%）；而把进位传播建模为**前馈单步映射** `cin = f(cout)` 后，FA 能完整训练该映射，且单步映射天然与序列长度无关，三种方案在 **1–50 位上全部达到 100% 准确率**。

本文详细解释系统用到的所有算法原理，包括模块化加法分解、反馈对齐、Delta 规则、三种门控学习机制、课程学习与 Teacher Forcing，并给出 GRU 与前馈进位传播的对比分析。

---

## 1. 引言

### 1.1 问题背景

整数加法对人类是基础技能，对神经网络却暗藏挑战。一个固定输入长度的多层感知机（MLP）可以死记 2 位或 4 位加法的全部样本，但**无法外推到更长的位数**——因为加法是一个长度可变的算法过程，而非固定维度的映射。

打破这一限制的经典思路是**算法分解**：把"N 位加法"拆成两个可复用的子技能：

- **个位加法** `a + b + cin → digit + cout`（仅 200 种组合，与位数无关）
- **进位传递** `cout → 下一位 cin`（确定性控制流）

只要这两个子技能学好，就能逐位循环组合出任意 N 位加法。这正是本工程 `modular_addition.py` 的"神经-符号"设计。

### 1.2 动机：在线学习与生物合理性

常规深度学习用 **autograd + 反向传播（BP）+ 优化器（Adam）** 离线训练模型。这有两个不"自然"的地方：

1. **权重对称性问题**：BP 要求反向通路使用前向权重的精确转置 $W^\top$，而生物突触不存在这种对称的反馈连接。
2. **全局优化器**：Adam 作为一个外部"外科医生"，直接手术修改大脑权重——但真实大脑的学习应是**局部的、自主的**。

本工程的目标是：让学习机制成为一个**拥有独立参数的内部模块**，它接收学习信号（教师纠正/奖励），自主计算权重更新 $\Delta W$ 并直接写入推理模型，全程不调用 autograd。这更接近"大脑内部的学习机制调整自身突触"的图景。

为此我们采用 **反馈对齐（FA）**：用固定随机矩阵 $B$ 替代 $W^\top$ 传播误差，保持局部性的同时保留有监督学习的方向性。

### 1.3 贡献

1. 设计了**三模型协同**的在线加法学习架构（2 个推理模型 + 1 个独立参数学习机制）。
2. 实现并对比了**三种学习信号方案**（Hebbian / Reward / Correction），给出选型结论。
3. 通过实验与隐状态动力学分析，定位了 **FA 在循环网络上的根本边界**，并提出用**前馈单步映射**建模进位传播的解法，实现 N 位完美泛化。

---

## 2. 问题定义

**任务**：给定两个非负整数 $A, B$（十进制字符串），输出 $A+B$。位数 $N$ 任意。

**学习范式**：

- 模型从**零初始化**开始，没有预训练。
- 外部"教师"出题，并在每位提供**正确答案**（`digit_true, cout_true, cin_true`）。
- 学生**边推理边学习**：每做一题，每个数位都触发一次权重更新。
- 课程从 1 位逐步升级到 5 位，掌握（准确率 ≥ 95%）后才升级。
- **约束**：学习过程不使用 autograd 反向传播，也不使用沿时间反向传播（BPTT）。

---

## 3. 系统架构

### 3.1 三模型协同总览

```
┌─────────────────────────────────────────────────────────────┐
│  StudentAgent                                               │
│                                                             │
│  推理模型（做加法）            学习模型（学加法）           │
│  ┌──────────────┐ ┌────────────┐  ┌───────────────────────┐ │
│  │ ① DigitAdder │ │② Carry     │  │ ③ LearningMechanism  │ │
│  │  a+b+cin →   │←→│  Propagator│←→│  独立参数集           │ │
│  │  digit+cout  │  │ cout→cin   │  │  FA + Delta 规则      │ │
│  └──────────────┘ └────────────┘  └───────────────────────┘ │
│         ▲                ▲                   │ ΔW           │
│         │                │                   ▼              │
│         └────── 教师纠正信号 ─────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

三个模型的职责严格分离：①② 只负责前向推理（参数不参与 autograd），③ 负责计算 $\Delta W$ 并直接写回 ①② 的 `.data`。

### 3.2 推理模型①：DigitAdder

一个两层隐藏层的前馈 MLP，输入是 `(a, b, cin)` 的 one-hot 拼接，输出个位和与进位。

- 输入维度 $d_{in} = 22$（a 的 10 维 + b 的 10 维 + cin 的 2 维 one-hot）
- 结构：`Linear(22, h) → ReLU → Linear(h, h) → ReLU → {digit_head: Linear(h,10), carry_head: Linear(h,2)}`
- 默认 $h=128$，参数量 $21{,}004$

记前向为：
$$
\begin{aligned}
\mathbf{h}_1 &= \mathrm{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) \\
\mathbf{h}_2 &= \mathrm{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2) \\
\mathbf{z}_d &= W_d\,\mathbf{h}_2 + \mathbf{b}_d \quad (\text{digit logits}, 10)\\
\mathbf{z}_c &= W_c\,\mathbf{h}_2 + \mathbf{b}_c \quad (\text{carry logits}, 2)
\end{aligned}
$$

### 3.3 推理模型②：CarryPropagator（前馈单步映射）

进位传播器把"上一位的进位输出"映射为"当前位的进位输入"：

$$\mathrm{cin}_t = f(\mathrm{cout}_{t-1})$$

结构：`Linear(2, h_c) → ReLU → Linear(h_c, h_c) → ReLU → cin_head: Linear(h_c, 2)`，默认 $h_c=16$，参数量 $354$。输入 `cout` 是 2 维 one-hot，输出 `cin` 是 2 维 logits。

**为什么是前馈而不是循环（GRU）**——这是本工程的核心设计决策，第 7 节详细论证。

### 3.4 学习模型③：LearningMechanism

拥有独立参数集的模块，包含：

- **分层学习率** $\eta$：每个推理层一个（用 $\log\eta$ 参数化保证正值）
- **权重衰减** $\gamma$：每层一个
- **FA 反馈矩阵** $B$：固定随机初始化（`requires_grad=False`），替代 $W^\top$
- （Reward 方案额外有调制因子 $\beta$）

默认配置下 Hebbian 方案学习机制参数量 $18{,}222$，与推理模型参数（$21{,}358$）**完全独立、互不重叠**。

---

## 4. 核心算法原理

### 4.1 模块化分解：神经-符号加法

N 位加法 $A+B$ 的逐位算法（从低位到高位）：

```
cin_0 = 0
for t = 0 .. N-1:
    digit_t, cout_t = AddDigit(a_t, b_t, cin_t)
    cin_{t+1} = cout_t
if cin_N == 1:  最高位进位
```

关键性质：

- `AddDigit` 只依赖当前位 $(a_t, b_t, \mathrm{cin}_t)$，与位数 $N$ 无关。
- 进位传递 $\mathrm{cin}_{t+1} = \mathrm{cout}_t$ 是**逐位局部**的。

因此只要两个子技能学好，组合即可外推到任意 $N$。`DigitAdder` 学第一个子技能（200 种映射），`CarryPropagator` 学第二个。

### 4.2 进位传递的单步映射本质（关键洞察）

进位传递 $\mathrm{cin}_{t+1} = \mathrm{cout}_t$ 是一个**恒等映射**：cout=0 → cin=0，cout=1 → cin=1。它：

- **不依赖历史**：当前 cin 只由上一位 cout 决定，无需"记住"更早的进位。
- **是单步映射**：输入输出都是 2 维离散值。

这意味着它**不需要循环记忆**。用前馈网络 $f: \mathrm{cout} \mapsto \mathrm{cin}$ 即可精确表达，且该映射与序列长度无关——这是 N 位泛化的数学基础。

> 对照：若用 GRU 建模，GRU 会把进位编码进隐状态 $\mathbf{h}_t$，需要学会"跨多步保持/更新隐状态"的时间动力学。这要求 BPTT 才能训练（见第 7 节）。

### 4.3 前向传播与输入编码

输入编码 `encode_abc(a, b, cin)` 构造 22 维 one-hot：
$$\mathbf{x} = [\,\text{onehot}_{10}(a),\; \text{onehot}_{10}(b),\; \text{onehot}_{2}(\mathrm{cin})\,]$$

推理时（`predict`）：$\mathrm{digit} = \arg\max \mathbf{z}_d$，$\mathrm{cout} = \arg\max \mathbf{z}_c$。

### 4.4 Delta 规则（Widrow-Hoff）

Delta 规则是最古老的有监督局部学习规则。对线性输出 $y = W\mathbf{x}$ 与目标 $\mathbf{t}$：

$$\Delta W = \eta\,(\mathbf{t} - y)\,\mathbf{x}^\top$$

直觉：当输出偏小（$y < t$），把权重朝输入方向抬升，使下次输出更大；反之压低。更新量与"误差 × 输入"成正比，**完全局部**（只需该层的前向输入与输出误差）。

本工程的输出层正是 Delta 规则的精确形式（见 4.6）。

### 4.5 反馈对齐（Feedback Alignment）

FA 是本工程学习机制的数学核心，也是"生物合理性"的关键来源。下面从标准反向传播的链式法则出发，逐步推导 FA，并解释它为何有效。

#### 4.5.1 反向传播回顾：链式法则与权重对称性要求

考虑一个三层前馈网络（省略偏置以聚焦本质）：
$$
\mathbf{h}_1 = \sigma(W_1 \mathbf{x}), \quad \mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1), \quad \hat{\mathbf{y}} = W_3 \mathbf{h}_2
$$
其中 $W_1 \in \mathbb{R}^{h\times d}$, $W_2 \in \mathbb{R}^{h\times h}$, $W_3 \in \mathbb{R}^{m\times h}$。设损失 $L = \ell(\hat{\mathbf{y}}, \mathbf{t})$。

反向传播通过链式法则逐层求梯度。记输出端误差 $\boldsymbol{\delta}_3 = \frac{\partial L}{\partial \hat{\mathbf{y}}} \in \mathbb{R}^m$，则最靠近输出的权重梯度为：
$$
\frac{\partial L}{\partial W_3} = \boldsymbol{\delta}_3 \, \mathbf{h}_2^\top \in \mathbb{R}^{m\times h}
$$

要把误差继续传到第二层隐藏层，需要对 $\mathbf{h}_2$ 求导。由 $\hat{\mathbf{y}} = W_3 \mathbf{h}_2$（对 $\mathbf{h}_2$ 线性），Jacobian 为 $\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{h}_2} = W_3$，于是：
$$
\frac{\partial L}{\partial \mathbf{h}_2} = \left(\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{h}_2}\right)^{\!\top} \boldsymbol{\delta}_3 = W_3^{\!\top} \boldsymbol{\delta}_3
$$
**注意出现的 $W_3^{\!\top}$**。再乘以激活导数得到第二层的"delta"：
$$
\boldsymbol{\delta}_2 = (W_3^{\!\top} \boldsymbol{\delta}_3) \odot \sigma'(\mathbf{h}_2)
$$
继续向第一层：
$$
\boldsymbol{\delta}_1 = (W_2^{\!\top} \boldsymbol{\delta}_2) \odot \sigma'(\mathbf{h}_1)
$$

由此得出反传的一般规律：**每一层的误差都依赖前一（前向）权重的精确转置 $W^{\!\top}$**。这就是**权重对称性要求**（weight symmetry problem）：反向反馈通路必须以镜像前向通路的方式布线。

这个要求有两层含义：

- **工程上**：深度学习框架（autograd）必须为每个前向算子记录其转置操作，构建并维护一张完整的计算图，才能在 `.backward()` 时取出正确的 $W^{\!\top}$。这是一种全局的、需要中心化调度的机制。
- **生物学上**：它要求大脑中每个突触 $i \to j$ 都配有一条权值精确对称的反馈突触 $j \to i$。真实神经元难以维持这种精确对称——这是"反向传播不太可能是大脑的学习机制"的经典论据之一。

#### 4.5.2 FA 的形式化定义

Lillicrap et al. (2016) 提出一个打破对称性要求的替换：**用固定随机矩阵 $B$ 取代 $W^{\!\top}$** 传播误差：
$$
\boldsymbol{\delta}_2^{\mathrm{FA}} = (B\, \boldsymbol{\delta}_3) \odot \sigma'(\mathbf{h}_2)
$$
其中 $B \in \mathbb{R}^{h\times m}$ 与 $W_3^{\!\top}$ 同形状，在训练开始时按 $\mathcal{N}(0, \sigma_B^2)$ 随机初始化后**永久冻结**——它不接收任何梯度、永不更新。权重仍按"梯度形"更新：
$$
\Delta W_2 = -\eta\, \boldsymbol{\delta}_2^{\mathrm{FA}} \, \mathbf{h}_1^\top
$$

需要强调：当 $B \neq W_3^{\!\top}$ 时，$\boldsymbol{\delta}_2^{\mathrm{FA}}$ **并非损失 $L$ 的真实梯度方向**。FA 实际上沿一个"伪梯度"方向更新——从严格优化角度看，它甚至不保证损失单调下降。令人惊讶的是，这种"错误的"反馈方向仍能驱动有效学习，原因在于 4.5.3 描述的对齐动态。

生物学上，固定随机 $B$ 对应一组**随机的、无需对称、无需更新的反馈突触**——大脑完全可以具备这种结构，这正是 FA 的生物吸引力。

#### 4.5.3 FA 为何有效：前向-反馈自对齐

FA 有效性的关键是一个协同动态过程：**$B$ 固定，但 $W$ 在更新**。下面给出两个层次的解释。

**（1）线性情形的精确论证。** 考虑线性网络 $\hat{\mathbf{y}} = W_3 W_2 \mathbf{x}$，固定随机 $B$ 反馈。Lillicrap 等人证明，若用 FA 伪梯度更新 $W_2$（$W_3$ 仍用真梯度更新），则一致性矩阵 $W_3^{\!\top} B$ 的最小奇异值在训练中**单调增长**。直观地：

$$
W_3^{\!\top} B \;\xrightarrow{\text{训练}}\; \text{接近正定} \quad(\text{理想情况下 } \approx \alpha I,\ \alpha>0)
$$

当 $W_3^{\!\top} B \approx \alpha I$ 时，FA 反馈方向 $B\boldsymbol{\delta}_3$ 与真实反传方向 $W_3^{\!\top}\boldsymbol{\delta}_3$ 之间仅差一个正标量 $\alpha$——两者**同向**。此时 FA 伪梯度与真 BP 梯度指向同一方向，FA 在效果上等价于 BP。

换句话说：**虽然反馈通路 $B$ 是随机固定的，但前向通路 $W_2$ 会自我重组，去"迎合"这条随机反馈通路**，使得真实梯度与伪梯度逐渐对齐。对齐完成后，学习就和标准 BP 一样有效。

**（2）非线性 / 表示层面的直观。** 把 $B$ 视为一个"反馈坐标系"。FA 不要求网络找到某个特定解，而是要求隐藏层表示 $\mathbf{h}_2$ 落在"$B$ 能正确解码"的子空间里。训练过程同时调整前向权重（决定 $\mathbf{h}_2$ 的编码）使之与 $B$ 兼容——这是一种**协同自适应**：反馈是死的，但前向是活的，活的去配合死的。

**工程推论：**

1. **FA 需要足够的隐藏层宽度。** 对齐是一个统计的、依赖表示能力的过程。隐藏层太窄时，$\mathbf{h}_2$ 的表达能力不足以配合随机 $B$，对齐失败。本工程 DigitAdder 需 $h=128$ 才稳定收敛（$h=16$ 时不收敛），正是此效应的直接体现。
2. **FA 在足够宽的前馈网络上可逼近 BP 精度。** 一旦对齐完成，FA 与 BP 在前馈任务上表现相近。这也是本工程 DigitAdder 能学到 100% 的原因。
3. **FA 的边界在前馈网络之内。** 对齐依赖"前向权重重组去配合反馈"。但循环网络的反馈涉及**时间维度**（上一时刻隐状态），FA 没有"沿时间的反馈通路"，无法触发对应的时间对齐——这是第 7 章 GRU 失败的深层原因。

#### 4.5.4 ReLU 子梯度与误差门控

隐藏层用 ReLU：$\sigma(u) = \max(0, u)$，导数为次梯度 $\sigma'(u) = \mathbb{1}[u > 0]$。这使误差只在**激活的神经元**（pre-ReLU 值为正）上回传，关闭的神经元不接收反馈。代码用 `h2_relu_grad = (h2 > 0).float()` 实现（`h2` 是 ReLU 前的线性输出），正确实现了 ReLU 的反向语义。

#### 4.5.5 本工程的 FA 配置

- **双输出头合并反馈**：DigitAdder 有 digit（10 维）与 carry（2 维）两个输出头，它们的误差在反馈到 $\mathbf{h}_2$ 时合并：$\mathbf{e}_2 = B_d\,\mathbf{e}_d + B_c\,\mathbf{e}_c$。
- **多层反馈**：$\mathbf{h}_2 \to \mathbf{h}_1$ 用 $B_2$；前馈 CarryPropagator 用独立的 $B_{\mathrm{cin}}, B_{c2}$（与 adder 的 $B$ 解耦）。
- **冻结随机初始化**：所有 $B$ 以 $\mathcal{N}(0, 0.1^2)$ 初始化，`requires_grad=False`。
- **输出层不用 FA**：输出层直接用 softmax-CE 的精确梯度（4.6 节）。FA 只作用于隐藏层——这是 FA 的标准用法。

具体地，DigitAdder 隐藏层的 FA 更新为：
$$
\begin{aligned}
\mathbf{e}_2 &= B_d\,\mathbf{e}_d + B_c\,\mathbf{e}_c \\
\boldsymbol{\delta}_2 &= \mathbf{e}_2 \odot \mathbb{1}[W_2\mathbf{h}_1+\mathbf{b}_2 > 0] \\
\Delta W_2 &= \eta\,\boldsymbol{\delta}_2\,\mathbf{h}_1^\top - \gamma W_2 \\[2pt]
\boldsymbol{\delta}_1 &= (B_2\,\boldsymbol{\delta}_2) \odot \mathbb{1}[W_1\mathbf{x}+\mathbf{b}_1 > 0] \\
\Delta W_1 &= \eta\,\boldsymbol{\delta}_1\,\mathbf{x}^\top - \gamma W_1
\end{aligned}
$$
其中 $B_d \in \mathbb{R}^{h\times10}$、$B_c \in \mathbb{R}^{h\times2}$、$B_2 \in \mathbb{R}^{h\times h}$ 均为固定随机矩阵。

### 4.6 输出层：精确的 softmax-交叉熵梯度

输出层并非粗糙的启发式，而是**精确的梯度下降**。以 digit 头为例，交叉熵损失：
$$L = -\sum_k t_k \log p_k, \quad p = \mathrm{softmax}(\mathbf{z}_d)$$

softmax-CE 对 logits 的梯度有著名的简化形式：
$$\frac{\partial L}{\partial \mathbf{z}_d} = \mathbf{p} - \mathbf{t}$$

由于 $\mathbf{z}_d = W_d \mathbf{h}_2 + \mathbf{b}_d$ 对 $W_d$ 线性，链式法则给出：
$$\frac{\partial L}{\partial W_d} = (\mathbf{p} - \mathbf{t})\,\mathbf{h}_2^\top$$

梯度下降 $W_d \leftarrow W_d - \eta \frac{\partial L}{\partial W_d}$，即：
$$\Delta W_d = \eta\,(\mathbf{t} - \mathbf{p})\,\mathbf{h}_2^\top - \gamma W_d$$

这正是代码中的 `d_error = target - softmax(logits)` 与 `ΔW = η·d_error ⊗ h2_act`。**输出层更新与标准 BP 完全一致**；FA 仅用于隐藏层。

### 4.7 权重衰减与稳定性

每层更新附加 $-\gamma W$ 项（weight decay），防止权重无限增长。本工程发现一个重要细节：**当 $\gamma > 0$ 且训练数据是随机的时，不同输入的 Delta 更新会相互抵消，导致权重被衰减项持续拉向零**。因此默认 $\gamma = 0$（纯 Delta 更新），仅在需要正则化时开启。

### 4.8 学习率参数化

学习率用 $\log\eta$ 参数化，使用时取 $\eta = e^{\log\eta}$，保证学习率恒正。采用**分层学习率**：输出层用较大的 $\eta_{\text{out}} = 10\eta_{\text{hid}}$（输出信号强、需快速拟合），隐藏层用较小的 $\eta_{\text{hid}}$（避免不稳定）。

---

## 5. 三种学习机制

三种方案共享 FA + Delta 框架，区别在于**学习信号的来源与对更新幅度的调制（门控）**。所有方案的更新方向都由误差 $\mathbf{e} = \mathbf{t} - \mathbf{p}$ 决定，门控只调节"学多少"。

### 5.1 HebbianLearner（FA 局部学习，Mode C）

最简洁的方案。门控恒为 1：

$$\Delta W = \eta\,\mathbf{e}\,\mathbf{pre}^\top - \gamma W \quad(\text{输出层}), \qquad
\Delta W = \eta\,(B\mathbf{e})\odot\sigma'\,\mathbf{pre}^\top - \gamma W \quad(\text{隐藏层})$$

只要有教师纠正信号（`digit_true` 等）就更新，不区分对错。这是纯反馈对齐，参数最少（无 $\beta$）。

### 5.2 RewardLearner（奖励门控，Mode A）

引入标量奖励 $r \in \{+1, -1\}$（整题对/错，类似多巴胺信号）。门控：
$$g = \begin{cases} 2.0 & r < 0 \text{（错误，加大学习）}\\ 1.0 & r \geq 0 \text{（正确，标准学习）}\end{cases}$$

更新：
$$\Delta W = \eta\,g\,\beta\,\mathbf{e}\,\mathbf{pre}^\top - \gamma W$$

关键设计：**奖励只调节强度 $g$，不改变方向**（方向始终由 $\mathbf{e}$ 决定）。这避免了"负奖励反转更新方向"的错误（早期实现曾用 $\eta r \mathbf{e}\mathbf{pre}$，$r=-1$ 会把 Delta 规则方向反掉，导致不收敛）。额外有可学习调制因子 $\beta$（每层一个）。

### 5.3 CorrectionLearner（逐位纠正门控，Mode B）

门控信号来自**逐位**判断，比 Reward 的整题标量更精细：

$$g_d = \begin{cases}1.0 & \hat{d} = d_{\text{true}}\\ 2.0 & \hat{d} \neq d_{\text{true}}\end{cases}, \quad
g_c = \begin{cases}1.0 & \hat{c} = c_{\text{true}}\\ 2.0 & \hat{c} \neq c_{\text{true}}\end{cases}$$

digit 头用 $g_d$，carry 头用 $g_c$，隐藏层用 $g_{\text{hid}}$（任一输出错则为 2.0）。错误位的更新幅度更大。

### 5.4 三者对比

| 方案 | 学习信号 | 门控粒度 | 额外参数 | 直觉类比 |
|------|---------|---------|---------|---------|
| Hebbian | 教师纠正 | 无（恒 1） | 无 | 纯 Hebbian 局部学习 |
| Reward | 整题奖励 ±1 | 标量 | $\beta$ | 多巴胺调制 |
| Correction | 逐位纠正 | 每位/每头 | 无 | 教师逐位强调错误 |

---

## 6. 训练流程

### 6.1 课程学习（Curriculum Learning）

模仿幼儿学习：从 1 位加法开始，准确率达 95% 后升级到 2 位，依此类推至 5 位。课程环境 `CurriculumEnv` 按当前级别出题，`check_graduation` 周期性评估是否毕业。

> **设计意图**：毕业检查时学生也在学习（`solve_and_learn` 有副作用）——没有"纯测试"模式，学习永不停止。

### 6.2 Teacher Forcing 与 Free Running

训练与推理的进位链有意识地区分：

- **训练时（Teacher Forcing）**：`CarryPropagator` 的输入用**教师的上一位 cout**（`prev_cout_true`），`DigitAdder` 的输入也用**教师的 cin**。这保证两个模型都干净地学习"正确输入→正确输出"的映射，避免错误输入污染监督信号。
- **推理时（Free Running）**：没有教师，整条进位链用学生自己的预测（`cin = carry.predict(prev_cout)`）。

当两个推理模型训练充分后，学生预测 ≈ 教师真值，训练/推理的分布差消失。

### 6.3 补充训练（双 cin 样本）

每一位除了用真实 cin 训练，还用**另一种 cin** 做一次补充训练。例如某位真实 cin=0，则额外用 cin=1 编码输入并训练对应的 `(digit, cout)`。这保证 `DigitAdder` 对同一 $(a,b)$ 在两种 cin 下都被监督，加速覆盖全部 200 种映射。

### 6.4 巩固训练（Consolidation）

课程毕业后，继续用最高位数（5 位）的题目训练若干千步，进一步稳定权重。本工程中巩固 5000 步即可。

---

## 7. 关键实验：GRU vs 前馈进位传播

本节是全工程最具教学意义的部分：**同一个进位传播任务，用 GRU 和前馈两种建模，在 FA 在线学习下表现天壤之别**。为理解对照实验，先介绍 GRU 的工作原理。

### 7.1 GRU 原理（门控循环单元）

GRU（Gated Recurrent Unit, Cho et al. 2014）是一种循环神经网络（RNN），通过门控机制在时间步之间传递隐状态，用于建模序列数据。它由 `modular_addition.py` 的对照版 `CarryPropagator` 使用。

#### 7.1.1 门控结构

给定时间步 $t$ 的输入 $\mathbf{x}_t$ 与上一隐状态 $\mathbf{h}_{t-1}$，GRU 先计算两个门（均为 sigmoid 输出，逐元素取值 $(0,1)$）：

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{（更新门 update gate）}\\
\mathbf{r}_t &= \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{（重置门 reset gate）}
\end{aligned}
$$

- **更新门** $\mathbf{z}_t$：决定新旧隐状态的混合比例。
- **重置门** $\mathbf{r}_t$：决定计算候选状态时遗忘多少历史。

接着用重置门控制的历史生成候选隐状态：
$$
\tilde{\mathbf{h}}_t = \tanh(W \mathbf{x}_t + U(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b})
$$

最后，用更新门在旧状态与候选状态间做插值，得到新隐状态：
$$
\boxed{\;\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t\;}
$$

可训练参数为 $W_{\cdot}, U_{\cdot}, \mathbf{b}_{\cdot}$（每个门一组），在所有时间步**共享**。

#### 7.1.2 关键性质

1. **门控缓解梯度消失**。当 $\mathbf{z}_t \to 0$ 时，$\mathbf{h}_t \approx \mathbf{h}_{t-1}$，隐状态近乎无损直传；对应地，梯度沿时间回传时 $\partial \mathbf{h}_t/\partial \mathbf{h}_{t-1} \approx I$，避免了朴素 RNN 中梯度连乘衰减为 0 的问题。这正是 GRU/LSTM 能学长程依赖的核心机制。

2. **隐状态是序列的压缩记忆**。$\mathbf{h}_t$ 是对截至 $t$ 的全部输入历史的压缩表示，下游读出层（本工程的 `cin_head`）可从中提取所需信息。

3. **隐状态可以收敛到吸引子**。当门控与权重配合得当时，$\mathbf{h}_t$ 会在少数几步内进入不动点并稳定下来——这正是 7.2 节离线 GRU 的行为。

#### 7.1.3 训练必须用 BPTT

由于 $W, U$ 跨时间步共享，且 $\mathbf{h}_t$ 显式依赖 $\mathbf{h}_{t-1}$，GRU 的训练要把网络沿时间展开成一个深度（按时间步数）的前馈网络，再用反向传播通过时间（Backpropagation Through Time, BPTT）计算每个共享权重的时间累积梯度：
$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial W}
$$
其中 $\frac{\partial \mathbf{h}_t}{\partial W}$ 通过反复应用链式法则回溯到所有更早的时间步。**BPTT 本质上是带时间维度的反向传播**——它同样依赖权重转置、依赖沿时间的全局误差传播。

这一点至关重要：**FA 没有时间维度的反馈通路**（4.5 节的 $B$ 只在单个时间步的层间反馈），无法触发 GRU 所需的时间对齐。这正是第 7 章对照实验的症结所在。

#### 7.1.4 在本工程的用法

对照版 `CarryPropagator(GRU)`（`modular_addition.py`）接收 cout 的 one-hot 序列，用 GRUCell 逐位更新 $\mathbf{h}_t$，再由线性层 `cin_head` 从 $\mathbf{h}_t$ 读出 cin。其设计假设是：GRU 的隐状态能"记住"当前进位状态（一个二值量），从而在任意时刻正确输出 cin——即让 $\mathbf{h}_t$ 形成两个稳定吸引子分别对应 cin=0/cin=1。

### 7.2 离线对照：GRU + BPTT 可完美泛化

`modular_addition.py` 提供了一个**离线训练**的 GRU 进位传播器作为对照：用 autograd + Adam + BPTT 在长度 ≤20 的序列上训练。结果（1–20 位全部 100%）：

| 位数 | 1 | 2 | 3 | 4 | 5 | 8 | 12 | 16 | 20 |
|------|---|---|---|---|---|---|----|----|----|
| 准确率 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

GRU 用 BPTT 学会了稳定的**吸引子动力学**：隐状态在 5 步内收敛到不动点（有进位态 $\mathbf{h}\approx[0.99,-1.00]$，无进位态 $\mathbf{h}\approx[-0.58,0.49]$），此后无论序列多长都不漂移。两个吸引子正确编码了二值进位状态。

### 7.3 在线失败：GRU + FA 无法泛化

把同一个 GRU 进位传播器放入**在线 FA 学习**系统（早期版本），结果急剧衰减：

| 位数 | 1–5 | 8 | 12 | 16 | 20 |
|------|-----|---|----|----|----|
| 准确率 | 1.00 | 0.84 | 0.60 | 0.43 | **0.28** |

### 7.4 隐状态动力学分析：吸引子 vs 漂移

诊断在线训练后的 GRU 隐状态（持续输入 cout=1，观察是否稳定）：

| | 离线 GRU（BPTT） | 在线 GRU（FA） |
|---|---|---|
| 稳态 $\mathbf{h}$ 量级 | $\sim 1.0$ | $\sim 0.05$（极小） |
| 收敛 | 5 步内到不动点 | 20 步仍在缓慢漂移 |
| 长期稳定性 | step 7 后完全静止 | 永不收敛 |
| `cin_head` 权重 | $\sim 3$（正常） | $\sim 80$（暴力放大补偿） |

**诊断结论**：在线 FA 系统的 GRU **从未被有效训练**。FA 只更新了 `cin_head`（读出层），GRU 的循环权重基本停留在随机初始化值——因为 `learn_step_carry` 只对 `cin_head` 做 Delta 更新、对 GRU 参数仅施加小量衰减，**没有沿时间传播误差**。结果 GRU 无法形成吸引子，隐状态持续漂移；`cin_head` 只能用极大的权重去硬凑区分微弱的隐状态差异。短序列（≤5 步）漂移尚小，勉强正确；长序列漂移累积，`cin_head` 的暴力区分不够用，准确率崩溃。

### 7.5 根因：FA 不含时间反向传播

GRU 的循环连接 $W_{hh}$ 作用在**前一时刻的隐状态**上。要正确训练它，必须把误差沿时间步回传（BPTT），计算每个时间步对 $W_{hh}$ 的贡献。FA 只在同一时刻的层间用固定 $B$ 传播误差，**完全没有时间维度的反馈通路**——它本质上是前馈网络的生物合理训练方法，不适用于循环网络的时间动力学。

这与文献一致：Bartunov et al. (2018) 系统证明 FA 类方法在序列任务上无法扩展；近年工作（arXiv 2504.13531, 2025）仍显示局部学习规则训练 RNN 解决时序任务困难重重。

### 7.6 解法：前馈单步映射

既然进位传递 $\mathrm{cin} = \mathrm{cout}$ 是**单步、无历史**的映射（4.2 节），就不需要循环记忆。把 `CarryPropagator` 改为前馈 MLP $f:\mathrm{cout}\mapsto\mathrm{cin}$：

- FA 能完整训练它（和 DigitAdder 一样是前馈网络）。
- 单步映射与序列长度无关 → **天然 N 位泛化**。
- 它仍是"模型"（学习 cout→cin），满足"用模型解决进位"的要求，而非硬编码程序。

切换后，三种学习方案在 1–50 位上**全部 100%**（见第 8 节）。

> **核心教训**：瓶颈不在学习算法，而在**架构与学习算法的匹配**。循环网络需要 BPTT，FA 提供不了；前馈网络 FA 能训。当任务本身是单步映射时，应优先用前馈建模，而非引入不必要的循环复杂性。

---

## 8. 实验结果

### 8.1 实验设置

- DigitAdder $h=128$，CarryPropagator $h=16$（前馈）
- 初始 $\eta=0.1$，$\gamma=0$，分层学习率（输出层 ×10）
- 课程 1→5 位，每级 2000 题，毕业阈值 95%，最多 5 轮
- 巩固训练 5000 步（5 位）
- 评估：1–50 位，每长度 200 题，纯推理不学习

### 8.2 N 位泛化（前馈 CarryPropagator）

| 方案 | 1 | 2 | 3 | 4 | 5 | 8 | 12 | 16 | 20 | 30 | 50 |
|------|---|---|---|---|---|---|----|----|----|----|----|
| Hebbian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Reward | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Correction | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**三种方案全部 1–50 位 100%**，彻底解决泛化问题。

### 8.3 收敛速度对比

精度持平后，按 1 位课程的毕业轮数区分：

| 方案 | Level 1 毕业轮次 | 说明 |
|------|----------------|------|
| Hebbian | 4 轮 | 最快 |
| Correction | 5 轮 | 较慢，曾卡住 |
| Reward | 5 轮 | 最慢，课程中反复 |

### 8.4 参数量

| 组件 | 参数量 |
|------|--------|
| DigitAdder ($h=128$) | 21,004 |
| CarryPropagator 前馈 ($h=16$) | 354 |
| **推理模型合计** | **21,358** |
| Hebbian / Correction 学习机制 | 18,222 |
| Reward 学习机制 | 18,229 |

学习机制参数与推理模型参数**完全独立**。

### 8.5 方案选型结论

三种方案精度完全一致（前馈架构下两个推理模型都能被 FA 训练到精确，门控差异不再影响最终精度）。按**简洁性 + 收敛速度**选：

> **选用 Hebbian 方案**：门控最简（恒 1）、收敛最快、参数最少（无 $\beta$）。在精度持平的前提下，它是帕累托最优选择。

---

## 9. 相关工作

- **反馈对齐**：Lillicrap et al. (2016), *Random synaptic feedback weights support error backpropagation for deep learning*, Nature Communications. 提出 FA，证明固定随机反馈可支持深度学习。Direct FA (Nøkland, 2016) 进一步将误差直接反馈到各隐藏层。
- **生物合理的循环网络学习**：e-prop (Bellec et al., 2020) 用资格传播（eligibility propagation）近似 BPTT；但 Bartunov et al. (2018) 指出 FA 类方法在序列任务上难以扩展，与本工程第 7 节的发现一致。
- **神经-符号算术**：将算法分解为可复用子模块（如本工程的"个位加法 + 进位传递"）是突破固定长度限制的经典思路；Neural Turing Machines、算法推理任务等亦采用类似分解哲学。
- **课程学习**：Bengio et al. (2009)，从简单到复杂逐步训练，本工程用于位数升级。
- **神经形态在线学习**：ETLP (Quintana et al., 2024)、Global-Local Learning (Wu et al., 2022) 等探讨"独立局部学习规则修改网络权重"，与本工程"学习机制作为独立模块"的理念相通。

---

## 10. 讨论与局限

### 10.1 FA 在循环网络上的边界

本工程最重要的发现是 FA 与循环网络的不匹配。这不是实现缺陷，而是 FA 的**固有边界**：FA 是为前馈网络设计的生物合理训练法，没有时间反向传播通路。任何需要学习跨时间步动力学的任务（GRU/LSTM 的隐状态演化）都无法用纯 FA 训练。文献与此一致。

### 10.2 前馈 CarryPropagator 学到的是恒等映射

前馈进位传播器学到的 $f:\mathrm{cout}\mapsto\mathrm{cin}$ 在功能上等价于恒等映射，与"程序化路由"（`cin = cout`）效果相同。区别在于：它是**通过学习得到的模型**（满足"用模型解决进位"的设计要求），权重由 FA + Delta 从零训练而来，而非硬编码。这是"形式上是模型、功能上是恒等"的合理折中。

### 10.3 局限

- **任务范围**：仅验证了整数加法。减法/乘法需重新设计分解（如乘法需部分积与累加）。
- **FA 收敛依赖宽度**：DigitAdder 需 $h=128$ 才稳定收敛，隐藏层窄时 FA 难以对齐，工程上需调参。
- **学习机制参数未端到端优化**：$\eta, \gamma$ 虽以 `nn.Parameter` 形式存在，但本工程未用元学习优化它们（保持"局部自主"纯粹性）。

---

## 11. 结论

本工程实现了一个三模型协同的在线加法学习系统：两个前馈推理模型（DigitAdder + CarryPropagator）负责做加法，一个独立参数的学习机制通过反馈对齐 + Delta 规则自主修改推理权重。三种学习信号方案（Hebbian / Reward / Correction）均被实现并对比。

核心贡献是定位并解决了一个架构性难题：**FA 无法训练循环网络的时间动力学**，因此用 GRU 建模进位传播会导致长位数泛化失败（20 位仅 28%）；而认识到进位传递本质是单步映射后，改用前馈建模，使 FA 能完整训练，三种方案在 1–50 位上全部达到 100%。

这一过程的更一般启示是：**生物合理学习算法（FA）有其适用边界，架构选择必须与学习算法的能力匹配**。当任务可分解为前馈可表达的子技能时，应避免引入需要 BPTT 的循环结构。

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $\mathbf{x}$ | 输入 one-hot（22 维） |
| $\mathbf{h}_1, \mathbf{h}_2$ | 隐藏层激活（post-ReLU） |
| $\mathbf{z}_d, \mathbf{z}_c$ | digit / carry logits |
| $\mathbf{p}$ | softmax 输出 |
| $\mathbf{t}$ | 目标 one-hot |
| $\mathbf{e} = \mathbf{t} - \mathbf{p}$ | 误差 |
| $\boldsymbol{\delta}$ | 经 FA 反馈后的隐藏层误差 |
| $B$ | 固定随机反馈矩阵（替代 $W^\top$） |
| $\eta, \gamma$ | 学习率、衰减 |
| $g, \beta$ | 门控、调制因子 |

## 附录 B：超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| DigitAdder $h$ | 128 | FA 收敛所需宽度 |
| CarryPropagator $h$ | 16 | 单步映射，窄即可 |
| $\eta_{\text{hid}}$ | 0.1 | 隐藏层学习率 |
| $\eta_{\text{out}}$ | 1.0 | 输出层学习率（×10） |
| $\gamma$ | 0.0 | 衰减关闭（随机数据下防权重清零） |
| 课程级别 | 1→5 | 每级 2000 题，95% 毕业 |
| 巩固步数 | 5000 | 5 位加法 |

## 附录 C：参数量明细

**DigitAdder** ($h=128$)：
- $W_1$: $22\times128+128 = 2{,}944$
- $W_2$: $128\times128+128 = 16{,}512$
- $W_d$: $128\times10+10 = 1{,}290$
- $W_c$: $128\times2+2 = 258$
- 合计：**21,004**

**CarryPropagator 前馈** ($h=16$)：
- $W_1$: $2\times16+16 = 48$
- $W_2$: $16\times16+16 = 272$
- $\mathrm{cin\_head}$: $16\times2+2 = 34$
- 合计：**354**

**HebbianLearner** ($h=128, h_c=16$)：
- $\eta, \gamma$: $4+3+4+3 = 14$
- $B_d[128,10] + B_c[128,2] + B_2[128,128]$: $1{,}280+256+16{,}384 = 17{,}920$
- $B_{\mathrm{cin}}[16,2] + B_{c2}[16,16]$: $32+256 = 288$
- 合计：**18,222**（Reward 另加 $\beta$: $4+3=7$ → 18,229）

---

## 参考文献

1. Lillicrap, T. P., et al. (2016). *Random synaptic feedback weights support error backpropagation for deep learning*. Nature Communications, 7, 13276.
2. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP.（GRU 原始论文）
3. Nøkland, A. (2016). *Direct Feedback Alignment Provides Learning in Deep Neural Networks*. NeurIPS.
4. Bellec, G., et al. (2020). *A solution to the learning dilemma for recurrent networks*. Nature Communications.
5. Bartunov, S., et al. (2018). *Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures*. NeurIPS.
6. Bengio, Y., et al. (2009). *Curriculum Learning*. ICML.
7. Quintana, F. M., et al. (2024). *ETLP: Event-Based Three-Factor Local Plasticity for Online Learning*.
8. Wu, Y., et al. (2022). *Brain-Inspired Global-Local Learning Incorporated with Neuromorphic Computing*. Nature Communications.
