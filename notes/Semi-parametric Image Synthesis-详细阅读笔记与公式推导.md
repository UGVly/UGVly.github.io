---
tags: [public]
category: 研究笔记
---

# Semi-parametric Image Synthesis：详细阅读笔记与公式推导

论文：

- 标题：*Semi-parametric Image Synthesis*
- 作者：Xiaojuan Qi, Qifeng Chen, Jiaya Jia, Vladlen Koltun
- arXiv：<https://arxiv.org/abs/1804.10992>
- CVPR 2018 OpenAccess PDF：<https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Semi-Parametric_Image_Synthesis_CVPR_2018_paper.pdf>

这篇论文的正文里，真正显式写出来的公式并不多，核心只有三块：

1. 检索分数；
2. 变换网络损失；
3. 合成网络感知损失。

但如果想真正把方法吃透，必须把论文里隐含的对象、算子和训练流程一并补全。下面我按“问题定义 -> 检索 -> 对齐 -> 排序与合成 -> 最终生成 -> 训练目标”的顺序，把整条链完整写一遍。

---

## 1. 这篇论文到底在做什么

给定一个语义布局
$$
L \in \{0,1\}^{h\times w\times c},
$$
其中 \(h\times w\) 是图像分辨率，\(c\) 是语义类别数，目标是合成一张真实感图像 \(\hat I\in\mathbb R^{h\times w\times 3}\)。

传统纯参数方法是直接学
$$
\hat I = g_\theta(L),
$$
也就是让网络把“所有外观细节”都记在参数里。

本文的方法不是这样。它把生成拆成两部分：

- **非参数部分**：从训练集构造一个外部 memory bank，测试时根据输入布局从里面检索真实图像片段；
- **参数部分**：用神经网络把这些检索到的片段对齐、排序、合成，并进一步修补与润色。

因此整条推理链写成
$$
L
\;\Longrightarrow\;
\{L_j\}_{j=1}^m
\;\Longrightarrow\;
\{P_{\sigma(j)}\}_{j=1}^m
\;\Longrightarrow\;
\{\tilde P_{\sigma(j)}\}_{j=1}^m
\;\Longrightarrow\;
C
\;\Longrightarrow\;
\hat I.
$$

这里：

- \(\{L_j\}_{j=1}^m\) 是输入布局中的连通语义区域；
- \(P_{\sigma(j)}\) 是为第 \(j\) 个区域从 memory 中检索到的真实图像片段；
- \(\tilde P_{\sigma(j)}\) 是经过空间变换网络对齐后的片段；
- \(C\) 是把所有片段按遮挡顺序贴到一起得到的 canvas；
- \(\hat I\) 是最终合成图像。

---

## 2. 数学对象与记号

### 2.1 语义布局

令像素网格为
$$
\Omega=\{1,\dots,h\}\times\{1,\dots,w\}.
$$

语义布局 \(L\) 用 one-hot 张量表示：
$$
L(x)\in\{e_1,\dots,e_c\},\qquad x\in\Omega,
$$
其中 \(e_k\in\{0,1\}^c\) 是第 \(k\) 类的 one-hot 向量。

等价地，
$$
L\in\{0,1\}^{h\times w\times c},
\qquad
\sum_{k=1}^c L_{x,k}=1.
$$

将 \(L\) 分解成同类像素的最大连通分量：
$$
L \rightsquigarrow \{L_j\}_{j=1}^m.
$$

每个 \(L_j\) 对应一个语义片段，其类别记作
$$
\kappa(j)\in\{1,\dots,c\}.
$$

记其支撑集为
$$
\Omega_j := \{x\in\Omega:\, L_j(x)=1\}.
$$

### 2.2 训练集与 memory bank

训练集为
$$
\mathcal D=\{(I^{(n)},L^{(n)})\}_{n=1}^N,
$$
其中
$$
I^{(n)}\in\mathbb R^{h\times w\times 3},
\qquad
L^{(n)}\in\{0,1\}^{h\times w\times c}.
$$

对每张训练布局 \(L^{(n)}\) 做 connected components 分解，得到大量训练片段。所有片段合在一起构成 memory bank
$$
M=\{P_i\}_{i=1}^{|M|}.
$$

每个片段 \(P_i\) 由一个三元组表示：
$$
P_i = \big(P_i^{\mathrm{color}},\, P_i^{\mathrm{mask}},\, P_i^{\mathrm{cont}}\big).
$$

其中：

1. **颜色片段**
$$
P_i^{\mathrm{color}}\in\mathbb R^{h\times w\times 3}
$$
表示整张图大小的 RGB 张量，但只有该片段所在区域保留原像素，其余位置补零。

2. **片段 mask**
$$
P_i^{\mathrm{mask}}\in\{0,1\}^{h\times w\times c}
$$
表示这个片段的 one-hot 语义 footprint。

3. **语义上下文**
$$
P_i^{\mathrm{cont}}\in\{0,1\}^{h\times w\times c}
$$
表示片段周围的语义环境。

更具体地，若片段 \(P_i\) 的支撑区域为 \(\Omega_i\)，其 bounding box 记为
$$
B_i = \operatorname{BBox}(\Omega_i),
$$
论文把它在长宽两个方向各扩张 \(25\%\)，得到扩张框
$$
\bar B_i = \operatorname{Expand}_{25\%}(B_i).
$$
于是可以把上下文写成
$$
P_i^{\mathrm{cont}}(x)=L^{(n_i)}(x)\,\mathbf 1[x\in \bar B_i].
$$

这一定义对应论文中的那句话：context 是“片段周围的语义布局，在一个放大的 bounding box 中截取出来”。

---

## 3. 论文方法的总公式

如果把整篇论文写成一个统一的复合映射，那么测试时的生成过程可以记为
$$
\hat I
=
f_{\theta_f}\!\big(C(L),\,L\big),
$$
其中 canvas \(C(L)\) 是由若干中间步骤得到的：
$$
C(L)
=
\operatorname{Compose}\Big(
\big\{
T_{\theta_T}\big(\mathcal T_j,\;P_{\sigma(j)}^{\mathrm{color}}\big)
\big\}_{j=1}^m;\;
\pi
\Big).
$$

这里：

- \(\sigma(j)\) 是检索索引；
- \(T_{\theta_T}\) 是 transformation network；
- \(\mathcal T_j\) 表示目标布局信息，至少包含 \(L\) 与目标局部 mask \(L_j^{\mathrm{mask}}\)；
- \(\pi\) 是 ordering network 给出的从后到前的绘制顺序；
- \(f_{\theta_f}\) 是最终 synthesis network。

这套写法有一个非常重要的性质：

**检索步骤 \(\sigma(j)\) 直接访问外部数据库 \(M\)，因此模型不是纯参数模型，而是 semi-parametric。**

---

## 4. 外部 Memory 的检索公式

### 4.1 查询对象

对测试布局 \(L\) 的每个连通分量 \(L_j\)，构造与训练片段同构的两个量：
$$
L_j^{\mathrm{mask}},\qquad L_j^{\mathrm{cont}}.
$$

其中：

- \(L_j^{\mathrm{mask}}\) 是该连通区域自身的 one-hot mask；
- \(L_j^{\mathrm{cont}}\) 是其在放大 bounding box 内的语义上下文。

### 4.2 IoU 的精确定义

论文把相似性建立在 IoU 上。对两个同形状的二值张量 \(A,B\in\{0,1\}^{h\times w\times c}\)，IoU 定义为
$$
\operatorname{IoU}(A,B)
=
\frac{|A\cap B|}{|A\cup B|}
=
\frac{\sum_{x,k} A_{x,k}B_{x,k}}
{\sum_{x,k}\max(A_{x,k},B_{x,k})}.
$$

由于 \(A,B\) 是二值张量，也可写成
$$
\operatorname{IoU}(A,B)
=
\frac{\langle A,B\rangle}
{\|A\|_1+\|B\|_1-\langle A,B\rangle}.
$$

### 4.3 论文中的检索分数

对第 \(j\) 个测试片段，候选集合限制在同语义类：
$$
\mathcal M_j:=\{i:\kappa(i)=\kappa(j)\}.
$$

然后定义分数
$$
s(i,j)
=
\operatorname{IoU}\big(P_i^{\mathrm{mask}},L_j^{\mathrm{mask}}\big)
+
\operatorname{IoU}\big(P_i^{\mathrm{cont}},L_j^{\mathrm{cont}}\big).
$$

最终检索结果是
$$
\sigma(j)
=
\arg\max_{i\in\mathcal M_j} s(i,j).
$$

这就是论文在 Section 4.2 的核心公式。

### 4.4 这个公式为什么合理

这个分数由两部分组成：

1. **mask IoU**
$$
\operatorname{IoU}\big(P_i^{\mathrm{mask}},L_j^{\mathrm{mask}}\big)
$$
衡量候选片段形状和目标片段形状是否匹配。

2. **context IoU**
$$
\operatorname{IoU}\big(P_i^{\mathrm{cont}},L_j^{\mathrm{cont}}\big)
$$
衡量片段周围语义环境是否匹配。

于是，作者不是单纯找“形状最像”的车，而是找“形状像、而且出现在类似周边环境中的车”。

在数学上，这个最大化问题等价于最小化一个加权 Jaccard 距离：
$$
\sigma(j)
=
\arg\min_{i\in\mathcal M_j}
\Big[
\big(1-\operatorname{IoU}(P_i^{\mathrm{mask}},L_j^{\mathrm{mask}})\big)
+
\big(1-\operatorname{IoU}(P_i^{\mathrm{cont}},L_j^{\mathrm{cont}})\big)
\Big].
$$

也就是说，它本质上是一个基于形状与上下文的 nearest-neighbor retrieval。

### 4.5 多样性版本

论文最后说，如果要生成多样结果，不必总取全局最优 \(\arg\max\)，而可以在 top-\(k\) 候选里随机采样：
$$
\sigma_k(j)\sim
\operatorname{Unif}\Big(
\operatorname{TopK}_{i\in\mathcal M_j} s(i,j)
\Big).
$$

这样同一个布局 \(L\) 可以生成多种风格不同但语义兼容的图像。

---

## 5. 变换网络 \(T\)：从检索到对齐

检索出的真实片段虽然语义类与上下文都匹配，但其位置、尺度、旋转角度通常与当前输入布局不一致，因此必须做几何对齐。

### 5.1 网络输入输出

论文把 transformation network 记为
$$
T\big(L,\;L_j^{\mathrm{mask}},\;P_{\sigma(j)}^{\mathrm{color}}\big),
$$
输出为对齐后的片段
$$
\tilde P_{\sigma(j)}.
$$

更抽象地，我们记目标条件为
$$
\mathcal T_j := \big(L,\;L_j^{\mathrm{mask}}\big),
$$
则
$$
\tilde P_{\sigma(j)}
=
T_{\theta_T}\big(\mathcal T_j,\;P_{\sigma(j)}^{\mathrm{color}}\big).
$$

### 5.2 仿射变换的标准写法

论文明确说这里使用的是 **2D affine transformation**。因此，网络本质上是预测一个仿射矩阵
$$
A_j
=
\begin{bmatrix}
a_{11} & a_{12} & t_x\\
a_{21} & a_{22} & t_y
\end{bmatrix}.
$$

对目标坐标 \((x_t,y_t)\)，其对应的源坐标 \((x_s,y_s)\) 满足
$$
\begin{bmatrix}
x_s\\
y_s
\end{bmatrix}
=
A_j
\begin{bmatrix}
x_t\\
y_t\\
1
\end{bmatrix}.
$$

### 5.3 STN 的双线性采样公式

若把待变换图像记为 \(U=P_{\sigma(j)}^{\mathrm{color}}\)，输出图像记为 \(V=\tilde P_{\sigma(j)}\)，则对通道 \(c\) 有
$$
V_c(x_t,y_t)
=
\sum_{m=1}^h\sum_{n=1}^w
U_c(m,n)\,
\max(0,1-|x_s-n|)\,
\max(0,1-|y_s-m|).
$$

这就是 spatial transformer 的标准双线性采样写法。它有两个好处：

- 对仿射参数可微；
- 对输入像素值也可微。

因此可以端到端地反向传播更新 \(\theta_T\)。

### 5.4 训练样本怎么构造

如果直接拿一个正确片段 \(P_i^{\mathrm{color}}\) 让网络去对齐 \(P_i^{\mathrm{mask}}\)，任务会过于平凡，因为它本来就是对齐的。

论文的处理方式是：对 \(P_i^{\mathrm{color}}\) 施加随机仿射扰动与裁剪，得到
$$
\hat P_i^{\mathrm{color}}
=
W_{\mathrm{rand}}\big(P_i^{\mathrm{color}}\big).
$$

于是训练任务变成：

- 输入：被扰乱过的片段 \(\hat P_i^{\mathrm{color}}\) 与目标布局条件；
- 输出：恢复到原始位置与形状的片段。

### 5.5 论文里的变换损失

论文给出的损失是
$$
L_T(\theta_T)
=
\sum_{P_i\in M}
\left\|
P_i^{\mathrm{color}}
-
T_{\theta_T}\big(\mathcal T_i,\hat P_i^{\mathrm{color}}\big)
\right\|_1.
$$

这里 \(\mathcal T_i\) 表示与第 \(i\) 个 memory 片段对应的目标布局信息。论文版面中在这一处的记号略有压缩，但上下文一致的理解就是：

> 给定目标 mask / layout，以及一个几何上被扰乱的片段，让网络把它对齐回正确位置。

### 5.6 为什么用颜色图像做 \(L_1\) 而不是只对 mask 做监督

如果只监督 mask，对齐会有大量不唯一解。比如一个车片段，单从 mask 很难分辨“正着放还是翻转后放”，而颜色纹理提供了更强的约束。

所以作者强调：损失定义在 color image 上，比定义在 mask 上更具体。

从优化角度看，\(L_1\) 损失的梯度是
$$
\nabla_{\theta_T} L_T
=
\sum_i
J_{\theta_T}\!\Big[T_{\theta_T}(\mathcal T_i,\hat P_i^{\mathrm{color}})\Big]^\top
\operatorname{sgn}\!\Big(
T_{\theta_T}(\mathcal T_i,\hat P_i^{\mathrm{color}})
-
P_i^{\mathrm{color}}
\Big),
$$
其中 \(J_{\theta_T}\) 是输出对参数的 Jacobian。因为 STN 的采样过程可微，所以这条链完整可导。

---

## 6. Ordering Network 与 Canvas 合成

### 6.1 为什么需要排序

若所有变换后的片段都互不重叠，那么直接贴到画布上就行。但现实情况是，它们会发生重叠：
$$
\operatorname{supp}(\tilde P_{\sigma(i)})
\cap
\operatorname{supp}(\tilde P_{\sigma(j)})
\neq \varnothing.
$$

一旦重叠，就必须决定谁在前，谁在后。

### 6.2 排序网络的输出

论文说 ordering network 的输出是一个 \(c\) 维 one-hot 向量，表示“哪一个语义类应该在前面”。把它写成概率向量：
$$
p_{ij}
=
O_{\theta_O}\big(\tilde P_{\sigma(i)},\tilde P_{\sigma(j)},L\big)
\in\Delta^{c-1}.
$$

设真实前景类标签为
$$
y_{ij}\in\{e_1,\dots,e_c\},
$$
则 cross-entropy 损失为
$$
L_O(\theta_O)
=
-
\sum_{(i,j)\in\mathcal A}
\sum_{r=1}^c
y_{ij,r}\log p_{ij,r},
$$
其中 \(\mathcal A\) 是训练时采样到的相邻片段对集合。

### 6.3 监督信号从哪来

论文的监督来自 segment 的相对深度顺序：

- Cityscapes / NYU 这类数据集自带 depth 或 stereo 信息；
- ADE20K 没有，就先用单目深度网络估计深度。

论文没有把“segment 级别深度标签如何从像素级深度聚合出来”写成公式；实现上只要能为相邻片段对给出一个可靠的 front/back 标号即可。

### 6.4 画布合成公式

设经过排序后，从后到前的绘制顺序为
$$
\pi(1),\pi(2),\dots,\pi(m).
$$

记第 \(t\) 个片段的可见 mask 为 \(M_{\pi(t)}\in\{0,1\}^{h\times w}\)，其 RGB 内容仍记为 \(\tilde P_{\pi(t)}\)。那么 canvas 的递推合成可以写成
$$
C^{(0)}(x)=0,
$$
$$
C^{(t)}(x)
=
M_{\pi(t)}(x)\,\tilde P_{\pi(t)}(x)
+
\big(1-M_{\pi(t)}(x)\big)\,C^{(t-1)}(x),
\qquad t=1,\dots,m.
$$

最终 canvas 是
$$
C = C^{(m)}.
$$

如果使用边界擦除后的 mask，公式完全一样，只是把 \(M_{\pi(t)}\) 换成边界处理后的 \(\bar M_{\pi(t)}\)。

这个公式本质上就是 alpha compositing 的二值特例。

---

## 7. 最终 Synthesis Network \(f\)

Canvas \(C\) 只是“半成品”：

- 有缺块；
- 不同 segment 的颜色和光照不一致；
- 边界容易穿帮；
- 阴影与局部相互作用通常缺失。

所以论文再接一个 synthesis network：
$$
\hat I = f_{\theta_f}(C,L,m),
$$
其中 \(m\in\{0,1\}^{h\times w}\) 是 canvas 中缺失像素的 binary mask。

论文行文中有时直接写 \(f(C,L)\)，有时提到还输入了 missing-pixel mask。统一理解就是：网络输入包含 canvas、语义布局，以及缺失区域指示。

### 7.1 编码器

把输入拼接成张量
$$
X_0 = [C,L,m].
$$

编码器是五层模块堆叠。若第 \(s\) 层编码器记为 \(\mathcal E_s\)，则可写成
$$
E_1 = \mathcal E_1(X_0),
$$
$$
E_s = \mathcal E_s\big(\operatorname{Pool}(E_{s-1})\big),\qquad s=2,\dots,5.
$$

论文说明每个 encoder output element 的感受野约为
$$
276\times 276,
$$
因此能建模较长程的纹理、色彩和光照关系。

### 7.2 解码器

解码器基于 CRN（cascaded refinement network）。若把第 \(s\) 个 refinement block 记为 \(\mathcal R_s\)，并把 resize 到对应尺度的 canvas / layout 记为 \(C_s,L_s\)，则有
$$
R_5 = \mathcal R_5([E_5,C_5,L_5]),
$$
$$
R_s
=
\mathcal R_s\Big(
\big[
E_s,\,
\operatorname{Up}(R_{s+1}),\,
C_s,\,
L_s
\big]
\Big),
\qquad s=4,3,2,1.
$$

最终输出是
$$
\hat I = \operatorname{Head}(R_1).
$$

这部分虽然不是论文逐行写出的公式，但和论文的结构描述完全一致。

---

## 8. 感知损失 \(L_f\) 的完整写法

### 8.1 论文显式给出的损失

对训练样本 \((I,L)\)，先构造一个模拟测试时瑕疵的 canvas \(C'\)，然后让 synthesis network 恢复原图 \(I\)：
$$
\hat I = f_{\theta_f}(C',L).
$$

论文给出的感知损失是
$$
L_f(\theta_f)
=
\sum_{(I,L)\in\mathcal D}
\sum_{l\in\mathcal S}
\lambda_l
\left\|
\Phi_l(I)
-
\Phi_l\big(f_{\theta_f}(C',L)\big)
\right\|_1,
$$
其中：

- \(\Phi_l\) 是固定的预训练 VGG-19 第 \(l\) 层特征；
- \(\mathcal S=\{\texttt{conv1\_2},\texttt{conv2\_2},\texttt{conv3\_2},\texttt{conv4\_2},\texttt{conv5\_2}\}\)；
- \(\lambda_l>0\) 是各层权重。

### 8.2 为什么这是“感知损失”

如果直接做像素级 \(L_1\)：
$$
\|I-\hat I\|_1,
$$
网络会倾向于平均化，常导致模糊。

而感知损失是在预训练视觉网络特征空间里逼近：
$$
\Phi_l(\hat I)\approx \Phi_l(I).
$$

低层特征约束颜色与边缘，高层特征约束纹理和结构，所以更适合照片合成任务。

### 8.3 梯度链式法则

由于 \(\Phi_l\) 冻结不训练，对 \(\theta_f\) 的梯度为
$$
\nabla_{\theta_f}L_f
=
\sum_{(I,L)}
\sum_{l\in\mathcal S}
\lambda_l\,
J_{\theta_f}\!\Big[\Phi_l(\hat I)\Big]^\top
\operatorname{sgn}\!\Big(\Phi_l(\hat I)-\Phi_l(I)\Big),
$$
其中
$$
\hat I = f_{\theta_f}(C',L).
$$

再把 Jacobian 展开一层：
$$
J_{\theta_f}\!\Big[\Phi_l(\hat I)\Big]
=
J_{\hat I}\!\big[\Phi_l(\hat I)\big]\,
J_{\theta_f}\!\big[f_{\theta_f}(C',L)\big].
$$

这说明优化时，误差会先通过固定的 VGG 特征网络反传回输出图像，再由合成网络继续反传回参数 \(\theta_f\)。

---

## 9. 模拟训练画布 \(C'\) 是怎么构造出来的

这是整篇论文非常关键、但很多人第一次读会忽略的地方。

作者没有在训练时直接把真实 memory 检索结果拿来当 canvas，而是从真实图像 \(I\) 出发，**人为制造出与测试时相似的伪缺陷**，构造模拟画布 \(C'\)。这样网络学到的是“修复与融合”的能力。

### 9.1 Step 1：Stenciling

对真实图像里的每个 segment \(P_j\)，从别的图像里检索一个同类 segment，取其 mask 来裁切 \(P_j\)：
$$
P_j^{\mathrm{stencil}}
=
P_j^{\mathrm{color}}\odot M_{r(j)},
$$
其中：

- \(r(j)\) 是检索到的 donor segment；
- \(M_{r(j)}\in\{0,1\}^{h\times w}\) 是 donor 的二值 mask；
- \(\odot\) 是逐像素乘法。

这一步人为制造“缺块”和“形状不一致”。

### 9.2 Step 2：Color transfer

论文在这里引用的是 Reinhard 等人的颜色迁移方法 [27]，正文没展开公式。若在 Lab 色彩空间中写，其标准形式是：
$$
\tilde x_c
=
\frac{\sigma_{r(j),c}}{\sigma_{j,c}+\varepsilon}
\big(x_c-\mu_{j,c}\big)
+
\mu_{r(j),c},
\qquad c\in\{L,a,b\}.
$$

这里：

- \(\mu_{j,c},\sigma_{j,c}\) 是当前 segment \(P_j\) 在通道 \(c\) 上的均值与标准差；
- \(\mu_{r(j),c},\sigma_{r(j),c}\) 是 donor segment 的统计量。

这一步会故意制造 segment 之间的色调、亮度不一致。

### 9.3 Step 3：Boundary elision

定义 segment 边界为 \(\partial\Omega_j\)，到边界的距离为
$$
d_j(x):=\operatorname{dist}(x,\partial\Omega_j).
$$

#### 内边界擦除

论文说：对距离边界 \(0.05h\) 以内的 segment 内部像素，随机擦掉 \(80\%\)，并填成白色。

于是内部边界带可写为
$$
B_j^{\mathrm{in}}
:=
\{x\in\Omega_j:\,d_j(x)\le 0.05h\}.
$$

若 \(R_j(x)\sim\operatorname{Bernoulli}(0.8)\)，则被擦掉的内部边界像素集合为
$$
\tilde B_j^{\mathrm{in}}
:=
\{x\in B_j^{\mathrm{in}}:\,R_j(x)=1\}.
$$

#### 外边界擦除

外部边界带定义为
$$
B_j^{\mathrm{out}}
:=
\{x\notin\Omega_j:\,d_j(x)\le 0.05h\}.
$$

论文对这部分直接 mask out，并填成黑色。这样做的目的是逼网络学会补阴影、接触区域、边缘融合等局部相互作用。

### 9.4 最终模拟画布

把经过上述处理的各个训练片段按真实布局或对应顺序合成，就得到模拟测试时缺陷的画布
$$
C'.
$$

于是训练目标变成
$$
(C',L)\mapsto I.
$$

这个设计非常漂亮，因为它把任务从“从零生成整张图”转化为：

- 以真实照片片段为原料；
- 修补缺失区域；
- 融合颜色与光照；
- 消除边界穿帮。

这也是为什么它比当时很多纯参数方法更真实。

---

## 10. 三个训练目标分别是什么

这篇论文不是端到端联合训练，而是分模块训练：

### 10.1 变换网络
$$
\min_{\theta_T} L_T(\theta_T).
$$

### 10.2 排序网络
$$
\min_{\theta_O} L_O(\theta_O).
$$

### 10.3 图像合成网络
$$
\min_{\theta_f} L_f(\theta_f).
$$

因此它不是
$$
\min_{\theta_T,\theta_O,\theta_f}
\big(L_T+\alpha L_O+\beta L_f\big)
$$
这样的联合优化，而是三段式 pipeline。

这也解释了论文最后为什么把“not trained end-to-end”列为未来工作之一。

---

## 11. 把整条推导压缩成一个完整流程

现在把前面所有对象串起来。

### 11.1 测试阶段

给定输入布局 \(L\)：

1. 连通域分解
$$
L \rightsquigarrow \{L_j\}_{j=1}^m.
$$

2. 为每个 \(L_j\) 检索 donor segment
$$
\sigma(j)
=
\arg\max_{i\in\mathcal M_j}
\Big[
\operatorname{IoU}(P_i^{\mathrm{mask}},L_j^{\mathrm{mask}})
+
\operatorname{IoU}(P_i^{\mathrm{cont}},L_j^{\mathrm{cont}})
\Big].
$$

3. 对齐 donor segment
$$
\tilde P_{\sigma(j)}
=
T_{\theta_T}(\mathcal T_j,P_{\sigma(j)}^{\mathrm{color}}).
$$

4. 用 ordering network 决定重叠片段顺序，得到 \(\pi\)。

5. 合成 canvas
$$
C^{(0)}=0,\qquad
C^{(t)}(x)
=
M_{\pi(t)}(x)\tilde P_{\pi(t)}(x)
+
\big(1-M_{\pi(t)}(x)\big)C^{(t-1)}(x).
$$

6. 用 synthesis network 输出最终图像
$$
\hat I = f_{\theta_f}(C,L,m).
$$

### 11.2 训练阶段

1. 从训练图像中抽 connected components 建立 memory \(M\)；
2. 训练 transformation network 的对齐能力；
3. 训练 ordering network 的前后遮挡判断；
4. 通过 stenciling + color transfer + boundary elision 构造 \(C'\)；
5. 用感知损失训练 synthesis network 恢复真实图像。

---

## 12. 结果怎么读

论文实验主要给出三类证据。

### 12.1 人工 A/B 偏好

平均来看：

- SIMS 比 Pix2pix 更真实：**94.1%**
- SIMS 比 CRN 更真实：**86.1%**

### 12.2 语义分割可解析性

把生成图再送进 PSPNet 做语义分割，比较其输出和原始布局的一致性。论文报告例如：

- Cityscapes-coarse：SIMS 的 IoU 为 **56.3%**，明显高于 Pix2pix 的 **30.1%** 与 CRN 的 **28.5%**
- ADE20K：SIMS 的 IoU 为 **38.4%**，高于 Pix2pix 的 **16.1%** 与 CRN 的 **23.1%**

这个实验很有意思，因为它同时检验：

- 图像是否像真的；
- 图像是否仍然忠实于输入布局。

### 12.3 频谱统计

作者比较了生成图像与真实图像的平均 power spectrum。SIMS 的频谱曲线与真实图像几乎重合，而 Pix2pix / CRN 则有明显尖刺。

如果把图像 \(I\) 的二维 Fourier 变换记为 \(\mathcal F(I)\)，功率谱写成
$$
S_I(\omega)
=
\big|\mathcal F(I)(\omega)\big|^2,
$$
那么数据集平均功率谱可记为
$$
\bar S(\omega)
=
\frac{1}{N}\sum_{n=1}^N S_{I^{(n)}}(\omega).
$$

论文的观察是：SIMS 的 \(\bar S(\omega)\) 更接近真实图像的 \(\bar S_{\mathrm{real}}(\omega)\)。

---

## 13. 这篇论文为什么有效

我觉得可以从下面三点理解。

### 13.1 高频细节不是全靠网络“背下来”

纯参数模型需要在权重里记住所有纹理模式。SIMS 则在测试时直接从数据库里拿真实片段，因此：

- 纹理更像真照片；
- 高频统计更自然；
- 细节生成压力从“凭空 hallucinate”变成“检索后融合”。

### 13.2 最终网络做的是“融合与修补”，不是“从零捏整张图”

这使得任务难度明显下降。网络主要负责：

- 对齐后的局部 harmonization；
- 缺块补全；
- 边缘过渡；
- 阴影和接触区域补充。

### 13.3 输入布局 \(L\) 与 canvas \(C\) 双重约束

若只看 canvas，网络可能会被 donor segment 带偏；
若只看 layout，又会退化成纯参数生成器。

两者同时输入后，网络既能利用真实 donor 的 appearance，又不会脱离目标语义布局。

---

## 14. 这篇论文的局限

论文自己也提到了，结合公式更容易看清问题在哪。

### 14.1 检索是离散的、不可微的

因为
$$
\sigma(j)=\arg\max_i s(i,j)
$$
是离散选择，所以整条链不能从最终图像损失直接反传回 memory 检索阶段。

### 14.2 不是端到端训练

三套网络分别训练，导致：

- 模块之间可能存在误差传播；
- 下游网络只能被动适应上游检索/对齐的偏差。

### 14.3 计算慢

若对每个 query segment 直接在同类 memory 中做暴力检索，复杂度大致是
$$
O\big(|\mathcal M_j|\cdot h\cdot w\cdot c\big).
$$

整张图有 \(m\) 个 segment，则整体检索量级大致是
$$
O\Big(
\sum_{j=1}^m |\mathcal M_j|\,h\,w\,c
\Big),
$$
这也是论文明确承认其实现比纯参数方法慢得多的原因。

### 14.4 Pairwise ordering 不一定全局一致

排序网络本质上学的是局部 front/back 判断。若多个片段同时重叠，pairwise 预测可能在理论上产生循环偏好。论文没有详细展开这部分的全局一致化策略，这属于实现层面的未完全形式化之处。

---

## 15. 今天回头看，这篇论文最有价值的地方

虽然这是 2018 年的工作，但它的思想今天看依然很前沿：

1. **它本质上是在做“视觉版 RAG”**  
   不是把所有知识压进参数，而是在推理时访问外部 memory。

2. **它把“生成”改造成“检索 + 对齐 + 融合”**  
   这对高保真图像任务尤其自然。

3. **它承认非参数库本身就是知识载体**  
   网络不用从头发明真实纹理，可以直接借用真实照片片段。

---

## 16. 一页纸总结

这篇论文的方法可以用下面四行式子概括：

### 第一步：检索
$$
\sigma(j)
=
\arg\max_{i\in\mathcal M_j}
\Big[
\operatorname{IoU}(P_i^{\mathrm{mask}},L_j^{\mathrm{mask}})
+
\operatorname{IoU}(P_i^{\mathrm{cont}},L_j^{\mathrm{cont}})
\Big].
$$

### 第二步：对齐
$$
\tilde P_{\sigma(j)}
=
T_{\theta_T}\big(\mathcal T_j,\;P_{\sigma(j)}^{\mathrm{color}}\big).
$$

### 第三步：合成 canvas
$$
C^{(t)}(x)
=
M_{\pi(t)}(x)\tilde P_{\pi(t)}(x)
+
\big(1-M_{\pi(t)}(x)\big)C^{(t-1)}(x).
$$

### 第四步：最终生成
$$
\hat I
=
f_{\theta_f}(C,L,m),
$$
并用感知损失训练
$$
L_f(\theta_f)
=
\sum_{(I,L)\in\mathcal D}
\sum_{l\in\mathcal S}
\lambda_l
\left\|
\Phi_l(I)-\Phi_l\big(f_{\theta_f}(C',L)\big)
\right\|_1.
$$

如果只用一句话总结这篇论文，那就是：

> 它不是让网络“无中生有”地生成整张图，而是先从真实世界里检索局部外观，再让网络负责把这些真实材料几何对齐、语义排序、边界融合，并补出缺失区域。

这正是它比同时期纯参数方法更真实的根本原因。
