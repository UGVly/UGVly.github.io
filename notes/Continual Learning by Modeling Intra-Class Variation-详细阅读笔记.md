---
tags: [public]
category: 研究笔记
---

# Continual Learning by Modeling Intra-Class Variation：详细阅读笔记

## 0. 论文信息

- 题目：Continual Learning by Modeling Intra-Class Variation
- 作者：Longhui Yu, Tianyang Hu, Lanqing Hong, Zhen Liu, Adrian Weller, Weiyang Liu
- 发表：TMLR 2023
- OpenReview：<https://openreview.net/forum?id=iDxfGaMYVr>
- arXiv：<https://arxiv.org/abs/2210.05398>
- 代码：<https://github.com/yulonghui/MOCA>

一句话概括：这篇论文的核心观点不是“继续学习时多做一点正则化”，而是“旧类样本太少，导致旧类表示空间的类内变化严重不足，进一步造成表示塌缩和梯度塌缩；只要把旧类表示的类内变化重新补出来，灾难性遗忘就会明显缓解”。

---

## 1. 我对全文主线的理解

作者的论证链条可以压缩成下面 5 步：

1. 记忆回放类 continual learning 方法虽然强，但旧类只保留了很少几个 prototype。
2. prototype 太少，旧类的类内变化远小于新类，也远小于 joint training。
3. 旧类表示因此会在表征空间里“收成一条线”甚至一个点，这就是 representation collapse。
4. 由于分类损失对表示的梯度本身由表示位置决定，表示塌缩会进一步诱导 gradient collapse。
5. 只要在表示空间里为旧类显式建模 intra-class variation，并把旧类特征朝“有信息的方向”做扰动，就能更接近 joint training 的梯度，从而减轻遗忘。

作者把这件事统一成一个框架：`MOCA = Modeling Intra-Class Variation`。

它有两大类实现：

- `model-agnostic MOCA`：不依赖当前网络参数，直接用参数分布在表示空间加扰动。
- `model-based MOCA`：利用当前模型本身产生扰动方向，让扰动更“懂模型”。

具体 5 个变体是：

- Gaussian
- vMF
- DOA-old / DOA-new
- VT
- WAP

其中，全文最强、最关键的变体是 `WAP`。

---

## 2. 问题设定与基本记号

### 2.1 continual learning 总目标

任务序列记为

$$
\mathcal{T}_1,\mathcal{T}_2,\dots,\mathcal{T}_t,
$$

第 \(t\) 个任务的数据分布为 \(\mathcal{D}^t\)，样本为 \((x^t,y^t)\sim\mathcal{D}^t\)。

设编码器为

$$
h_\theta:\mathcal{X}\to\mathbb{R}^d,
$$

分类头为

$$
g_\phi:\mathbb{R}^d\to\mathbb{R}^k.
$$

其中 \(k\) 是当前已见过的总类别数。论文把 continual learning 的总目标写成

$$
\min_{\theta,\phi}
\sum_{\tau=1}^{t}
\mathbb{E}_{(x^\tau,y^\tau)\sim \mathcal{D}^\tau}
\left[
\mathcal{L}\big(g_\phi(h_\theta(x^\tau)),y^\tau\big)
\right].
\tag{1}
$$

通常 \(\mathcal{L}\) 取交叉熵：

$$
\mathcal{L}_{\mathrm{ce}}(g_\phi(h_\theta(x)),e_y)
=
-\log
\frac{\exp(g_\phi(h_\theta(x))_y)}
{\sum_i \exp(g_\phi(h_\theta(x))_i)}.
$$

### 2.2 三种 continual learning 场景

#### 1. offline memory-based continual learning

有 replay buffer \(\mathcal{M}\)，训练目标：

$$
\min_{\theta,\phi}
\mathbb{E}_{(x^t,y^t)\sim \mathcal{D}^t}
\left[
\mathcal{L}_{\mathrm{ce}}(g_\phi(h_\theta(x^t)),y^t)
\right]
+
\mathbb{E}_{(x^{old},y^{old})\in \mathcal{M}}
\left[
\mathcal{L}_{\mathrm{ce}}(g_\phi(h_\theta(x^{old})),y^{old})
\right].
\tag{2}
$$

#### 2. proxy-based continual learning

不能存旧样本，只存每个旧类的均值表示 \(\bar f_i\)：

$$
\min_{\theta,\phi}
\mathbb{E}_{(x^t,y^t)\sim \mathcal{D}^t}
\left[
\mathcal{L}_{\mathrm{ce}}(g_\phi(h_\theta(x^t)),y^t)
\right]
+
\sum_{i=1}^{k_{t-1}}
\mathbb{E}
\left[
\mathcal{L}_{\mathrm{ce}}(g_\phi(\bar f_i),y_i)
\right].
\tag{3}
$$

#### 3. online memory-based continual learning

目标形式与式 (2) 相同，但数据流式到达，每个样本只能使用一次。

---

## 3. 论文最重要的推导：为什么“建模类内变化”会帮助 continual learning

这一节是整篇论文真正的理论出发点。

### 3.1 交叉熵对表示的梯度：一步一步推

对单个样本 \((x,y)\)，令

$$
f = h_\theta(x)\in\mathbb{R}^d,\qquad
z = g_\phi(f)\in\mathbb{R}^k,\qquad
p = \mathrm{softmax}(z)\in\mathbb{R}^k.
$$

交叉熵可写为

$$
\mathcal{L}_{\mathrm{ce}}(z,y)
=
-\log p_y
=
-z_y + \log\sum_{j=1}^k e^{z_j}.
$$

于是对每个 logit \(z_j\)：

$$
\frac{\partial \mathcal{L}_{\mathrm{ce}}}{\partial z_j}
=
\frac{\partial}{\partial z_j}
\left(
-z_y + \log\sum_{\ell=1}^k e^{z_\ell}
\right)
=
-\mathbf{1}(j=y)+\frac{e^{z_j}}{\sum_{\ell=1}^k e^{z_\ell}}.
$$

也就是

$$
\nabla_z \mathcal{L}_{\mathrm{ce}}
=
p - e_y.
$$

再对表示 \(f\) 用链式法则：

$$
\nabla_f \mathcal{L}_{\mathrm{ce}}
=
J_{g_\phi}(f)^\top (p-e_y),
\tag{4a}
$$

其中 \(J_{g_\phi}(f)\) 是分类头 \(g_\phi\) 对输入特征 \(f\) 的 Jacobian。

如果分类头是最常见的线性层

$$
g_\phi(f)=Wf+b,
$$

则

$$
J_{g_\phi}(f)=W,
$$

从而

$$
\nabla_f \mathcal{L}_{\mathrm{ce}}
=
W^\top (p-e_y).
\tag{4b}
$$

对数据分布取期望，就得到论文中的结论：

$$
\frac{\partial \mathcal{L}_{\mathrm{ce}}}{\partial h_\theta(x)}
=
\mathbb{E}_{(f,y)\sim \tilde{\mathcal{D}}}
\left[
\big(\mathrm{Softmax}(g_\phi(f))-e_y\big)\cdot
\frac{\partial g_\phi(f)}{\partial f}
\right].
\tag{4}
$$

### 3.2 这一步告诉了我们什么

式 (4) 说明：反向传播到编码器的梯度，主要由三件事决定：

- 当前特征 \(f\)
- 标签 \(y\)
- 当前分类头参数 \(\phi\)

其中 \(\phi\) 相对容易估计或维护，而 continual learning 真正缺失的是旧类特征分布本身。

所以作者提出一个关键视角：

> continual learning 中如果想逼近 joint training 的梯度，本质上是在逼近旧类的条件特征分布 \(p(f\mid y)\)。

### 3.3 从“prototype replay”到“feature distribution replay”

设旧类 \(y\) 在理想 joint training 下的特征分布为

$$
f^y \sim \tilde{\mathcal{D}}(y,\theta).
$$

传统 replay 只保留有限个 prototype：

$$
f_M^y \in \{h_\theta(x_1^y),\dots,h_\theta(x_m^y)\}.
$$

MOCA 的做法是把真实旧类特征写成

$$
f^y = f_M^y + \Delta f^y,
$$

其中：

- \(f_M^y\) 是 replay buffer 中的 prototype
- \(\Delta f^y\) 是“真实类内变化”和 prototype 之间的偏移

于是原始 replay 其实只在逼近 \(f_M^y\)，而 MOCA 试图进一步逼近 \(\Delta f^y\) 的分布。

### 3.4 一个更严谨的补充解释

论文正文没有把这个地方写成正式命题，但可以自然补成下面这个结论。

定义梯度映射

$$
\Gamma_\phi(f,y)
:=
J_{g_\phi}(f)^\top
\big(
\mathrm{softmax}(g_\phi(f))-e_y
\big).
$$

则 joint training 下旧类 \(y\) 的期望梯度为

$$
G_y^{\mathrm{joint}}
=
\mathbb{E}_{f\sim \tilde{\mathcal{D}}(y,\theta)}
\left[\Gamma_\phi(f,y)\right].
$$

若 MOCA 所构造的近似分布记为 \(\hat{\mathcal{D}}(y,\theta)\)，则对应梯度为

$$
G_y^{\mathrm{MOCA}}
=
\mathbb{E}_{f\sim \hat{\mathcal{D}}(y,\theta)}
\left[\Gamma_\phi(f,y)\right].
$$

如果 \(\Gamma_\phi(\cdot,y)\) 对 \(f\) 是 \(L_\Gamma\)-Lipschitz 的，那么有

$$
\left\|
G_y^{\mathrm{MOCA}}-G_y^{\mathrm{joint}}
\right\|
\le
L_\Gamma\,
W_1\!\left(
\hat{\mathcal{D}}(y,\theta),
\tilde{\mathcal{D}}(y,\theta)
\right),
$$

其中 \(W_1\) 是 Wasserstein-1 距离。

这不是论文原文里的定理，而是从论文推导自然补出的一个数学解释：  
**只要你把旧类特征分布拟合得更像 joint training，旧类梯度统计量就会更像 joint training。**

---

## 4. 为什么作者坚持“建模表示空间”，而不是“建模原图空间”

作者给了三层理由，我把它整理成更清楚的逻辑：

### 4.1 表示空间维度更低，建模更容易

原始图像 \(x\) 的维度很高，而中间表示 \(f=h_\theta(x)\) 的维度通常显著更低。  
有限内存下，直接估计 \(p(x\mid y)\) 远比估计 \(p(f\mid y)\) 困难。

### 4.2 表示空间更规整

论文依赖了 neural collapse / hyperspherical uniformity 一系列经验与理论现象：

- 特征往往会向球面结构聚集
- 类间呈近似均匀分离
- 类内分布在单位球面附近更接近可参数化分布

这正是作者后面引入 hyperspherical perturbation、vMF 分布的依据。

### 4.3 表示空间有类间共享结构

图像空间里，“猫的变化”和“车的变化”很不一样；  
而在深层表示空间里，不同类别的局部变化方向往往更可比较，所以像 VT 这种“把新类变化转移给旧类”的做法才有机会成立。

---

## 5. MOCA 的另一个视角：它其实是一种隐式数据增强

论文把增强后的特征写成

$$
f_M^y+\Delta f^y.
$$

并定义一个“隐式增强样本”：

$$
\tilde x^*
:=
\arg\min_{\tilde x}
\left\|
h_\theta(\tilde x)-f_M^y-\Delta f^y
\right\|_F^2
=
\arg\min_{\tilde x}
\left\|
h_\theta(\tilde x)-h_\theta(x)-\Delta f^y
\right\|_F^2.
\tag{5}
$$

如果最小值能达到 0，就有

$$
h_\theta(\tilde x^*)
=
h_\theta(x)+\Delta f^y.
$$

于是：

- 在前向上，\(\tilde x^*\) 与 MOCA 增强特征对应同一个表示；
- 在“对表示 \(f\) 的梯度”层面，两者完全一致；
- 因而可以把 MOCA 理解成“一次性代表一大簇等价隐式增强样本”。

### 5.1 这里我认为需要补一条严格性说明

论文原话更接近“生成相同梯度”，但严格说：

- 它保证的是 **loss 对特征 \(f\) 的梯度相同**
- 不自动保证 **loss 对参数 \(\theta\) 的梯度完全相同**

因为对参数梯度还有一项

$$
\frac{\partial f}{\partial \theta}
=
\frac{\partial h_\theta(x)}{\partial \theta},
$$

而对隐式样本 \(\tilde x^*\) 则是

$$
\frac{\partial h_\theta(\tilde x^*)}{\partial \theta}.
$$

这两者一般不必相同。

所以更严谨的理解应当是：

> MOCA 提供的是“特征层面的隐式增强解释”，而不是“参数梯度严格等价定理”。

这不影响方法有效，但理解上最好分清楚。

---

## 6. MOCA 总框架：先做无约束扰动，再投到球面上

### 6.1 基本形式

MOCA 最终都把旧类 prototype 特征 \(h_\theta(x)\) 变成

$$
f = h_\theta(x)+\Delta f.
$$

作者希望这个 \(\Delta f\) 主要改变方向、尽量不改变特征范数，于是先引入一个无约束扰动 \(\tilde{\Delta}f\)，再做球面投影：

下面统一记

$$
\mathcal{P}_{\mathbb S}(v)=\frac{v}{\|v\|}
$$

为投影到单位球面的算子。

$$
f
=
\frac{\|h_\theta(x)\|}
{\|h_\theta(x)+\tilde{\Delta}f\|}
\big(h_\theta(x)+\tilde{\Delta}f\big).
$$

### 6.2 把球面扰动 \(\Delta f\) 完整展开

令

$$
h := h_\theta(x),\qquad
\tilde\Delta := \tilde{\Delta}f.
$$

则

$$
f
=
\frac{\|h\|}{\|h+\tilde\Delta\|}(h+\tilde\Delta).
$$

因此

$$
\Delta f
=
f-h
=
\frac{\|h\|}{\|h+\tilde\Delta\|}(h+\tilde\Delta)-h.
$$

把它通分：

$$
\Delta f
=
\frac{\|h\|h+\|h\|\tilde\Delta-\|h+\tilde\Delta\|h}
{\|h+\tilde\Delta\|}.
$$

整理得

$$
\Delta f
=
\frac{
\big(\|h\|-\|h+\tilde\Delta\|\big)h
+
\|h\|\tilde\Delta
}
{\|h+\tilde\Delta\|}.
\tag{6}
$$

这就是论文里 “hyperspherical augmentation” 的显式展开式。

### 6.3 为什么这个投影有意义

由定义立刻得到

$$
\|f\|
=
\left\|
\frac{\|h\|}{\|h+\tilde\Delta\|}(h+\tilde\Delta)
\right\|
=
\|h\|.
$$

也就是说：

- 扰动不改变特征范数
- 它只改变特征在球面上的角位置
- 所以它更像“语义方向扰动”，而不是无意义的尺度放大

### 6.4 角度与 \(\lambda\) 的显式关系

这一点论文只做了实验图，我把公式补出来。

令

$$
u = \mathcal{P}_{\mathbb S}(h),\qquad
r\in\mathbb{S}^{d-1},\qquad
f = \|h\|\cdot \mathcal{P}_{\mathbb S}(u+\lambda r).
$$

设 \(f\) 与原方向 \(u\) 的夹角为 \(\alpha\)，则

$$
\cos\alpha
=
u^\top \frac{u+\lambda r}{\|u+\lambda r\|}
=
\frac{1+\lambda\,u^\top r}
{\sqrt{1+2\lambda\,u^\top r+\lambda^2}}.
\tag{7}
$$

这个式子非常关键，因为它说明：

- 同样的 \(\lambda\)，真实角扰动不只取决于大小，还取决于方向 \(u^\top r\)
- 若 \(r\) 与 \(u\) 同向，则角度很小
- 若 \(r\) 更接近切空间方向，则角度更大
- 所以“方向是否有信息”比“方差够不够大”更重要

这其实正好解释了论文实验里一个核心结论：  
**WAP 往往比 Gaussian 更好，不是因为它噪声更大，而是因为它方向更对。**

### 6.5 小扰动的一阶近似

若 \(u\) 已经单位化，且 \(\delta\) 很小，则

$$
\mathcal{P}_{\mathbb S}(u+\delta)
=
\frac{u+\delta}{\|u+\delta\|}
\approx
u + (I-uu^\top)\delta + O(\|\delta\|^2).
\tag{8}
$$

这说明球面投影会自动去掉径向分量，只保留切向扰动。  
换句话说，MOCA 真正在用的是“球面切空间上的语义扰动”。

---

## 7. MOCA 的反向传播视角：它本质上是“梯度增强”

论文把增强特征写成

$$
f = h_\theta(x)+\Delta f.
$$

于是若定义增强损失

$$
\mathcal{L}_{\mathrm{MOCA}}(\theta,\phi)
:=
\mathcal{L}_{\mathrm{ce}}(g_\phi(f),y),
$$

那么按微分有

$$
df = d(h_\theta(x)) + d(\Delta f).
$$

论文把这个效果概括成

$$
\frac{\partial \mathcal{L}_{\mathrm{ce}}}{\partial f}
=
\frac{\partial \mathcal{L}_{\mathrm{ce}}}{\partial h_\theta(x)}
+
\frac{\partial \mathcal{L}_{\mathrm{ce}}}{\partial \Delta f}.
$$

更严谨地说，如果把 \(\Delta f\) 也看成 \(h_\theta(x)\) 的函数，那么真正的总导数应为

$$
\frac{d\mathcal{L}}{dh}
=
\frac{\partial \mathcal{L}}{\partial f}
\left(
I+\frac{\partial \Delta f}{\partial h}
\right).
$$

所以 MOCA 的本质是：

- 训练点不再是原始旧类 prototype 所在的位置
- 而是它附近一簇被人为拓展后的球面邻域
- 于是旧类梯度的方向变丰富了，gradient collapse 被缓解

---

## 8. Model-agnostic MOCA

### 8.1 Gaussian

记旧类 prototype 为 \(x^{old}\)，对应特征为 \(h_\theta(x^{old})\)。  
Gaussian 版本定义为

$$
f
=
\left\|h_\theta(x^{old})\right\|
\cdot
\mathcal{P}_{\mathbb S}
\Big(
\mathcal{P}_{\mathbb S}(h_\theta(x^{old}))
+
\lambda \epsilon
\Big),
\qquad
\epsilon\sim\mathcal{N}(0,I).
\tag{9}
$$

这里 \(\lambda\) 同时扮演：

- 噪声尺度
- 扰动角度控制参数

的角色。

由于球面投影的存在，真正起作用的是噪声在切空间里的分量，而不是完整高斯向量本身。

### 8.2 vMF

作者认为既然是在球面上建模类内变化，那么 von Mises-Fisher 分布比普通高斯更自然。

其密度为

$$
p(\epsilon\mid \mu,\kappa)
=
\frac{\kappa^{d/2-1}}
{(2\pi)^{d/2} I_{d/2-1}(\kappa)}
\exp(\kappa\mu^\top \epsilon),
\qquad
\mu = \mathcal{P}_{\mathbb S}(h_\theta(x^{old})).
\tag{10}
$$

然后仍使用

$$
f
=
\|h_\theta(x^{old})\|
\cdot
\mathcal{P}_{\mathbb S}
\Big(
\mathcal{P}_{\mathbb S}(h_\theta(x^{old}))
+
\lambda \epsilon
\Big).
\tag{11}
$$

其中：

- \(\kappa\) 越大，分布越集中到均值方向 \(\mu\)
- \(\kappa=0\) 时退化成球面均匀分布

### 8.3 Gaussian 与 vMF 的本质区别

我的理解是：

- Gaussian 是“先在欧氏空间加噪，再投球面”
- vMF 是“直接在球面上定义方向分布”

因此 vMF 在控制真实角度扰动时更自然，作者也在附录里明确说：  
**最佳超参下，vMF 的最优性能普遍略优于 Gaussian，但 Gaussian 实现更简单、更高效。**

---

## 9. Model-based MOCA

这是论文真正有意思的部分。

### 9.1 总体思想

作者认为只用固定分布加噪声太“盲”了，因为真实特征分布是依赖当前模型参数 \(\theta\) 的。  
所以他们希望扰动方向本身也由模型给出。

论文写了两个一般形式：

$$
\text{Perturbation I:}\quad
\tilde{\Delta}f
=
\lambda_1 h_\theta(x)-\lambda_2 h_{\theta+\Delta\theta}(x),
$$

$$
\text{Perturbation II:}\quad
\tilde{\Delta}f
=
\lambda_1 h_\theta(x)-\lambda_2 h_\theta(x+\Delta x).
\tag{12}
$$

### 9.2 一个必须指出的记号问题

这里论文正文有一个小但真实存在的不严谨点：

- 式 (12) 写的是“原特征减去扰动后特征”
- 但后面 DOA / WAP / VT 的具体公式全部变成了“原方向加上某个归一化方向”

尤其结合附录 D.2 “朝向新类流形的方向比反方向更有效”的实验结论来看，  
**作者真正实现和想表达的，是“把旧类特征朝某个 model-conditioned 方向推过去”，而不是机械地使用式 (12) 的减法记号。**

所以读这篇论文时，应该以具体算法式和实验解释为准，而不是死扣式 (12) 的符号。

### 9.3 DOA：Dropout-based Augmentation

Dropout 对网络参数施加随机 mask，本质上对应一个随机模型扰动。  
作者把它解释成 feature augmentation：

$$
\tilde{\Delta}f
=
\lambda_1 h_\theta(x)-\lambda_2 h_{\mathrm{Dropout}(\theta)}(x).
$$

最后写成

$$
f
=
\left\| h_\theta(x^{old}) \right\|
\cdot
\mathcal{P}_{\mathbb S}
\left(
\mathcal{P}_{\mathbb S}\big(h_\theta(x^{old})\big)
+
\lambda\cdot
\mathcal{P}_{\mathbb S}\big(h_{\mathrm{Dropout}(\theta)}(x)\big)
\right).
\tag{13}
$$

其中有两个版本：

- `DOA-old`：\(x=x^{old}\)
- `DOA-new`：\(x=x^{new}\)

这两个版本的差异非常重要：

- DOA-old 只利用旧类自己的 dropout 随机性
- DOA-new 直接把新类更丰富的表示结构拿来帮助旧类分散开

这也解释了为什么实验里几乎总是 `DOA-new > DOA-old`。

#### 一阶近似理解

若把 dropout 对网络的影响记成参数扰动 \(\Delta\theta_{\mathrm{drop}}\)，则

$$
h_{\theta+\Delta\theta_{\mathrm{drop}}}(x)
\approx
h_\theta(x)
+
J_\theta h_\theta(x)\,\Delta\theta_{\mathrm{drop}}.
$$

所以 DOA 本质上是在用

$$
J_\theta h_\theta(x)\,\Delta\theta_{\mathrm{drop}}
$$

做随机但“模型相关”的特征方向扩张。

### 9.4 WAP：Weight-Adversarial Perturbation

这是全文最重要的变体。

作者不再随机扰动 \(\theta\)，而是寻找一个最容易把旧类样本“误导成新类”的参数扰动：

$$
f
=
\left\| h_\theta(x^{old}) \right\|
\cdot
\mathcal{P}_{\mathbb S}
\left(
\mathcal{P}_{\mathbb S}\big(h_\theta(x^{old})\big)
+
\lambda\cdot
\mathcal{P}_{\mathbb S}\big(h_{\theta+\Delta\theta}(x^{old})\big)
\right),
\tag{14}
$$

其中 \(\Delta\theta\) 由下面的对抗问题产生：

$$
\Delta\theta
=
\arg\min_{\|\Delta\theta\|\le \epsilon}
\mathcal{L}_{\mathrm{ce}}
\left(
g_\phi(h_{\theta+\Delta\theta}(x^{old})),
y^{new}
\right).
\tag{15}
$$

这里 \(y^{new}\) 是随机采样的新类标签。

#### 这一步真正做了什么

式 (15) 的含义不是“保持旧类不变”，而是反过来：

- 人为找一组参数扰动
- 让旧类样本在该扰动模型下更像新类
- 即，把旧类样本推向 old/new decision boundary 的混淆方向

然后再拿这个“最容易混淆的方向”去增强旧类特征，并仍然用旧类标签训练。

这就会产生一个很自然的效果：

- 模型被迫把旧类在这个危险方向上也分开
- 等价于增大 old/new margin

#### 一阶泰勒展开解释

若 \(\Delta\theta\) 足够小，则

$$
h_{\theta+\Delta\theta}(x^{old})
\approx
h_\theta(x^{old})
+
J_\theta h_\theta(x^{old})\,\Delta\theta.
\tag{16}
$$

因此 WAP 其实是在寻找一个受约束的 \(\Delta\theta\)，使得

$$
J_\theta h_\theta(x^{old})\,\Delta\theta
$$

朝向“最容易把旧类判成新类”的局部方向。  
这比 Gaussian 的各向同性噪声明显更有判别性。

### 9.5 WAP 的算法步骤

根据附录 Algorithm 1，WAP 的训练流程可以概括为：

1. 从旧类 buffer 采一批 \(B_m\)。
2. 复制当前模型，得到 proxy model \(\theta_{adv}\)。
3. 用 PGD 在 \(\|\Delta_{B_m}\|_2\le \epsilon\) 约束内迭代更新 proxy 扰动。
4. 每一步都随机采新的新类标签 \(y^{adv}\)，最小化旧样本被分到这些新类上的交叉熵。
5. 得到对抗 proxy 模型 \(\theta_{adv}=\theta+\Delta_{B_m}\)。
6. 用 \(\theta_{adv}\) 产生旧类增强特征 \(f_i\)。
7. 对增强旧类样本仍使用旧标签 \(y_i^{old}\) 计算损失。
8. 再与真实新类 batch 的损失一起更新主模型。

实验中关键超参：

- inner lr \(\zeta=10\)
- inner iteration \(T=1\)
- offline 场景下 \(\lambda=2.0\)

### 9.6 VT：Variation Transfer

VT 不从参数扰动入手，而是直接把新类样本的类内残差转给旧类。  
设新类均值对应的“虚拟样本”为 \(\tilde x^{new}\)，则

$$
f
=
\left\| h_\theta(x^{old}) \right\|
\cdot
\mathcal{P}_{\mathbb S}
\left(
\mathcal{P}_{\mathbb S}\big(h_\theta(x^{old})\big)
+
\lambda\cdot
\mathcal{P}_{\mathbb S}
\Big(
h_\theta(x^{new})-h_\theta(\tilde x^{new})
\Big)
\right).
\tag{17}
$$

并且

$$
h_\theta(\tilde x^{new})
=
\bar h_\theta(x^{new})
=
\mathbb{E}_{x^{new}} h_\theta(x^{new}).
$$

所以 VT 的核心假设是：

> 不同类别在表示空间里的类内变化结构有相似性，因此新类的局部变化方向可以迁移给旧类。

这个假设在 proxy-based setting 特别有价值，因为那时旧类原始样本根本不可用。

---

## 10. MOCA 与 large-margin softmax 的关系

附录 C 把 MOCA 与 large-margin softmax 联系起来。

设特征为 \(x_i\)，对应标签为 \(y_i\)，第 \(j\) 类分类器为 \(w_j\)，\(x_i\) 与 \(w_j\) 的夹角为 \(\theta_j\)。

论文写 large-margin 形式为

$$
\mathcal{L}_{\mathrm{Large\text{-}Margin}}
=
\sum_i
\frac{
\exp\big(\|w_{y_i}\|\,\|x_i\|\,\cos(\theta_{y_i}+\Delta\theta)\big)
}{
\sum_j
\exp\big(
\|w_j\|\,\|x_i\|\,\cos(\theta_j+\mathbf{1}(j=y_i)\Delta\theta)
\big)
}.
\tag{18}
$$

MOCA 对应形式写成

$$
\mathcal{L}_{\mathrm{MOCA}}
=
\sum_i
\frac{
\exp\big(\|w_{y_i}\|\,\|x_i\|\,\cos(\theta_{y_i}+\Delta\theta_{y_i})\big)
}{
\sum_j
\exp\big(
\|w_j\|\,\|x_i\|\,\cos(\theta_j+\Delta\theta_j)
\big)
}.
\tag{19}
$$

作者想表达的是：

- large-margin softmax 只显式改动目标类角度
- MOCA 通过扰动特征，会同时诱导所有类别角度一起变化
- 但主要变化仍集中在目标类对应角度上

### 10.1 这里也有一个记号问题

严格说，式 (18)(19) 写出来的是“目标类 softmax 概率”，而不是通常意义上的 cross-entropy loss；  
真正的 CE 还应再套一个 \(-\log\)。

不过因为 \(-\log\) 是单调变换，作者的直觉并不会因此彻底失效：  
MOCA 依然可以被理解为一种“用虚拟困难样本扩展 margin”的机制。

我的理解是：

> 这一节更像是几何类比，而不是严格定理。

---

## 11. 论文里的实验设置

### 11.1 数据集与任务划分

- CIFAR-10：5 个任务，每个任务 2 类
- CIFAR-100：5 个任务，每个任务 20 类
- TinyImageNet：10 个任务，每个任务 20 类
- online 额外使用 MiniImageNet

### 11.2 backbone 与评测

- backbone：ResNet18
- 指标：最终 continual task 结束后的 full test accuracy
- 所有方法都对特征做 hyperspherical projection，让分类头更依赖角度而不是范数

### 11.3 关键超参

- offline：\(\lambda=2.0\)
- online：\(\lambda=0.8\)
- proxy-based：\(\lambda=1.0\)
- DOA dropout rate：0.5
- WAP：\(\zeta=10\), \(T=1\)
- offline batch size：32
- online batch size：10
- offline training epoch：50

---

## 12. 实验结果：不同 MOCA 变体的直接比较

作者先在 CIFAR-100 上用 ER 做统一底座，比较所有变体。

| Setting | Baseline | Gaussian | vMF | DOA-old | DOA-new | VT | WAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Offline | 31.08 | 37.29 | 38.76 | 33.67 | 38.75 | 39.78 | 41.02 |
| Online | 31.90 | 32.78 | 31.25 | 30.20 | 29.48 | 32.55 | 33.72 |
| Proxy | 31.26 | 42.54 | 42.24 | - | 45.72 | 46.77 | - |

### 12.1 这里最重要的结论

- 大多数 MOCA 变体都明显优于 baseline。
- `model-based` 通常优于 `model-agnostic`。
- offline / online 下最强的是 `WAP`。
- proxy-based 下不能用 WAP，最强的是 `VT`。
- `DOA-new > DOA-old`，说明“借助新类流形的信息”是有帮助的。
- 仅仅扩大方差还不够，**扰动方向** 才是决定效果的关键。

---

## 13. 扰动强度 \(\lambda\) 与扰动角度的影响

Figure 8 的结论可以概括成：

- 大部分变体在较宽的 \(\lambda\) 范围内都能稳定优于 baseline。
- model-based 变体整体优于 model-agnostic 变体。
- WAP 在几乎所有扰动强度与角度约束下都最稳、最好。

结合前面的角度公式

$$
\cos\alpha
=
\frac{1+\lambda u^\top r}{\sqrt{1+2\lambda u^\top r+\lambda^2}},
$$

可以看出这组实验其实在验证两件事：

1. 需要足够的角扩张，旧类才不会塌成一团；
2. 但更重要的是 \(r\) 不能乱来，否则会把特征推到无意义方向。

附录中作者还用 angular fisher score 评估过 \(\lambda\)：

| Method | \(\lambda=0\) | \(\lambda=1\) | \(\lambda=2\) | \(\lambda=3\) | \(\lambda=4\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ER w/ Gaussian | 1.48 | 1.29 | 1.07 | 1.08 | 1.38 |
| ER w/ WAP | 1.48 | 1.09 | 1.08 | 2.54 | NaN |

作者据此认为 \(\lambda=2\) 附近最好：  
太小，MOCA 不起作用；太大，会破坏收敛。

---

## 14. 分类器可分性分析

作者统计了不同任务之间 classifier weight 的平均夹角。Figure 9 的结论是：

- Gaussian 和 WAP 都能增大不同任务分类器之间的角度
- 分类器角度更大，意味着分类器更可分
- 在角度型分类器下，这通常也对应表示空间的类间分离更强

所以 MOCA 不只是“让旧类更散”，它还进一步提升了最终判别边界的区分性。

---

## 15. Offline continual learning 主结果

下面把 Table 2 拆开抄一遍，便于以后查。

### 15.1 CIFAR-10

| Method | M=200 | M=500 | M=2000 |
| --- | ---: | ---: | ---: |
| GEM | 29.99±3.92 | 29.45±5.64 | 27.20±4.50 |
| GSS | 38.62±3.59 | 48.97±3.25 | 60.40±4.92 |
| iCaRL | 32.44±0.93 | 34.95±1.23 | 33.57±1.65 |
| ER | 49.07±1.65 | 61.58±1.12 | 76.89±0.99 |
| ER + Gaussian | 61.52±1.42 | 68.54±2.01 | 78.27±0.52 |
| ER + WAP | 63.12±2.15 | 72.07±1.37 | 80.38±0.95 |
| DER++ | 64.88±1.17 | 72.70±1.36 | 78.54±0.97 |
| DER++ + Gaussian | 63.02±0.53 | 71.04±0.72 | 79.22±0.42 |
| DER++ + WAP | 65.12±0.77 | 75.01±0.24 | 81.54±0.12 |
| ER-ACE | 63.18±0.56 | 71.98±1.30 | 80.01±0.76 |
| ER-ACE + Gaussian | 65.21±0.89 | 72.01±0.76 | 78.92±0.58 |
| ER-ACE + WAP | 66.56±0.81 | 72.86±1.02 | 80.24±0.50 |

#### 我的读法

- 对最简单的 ER，WAP 增益极大：`+14.05 / +10.49 / +3.49`
- 对 DER++，Gaussian 甚至在 M=200/500 下不如原始 DER++，说明“各向同性扩张”与 distillation 机制并不天然兼容
- WAP 对三条 baseline 都是稳定正增益

### 15.2 CIFAR-100

| Method | M=200 | M=500 | M=2000 |
| --- | ---: | ---: | ---: |
| GEM | 20.75±0.66 | 25.54±0.65 | 37.56±0.87 |
| GSS | 19.42±0.29 | 21.92±0.34 | 27.07±0.25 |
| iCaRL | 28.00±0.91 | 33.25±1.25 | 42.19±2.42 |
| ER | 22.14±0.42 | 31.02±0.79 | 43.54±0.59 |
| ER + Gaussian | 27.51±0.93 | 37.54±0.71 | 49.61±1.01 |
| ER + WAP | 30.16±1.02 | 40.24±0.78 | 52.92±0.03 |
| DER++ | 29.68±1.38 | 39.08±1.76 | 54.38±0.86 |
| DER++ + Gaussian | 30.59±0.40 | 40.52±0.29 | 53.70±0.42 |
| DER++ + WAP | 32.18±0.67 | 43.78±0.89 | 55.04±0.81 |
| ER-ACE | 35.09±0.92 | 43.12±0.85 | 53.88±0.42 |
| ER-ACE + Gaussian | 37.01±0.70 | 44.57±0.83 | 54.84±0.12 |
| ER-ACE + WAP | 37.46±0.77 | 45.79±0.73 | 56.02±0.64 |

#### 我的读法

- 这是论文最有说服力的一组结果。
- ER + WAP 相对 ER 的提升高达：`+8.02 / +9.22 / +9.38`
- DER++ + WAP 与 ER-ACE + WAP 也都有稳定收益
- 即便 baseline 已经很强，WAP 依然有效

### 15.3 TinyImageNet

| Method | M=200 | M=500 | M=2000 |
| --- | ---: | ---: | ---: |
| GEM | - | - | - |
| GSS | 8.57±0.13 | 9.63±0.14 | 11.94±0.17 |
| iCaRL | 5.50±0.52 | 11.00±0.55 | 18.10±1.13 |
| ER | 8.65±0.16 | 10.05±0.28 | 18.19±0.47 |
| ER + Gaussian | 9.42±0.12 | 12.94±0.52 | 21.43±0.78 |
| ER + WAP | 10.41±0.37 | 16.27±0.25 | 22.62±0.10 |
| DER++ | 10.96±1.17 | 19.38±1.41 | 30.11±0.57 |
| DER++ + Gaussian | 10.52±0.12 | 15.75±0.35 | 25.28±0.30 |
| DER++ + WAP | 12.07±0.35 | 21.24±0.47 | 29.33±0.71 |
| ER-ACE | 14.29±0.74 | 20.87±0.69 | 30.10±0.92 |
| ER-ACE + Gaussian | 16.72±0.41 | 22.82±0.39 | 30.92±0.41 |
| ER-ACE + WAP | 17.05±0.22 | 23.56±0.85 | 32.54±0.72 |

#### 我的读法

- 复杂数据集上，WAP 依然普遍有效
- 但不是所有 baseline 上都一定“碾压式”提升
- 例如 DER++ 在 M=2000 上本身就很强，WAP 反而略低于原始 DER++（29.33 vs 30.11）
- 这说明 MOCA 不是无条件单调增强，而是一个很强但仍有适用边界的 plug-in

### 15.4 作者自己的结论，我认为是成立的

- Gaussian 对 ER 很有效，但对 DER++ 兼容性较差
- WAP 更稳定，因为它只沿“最有信息、最靠边界”的方向做扰动
- 中等内存规模时，MOCA 最能发挥价值

---

## 16. Online continual learning 主结果

### 16.1 CIFAR-10 / CIFAR-100 / MiniImageNet

| Method | C10 M=20 | C10 M=100 | C100 M=20 | C100 M=100 | Mini M=20 | Mini M=100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-GEM | 18.56 | 18.60 | 3.50 | 3.26 | 2.94 | 3.04 |
| MIR | 24.20 | 45.44 | 12.70 | 17.50 | 11.54 | 11.48 |
| SS-IL | 35.54 | 42.78 | 16.20 | 26.24 | 16.96 | 24.38 |
| iCaRL | 40.94 | 49.76 | 17.55 | 19.86 | 12.30 | 15.20 |
| ER | 31.90 | 42.48 | 13.52 | 25.24 | 16.16 | 21.62 |
| ER + WAP | 33.72 | 41.26 | 13.90 | 23.29 | 17.22 | 19.52 |
| DER++ | 34.36 | 43.38 | 12.84 | 13.74 | 17.00 | 18.56 |
| DER++ + WAP | 41.56 | 45.64 | 18.76 | 22.54 | 16.32 | 19.20 |
| ER-ACE | 42.90 | 53.88 | 16.88 | 27.48 | 21.00 | 28.96 |
| ER-ACE + WAP | 43.57 | 54.42 | 18.90 | 29.52 | 22.56 | 29.70 |

### 16.2 我对 online 结果的解读

- 大方向上，WAP 仍然大多有效。
- 但收益显著弱于 offline。
- 有些 setting 甚至回退：
  - ER 在 CIFAR-10 M=100 下下降 1.22
  - ER 在 CIFAR-100 M=100 下下降 1.95
  - ER 在 MiniImageNet M=100 下下降 2.10
  - DER++ 在 MiniImageNet M=20 下也略降

这和作者的解释一致：

> online continual learning 不仅有 forgetting，还有 “新数据只看一次带来的 underfitting”；MOCA 主要缓解前者，不解决后者。

所以如果把这篇论文往 online 场景用，最好带着这个预期：  
**它更像一个 forgetting reducer，而不是全能的 online learner。**

---

## 17. Proxy-based continual learning 主结果

| Method | C100 T=5 | C100 T=10 | C100 T=20 | Mini T=5 | Mini T=10 | Mini T=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PASS | 64.39 | 57.36 | 58.09 | 48.26 | 46.54 | 42.09 |
| PASS + DOA | 66.82 | 63.30 | 62.62 | 47.92 | 47.55 | 47.11 |
| PASS + VT | 67.75 | 63.64 | 63.09 | 48.35 | 47.90 | 47.33 |

### 17.1 结论

- WAP 不能用于 proxy-based，因为旧类样本本身不可得。
- VT 成了最强变体。
- VT 相对 PASS 的提升：
  - CIFAR-100：`+3.36 / +6.28 / +5.00`
  - MiniImageNet：`+0.09 / +1.36 / +5.24`

这组结果很重要，因为它说明：

> 即便完全没有旧图像，只要还能在表示空间里合理迁移“类内变化方向”，continual learning 依然能变好。

---

## 18. Performance over tasks 与梯度多样性实验

### 18.1 Figure 10：随任务推进的平均测试精度

作者在 CIFAR-100 和 TinyImageNet 上画了训练过程中每一步的平均测试准确率。  
结论非常稳定：

- 所有 MOCA 变体几乎都在整条训练曲线上压过 ER baseline
- WAP 全程最好
- DOA-old 虽然优于 baseline，但明显不如 DOA-new

这再次支持一个核心命题：  
**仅靠对旧类自身做随机扰动不够，借助新类流形信息更关键。**

### 18.2 Figure 3 / Figure 12：梯度奇异值谱

作者把“梯度是否塌缩”具体化为训练梯度矩阵的奇异值分布：

- baseline ER 的奇异值衰减很快，说明梯度主要集中在少数方向
- Gaussian 与 WAP 都能让奇异值谱更平、更丰富
- 更大的 memory size 会让 ER 的谱更像 joint training

附录 Figure 12 进一步显示：

- DOA-new、VT、WAP 的梯度多样性明显优于 DOA-old
- 能利用新类信息的方法，确实更接近 joint training 的梯度结构

这组实验我认为是全文最能“撑住核心假设”的证据之一。

---

## 19. 与其他“多样化梯度/表示”的经典方法比较

作者把 MOCA 和 Re-weighting、Focal Loss、Manifold Mixup 都放在 ER 上比较。

| Method | C10 k=200 | C10 k=500 | C10 k=2000 | C100 k=200 | C100 k=500 | C100 k=2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ER | 49.07 | 61.58 | 76.89 | 21.71 | 28.12 | 43.10 |
| + Re-weighting | 53.02 | 66.54 | 77.92 | 24.58 | 30.12 | 44.31 |
| + Focal Loss | 46.07 | 60.97 | 77.26 | 22.43 | 27.19 | 43.37 |
| + Manifold Mixup | 55.21 | 67.02 | 77.54 | 23.97 | 29.33 | 45.21 |
| + Model-based MOCA (WAP) | 63.12 | 72.07 | 80.38 | 30.16 | 40.24 | 52.92 |

结论非常清楚：

- WAP 全列最优
- 说明 MOCA 的收益并不只是“普通重加权”或“普通表示增强”带来的
- 它确实更贴近 continual learning 的特殊痛点：旧类梯度塌缩

---

## 20. 附录消融：方向真的比方差更重要

作者做了一个非常有解释力的实验：  
直接把“新类特征”加到旧类特征上，或者反过来减掉它。

| Method | Perturbed Angle | Original Angle | Accuracy |
| --- | ---: | ---: | ---: |
| Baseline | - | 72.51 | 29.94 |
| Minus New Feature | 90.12 | 70.91 | 27.35 |
| Add New Feature | 71.34 | 77.58 | 32.60 |

我的理解：

- “减去新类特征”当然也扩大了旧类方差
- 但它把旧类朝远离 decision boundary 的方向推开了
- 于是产生的训练信号不够 informative

而“加上新类特征”恰好相反：

- 先把旧类推向容易与新类混淆的区域
- 再用旧标签训练
- 这就逼模型学出更大的 margin

这几乎就是 WAP 设计哲学的最直接实证版。

---

## 21. 附录消融：memory size、稳定性、分类头与计算开销

### 21.1 memory size

| Method | k=50 | k=200 | k=2000 | k=20000 |
| --- | ---: | ---: | ---: | ---: |
| ER | 19.94 | 22.14 | 43.54 | 66.39 |
| + Gaussian | 23.56 | 27.51 | 49.61 | 67.34 |
| + WAP | 25.12 | 30.16 | 52.92 | 67.95 |

结论：

- MOCA 在小、中、大 buffer 下都有效
- 最显著收益出现在中等 buffer（200 到 2000）
- buffer 极小时，prototype 覆盖不够，扰动质量受限
- buffer 极大时，问题已接近 joint training，MOCA 的边际收益自然变小

### 21.2 WAP 的 \(\zeta\) 稳定性

| Method | \(\zeta=0.1\) | \(\zeta=5\) | \(\zeta=10\) | \(\zeta=50\) |
| --- | ---: | ---: | ---: | ---: |
| ER + WAP | 24.52 | 27.51 | 30.16 | 28.14 |

结论：

- WAP 对 \(\zeta\) 不算特别敏感
- 从 0.1 到 50 都比 ER baseline 22.14 好
- \(\zeta=10\) 最佳
- 作者还提到若把 inner iteration 从 \(T=1\) 提到 \(T=2\)，CIFAR-100 上 ER-WAP 可从 30.16 升到 30.92

### 21.3 hyperspherical classifier

WAP 在普通分类头与球面分类头下：

| Method | k=200 | k=500 |
| --- | ---: | ---: |
| ER | 22.14 | 31.02 |
| + WAP (normal classifier) | 29.33 | 39.25 |
| + WAP (hyperspherical classifier) | 30.16 | 40.24 |

baseline ER 的分类头影响：

| Method | C10 k=200 | C10 k=500 | C100 k=200 | C100 k=500 |
| --- | ---: | ---: | ---: | ---: |
| ER (normal classifier) | 49.54 | 61.97 | 21.92 | 30.14 |
| ER (hyperspherical classifier) | 49.07 | 61.58 | 22.14 | 31.02 |

作者的意思是：

- 对 baseline，本身分类头形式影响不大
- 但对 MOCA，球面分类头更稳，因为它去掉了特征范数扰动的干扰

### 21.4 perturb weight vs perturb feature

| Method | k=200 | k=500 |
| --- | ---: | ---: |
| ER | 22.14 | 31.02 |
| + FGSM | 21.54 | 29.87 |
| + WAP | 30.16 | 40.24 |

这组实验非常关键：

- 直接 adversarially perturb feature 是不行的，甚至会伤害性能
- 说明 WAP 真正有用的点不只是“对抗扰动”四个字
- 而是“通过扰动编码器权重，得到考虑 feature manifold 的高信息方向”

### 21.5 计算开销

| Method | Training time (s) | Accuracy (%) |
| --- | ---: | ---: |
| ER | 5562 | 22.14 |
| + Gaussian | 6147 | 27.51 |
| + WAP | 7109 | 30.16 |

简单换算：

- Gaussian 相对 ER 训练时间增加约 `10.5%`
- WAP 相对 ER 训练时间增加约 `27.8%`

但在 CIFAR-100 / buffer=200 上，WAP 带来 `+8.02` 个点的精度提升。  
这个性价比我认为是相当不错的。

---

## 22. 我认为这篇论文最有价值的 6 个结论

### 22.1 “旧类类内变化不足”是 continual learning 中一个非常具体、可操作的中层问题

这比泛泛地说“灾难性遗忘来自分布漂移/参数覆盖”更可落地，因为它直接指向了表示分布和梯度分布。

### 22.2 表示塌缩会诱发梯度塌缩

作者没有给出很重的理论，但实验上这件事是被清楚验证了的。  
这也是为什么他们始终围绕 feature-level augmentation，而不是只做参数正则。

### 22.3 方向比方差更重要

这是全文最核心的经验发现：

- Gaussian 说明“方差补一点”确实有帮助
- WAP 说明“朝边界方向补”更有效
- 附录的 add / minus new feature 实验证明“方向不对，扩大方差也可能没用甚至有害”

### 22.4 model-based variation 往往优于 model-agnostic variation

如果你要在这篇论文里挑一个最值得复用的思想，那就是：

> 用当前模型来决定旧类应该往哪里扩，而不是用一个盲噪声分布乱扩。

### 22.5 proxy-based 场景里，VT 的价值很高

这说明作者的框架并不依赖某一个具体技巧，而是一个更普适的“在表示空间补类内变化”的思路。

### 22.6 WAP 最像“continual learning 版的困难样本构造”

它做的事其实非常像 large-margin / adversarial training / hard example mining 的结合体，只不过发生在 continual learning 的 old/new imbalance 场景下。

---

## 23. 我认为这篇论文的局限与需要谨慎的地方

### 23.1 理论部分更多是“动机推导”，不是严格收敛理论

它说明了为什么 feature distribution 很关键，但没有证明某个具体变体一定逼近 joint training 最优。

### 23.2 文中确实有几处公式/表述不够严谨

我在阅读时标出的主要有三处：

- 式 (12) 的符号与后续 DOA/WAP/VT 具体公式不完全一致
- 附录 C 的“loss”写成了 softmax 概率形式，少了 \(-\log\)
- Section 3.3 的“same gradient”更准确地说应是“same feature-level gradient”

这些不影响主实验可信度，但复现或二次推导时最好自己修正理解。

### 23.3 WAP 不是处处单调增益

尤其 online / 某些强 baseline / 某些大 buffer setting 下，会出现回退。  
所以它更像一个很强的经验增益模块，而非理论上保证改进的插件。

### 23.4 WAP 无法直接用于 proxy-based setting

因为它需要旧类真实样本来构造 adversarial proxy model。

### 23.5 论文没有把“为什么旧类变化近似可由新类变化迁移”讲到很深

VT 的假设经验上有效，但仍偏启发式。

---

## 24. 如果我要复现或扩展这篇工作，我会怎么做

### 24.1 复现优先级

如果只是想复现实验现象，我会按下面顺序：

1. 先做 ER baseline
2. 先加 Gaussian，确认“只补方差”确实有效
3. 再加 WAP，确认“方向性扰动”额外带来的收益
4. 最后再看 DOA / VT / vMF

### 24.2 我最想继续追的三个方向

1. 直接学习一个 conditional perturbation generator，显式拟合 \(p(\Delta f\mid f,y,\theta)\)
2. 把 WAP 和 distillation 更紧地耦合，解决 Gaussian 与 DER++ 兼容性一般的问题
3. 用更正式的梯度匹配目标替代当前启发式扰动，例如最小化 replay gradient 与 joint gradient 的距离

---

## 25. 最后总结

这篇论文最值得记住的，不是某个单独公式，而是下面这句话：

> continual learning 里，旧类忘得快，不只是因为旧样本少，更因为“旧类在表示空间里不再像一个有厚度的分布，而是塌成了几个点”；只要把这个厚度补回来，很多 forgetting 就能缓解。

MOCA 的全部设计，都是围绕这件事展开的。

如果只记一个最强变体，那就是：

- `WAP = 用最容易把旧类推向新类边界的对抗权重扰动，构造旧类的困难方向，再逼模型把这些方向也学稳`

这也是为什么它会比简单高斯扰动更稳定、更强。

---

## 26. 参考链接

- TMLR / OpenReview：<https://openreview.net/forum?id=iDxfGaMYVr>
- arXiv：<https://arxiv.org/abs/2210.05398>
- 官方代码：<https://github.com/yulonghui/MOCA>
