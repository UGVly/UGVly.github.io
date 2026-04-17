---
tags: [public]
---

# Robust Learning with Jacobian Regularization：论文阅读与公式推导

这篇笔记对应论文：

- Judy Hoffman, Daniel A. Roberts, Sho Yaida, *Robust Learning with Jacobian Regularization*, arXiv:1908.02729.

我想完成三件事：

- 把论文主文里的核心公式从头推一遍；
- 把附录里“随机投影近似为什么成立”与 “cyclopropagation 精确梯度”整理成更顺手的形式；
- 把论文真正证明了什么、只是经验上观察到什么、以及阅读时要注意的细节分开说清楚。

---

## 1. 论文到底在做什么

作者考虑一个分类模型
$$
f_\theta:\mathbb{R}^d\to\mathbb{R}^C,
\qquad
x\mapsto z(x)=f_\theta(x),
$$
其中 \(z(x)\) 是 **logits**，不是 softmax 后的概率。

论文的核心主张是：如果我们在训练时直接惩罚输入到 logits 的 Jacobian
$$
J(x):=\frac{\partial z(x)}{\partial x}\in\mathbb{R}^{C\times d},
$$
也就是惩罚
$$
\|J(x)\|_F^2,
$$
那么模型在输入扰动下会更稳定，决策边界会在训练样本附近“被推远”，从而提升对白噪声和对抗扰动的鲁棒性，同时不明显牺牲干净数据上的精度。

论文最重要的三点贡献可以压缩成：

1. 用输入扰动的 Taylor 展开说明，局部稳定性的一阶主导量就是输入输出 Jacobian。
2. 提出一个便宜很多的随机投影估计器，用常数级额外开销近似 \(\|J(x)\|_F^2\)。
3. 在 MNIST、CIFAR-10、ImageNet 上展示：Jacobian regularization 往往能显著提升鲁棒性，尤其对白噪声与 CW 攻击。

---

## 2. 从输入扰动到输入输出 Jacobian

### 2.1 论文式 (1)-(2)：Taylor 展开

对输入 \(x\in\mathbb{R}^d\) 施加一个小扰动 \(\delta\in\mathbb{R}^d\)，记扰动后的输入为
$$
\widetilde{x}=x+\delta.
$$

第 \(c\) 个 logit 在 \(x\) 处做一阶 Taylor 展开：
$$
z_c(x+\delta)
=
z_c(x)+\sum_{i=1}^d \delta_i \frac{\partial z_c}{\partial x_i}(x)+O(\|\delta\|^2).
$$

把所有输出分量合在一起，就得到向量形式
$$
z(x+\delta)
=
z(x)+J(x)\delta+O(\|\delta\|^2),
$$
其中
$$
J_{c,i}(x):=\frac{\partial z_c}{\partial x_i}(x).
$$

这就是论文中“输入输出 Jacobian 控制局部稳定性”的最基本来源。

### 2.2 一个直接的稳定性上界

由上式立刻得到
$$
\|z(x+\delta)-z(x)\|_2
\le
\|J(x)\delta\|_2+O(\|\delta\|^2)
\le
\|J(x)\|_2\|\delta\|_2+O(\|\delta\|^2).
$$

又因为谱范数不超过 Frobenius 范数，
$$
\|J(x)\|_2\le \|J(x)\|_F,
$$
于是
$$
\|z(x+\delta)-z(x)\|_2
\le
\|J(x)\|_F\|\delta\|_2+O(\|\delta\|^2).
$$

所以从局部 Lipschitz 常数的角度看，最小化 \(\|J(x)\|_F\) 确实会压低输出对输入小扰动的敏感度。

这一步不是论文单独写成定理，但它是论文全部直觉的核心。

---

## 3. Jacobian 正则项与联合训练目标

### 3.1 论文式 (3)：Jacobian regularizer

论文定义的 Jacobian 正则项是
$$
\|J(x)\|_F^2
:=
\sum_{c=1}^C\sum_{i=1}^d J_{c,i}(x)^2.
$$

如果写成迹的形式，就是
$$
\|J(x)\|_F^2
=
\operatorname{Tr}(J(x)J(x)^\top)
=
\operatorname{Tr}(J(x)^\top J(x)).
$$

### 3.2 论文式 (4)-(5)：和监督损失拼起来

设 mini-batch 为
$$
\mathcal{B}=\{(x^\alpha,y^\alpha)\}_{\alpha\in\mathcal{B}},
$$
原始监督训练目标记为
$$
\mathcal{L}_{\mathrm{bare}}(\theta)
=
\left[
\frac{1}{|\mathcal{B}|}\sum_{\alpha\in\mathcal{B}}
\mathcal{L}_{\mathrm{super}}(f_\theta(x^\alpha),y^\alpha)
\right]
+
\mathcal{R}(\theta),
$$
其中 \(\mathcal{R}(\theta)\) 可以是 weight decay、dropout 等别的正则项。

加入 Jacobian regularization 后，联合目标变成
$$
\mathcal{L}_{\mathrm{joint}}(\theta)
=
\mathcal{L}_{\mathrm{bare}}(\theta)
+
\frac{\lambda_{\mathrm{JR}}}{2}
\left[
\frac{1}{|\mathcal{B}|}
\sum_{\alpha\in\mathcal{B}}
\|J(x^\alpha)\|_F^2
\right].
$$

这里 \(\lambda_{\mathrm{JR}}>0\) 是 Jacobian 正则的权重。

---

## 4. 为什么在线性模型里它就等于 \(L^2\) 正则

论文有一句非常关键的话：**在线性模型里，Jacobian regularization 精确退化成 \(L^2\) regularization。**

这一步可以完整写出来。

设模型是
$$
z(x)=Wx+b,
\qquad
W\in\mathbb{R}^{C\times d},\ b\in\mathbb{R}^C.
$$

那么对输入求导：
$$
J(x)=\frac{\partial (Wx+b)}{\partial x}=W.
$$

因此
$$
\|J(x)\|_F^2=\|W\|_F^2,
$$
完全不依赖于 \(x\)。

于是 batch Jacobian 正则项就是
$$
\frac{\lambda_{\mathrm{JR}}}{2|\mathcal{B}|}\sum_{\alpha\in\mathcal{B}}\|J(x^\alpha)\|_F^2
=
\frac{\lambda_{\mathrm{JR}}}{2}\|W\|_F^2,
$$
这正是标准的 weight decay。

### 4.1 为什么这和 margin 相关

以二分类线性模型
$$
g(x)=w^\top x+b
$$
为例，几何间隔是
$$
\gamma(x,y)=\frac{y(w^\top x+b)}{\|w\|_2}.
$$

如果在保证训练样本函数间隔 \(y(w^\top x+b)\) 不变的前提下减小 \(\|w\|_2\)，几何间隔就会增大。所以在线性模型里，Jacobian regularization 与 margin 增大是完全一致的。

论文把这一点当作对非线性情形的启发：在线性情形里缩小 Jacobian 就是缩小权重范数；在非线性情形里它不再等同于 weight decay，但仍然会把局部决策边界往外推。

---

## 5. 一个很重要的补充推论：它为什么会“增大决策单元”

论文用图展示了 decision cells 变大，但把这一点写成公式其实也不难。

设当前样本 \(x\) 被分类为类别 \(c\)，对任意竞争类别 \(k\neq c\)，定义 logit gap
$$
g_k(x):=z_c(x)-z_k(x).
$$

决策边界 \(c\leftrightarrow k\) 就由
$$
g_k(x)=0
$$
刻画。

对扰动 \(\delta\) 做一阶展开：
$$
g_k(x+\delta)
=
g_k(x)+\nabla_x g_k(x)^\top \delta+O(\|\delta\|^2).
$$

而
$$
\nabla_x g_k(x)
=
\nabla_x z_c(x)-\nabla_x z_k(x)
=
J_{c,:}(x)-J_{k,:}(x).
$$

于是
$$
|\nabla_x g_k(x)^\top \delta|
\le
\|J_{c,:}(x)-J_{k,:}(x)\|_2\,\|\delta\|_2.
$$

再利用
$$
\|J_{c,:}(x)-J_{k,:}(x)\|_2
\le
\|J_{c,:}(x)\|_2+\|J_{k,:}(x)\|_2
\le
\sqrt{2}\sqrt{\|J_{c,:}(x)\|_2^2+\|J_{k,:}(x)\|_2^2}
\le
\sqrt{2}\|J(x)\|_F,
$$
可得
$$
|\nabla_x g_k(x)^\top \delta|
\le
\sqrt{2}\|J(x)\|_F\,\|\delta\|_2.
$$

因此忽略高阶项时，只要
$$
g_k(x)>\sqrt{2}\|J(x)\|_F\,\|\delta\|_2
\qquad
\forall k\neq c,
$$
那么 \(x+\delta\) 仍然不会跨过任何一条决策边界。

换句话说，一个局部的“可容忍扰动半径”近似可以写成
$$
r_{\mathrm{local}}(x)
\gtrsim
\min_{k\neq c}
\frac{g_k(x)}{\sqrt{2}\|J(x)\|_F}.
$$

这不是论文正文里的原公式，而是从论文公式直接推出的补充解释。它把论文反复强调的“decision cell 变大”写成了清楚的半径比值：在 logit gap 固定时，\(\|J(x)\|_F\) 越小，局部 margin 半径就越大。

---

## 6. 如何精确计算 \(\|J(x)\|_F^2\)

### 6.1 论文式 (6)：迹恒等式

设 \(\{e_1,\dots,e_C\}\) 是输出空间 \(\mathbb{R}^C\) 的标准正交基。则
$$
\|J(x)\|_F^2
=
\operatorname{Tr}(J J^\top)
=
\sum_{m=1}^C e_m^\top J J^\top e_m.
$$

注意
$$
e_m^\top J J^\top e_m
=
\|J^\top e_m\|_2^2.
$$

另一方面，
$$
\nabla_x(e_m^\top z(x))
=
J(x)^\top e_m.
$$

所以
$$
\|J(x)\|_F^2
=
\sum_{m=1}^C
\left\|
\nabla_x\big(e_m^\top z(x)\big)
\right\|_2^2.
$$

这就是论文正文里“可以用自动微分做”的关键恒等式。

### 6.2 这个精确算法为什么贵

因为你要把 \(m=1,\dots,C\) 全扫一遍，也就是要做 \(C\) 次向量-雅可比积：
$$
e_m^\top z(x)\ \longrightarrow\ \nabla_x(e_m^\top z(x)).
$$

如果类别数 \(C\) 很大，比如 ImageNet 的 \(C=1000\)，这个成本就会很高。论文的主要工程贡献，就是把这里的 \(C\) 次开销近似成常数次。

---

## 7. 随机投影近似：为什么它是无偏的

### 7.1 论文式 (7)：核心结论

令 \(\hat v\) 在单位球面 \(S^{C-1}\) 上均匀采样，则
$$
\|J(x)\|_F^2
=
C\,\mathbb{E}_{\hat v\sim S^{C-1}}
\left[\|\hat v^\top J(x)\|_2^2\right].
$$

这是论文最重要的近似基础。

### 7.2 无偏性的完整推导

记
$$
A:=J J^\top\in\mathbb{R}^{C\times C}.
$$

由于 \(\hat v\) 在球面上均匀分布，具有旋转对称性，所以
$$
\mathbb{E}[\hat v\hat v^\top]=\frac{1}{C}I_C.
$$

于是
$$
\mathbb{E}\big[\|\hat v^\top J\|_2^2\big]
=
\mathbb{E}\big[\hat v^\top J J^\top \hat v\big]
=
\mathbb{E}\big[\hat v^\top A\hat v\big].
$$

再把标量写成迹：
$$
\hat v^\top A\hat v
=
\operatorname{Tr}(\hat v^\top A\hat v)
=
\operatorname{Tr}(A\hat v\hat v^\top).
$$

取期望得到
$$
\mathbb{E}\big[\hat v^\top A\hat v\big]
=
\operatorname{Tr}\big(A\,\mathbb{E}[\hat v\hat v^\top]\big)
=
\operatorname{Tr}\left(A\cdot \frac{1}{C}I_C\right)
=
\frac{1}{C}\operatorname{Tr}(A).
$$

而
$$
\operatorname{Tr}(A)
=
\operatorname{Tr}(J J^\top)
=
\|J\|_F^2.
$$

所以
$$
\mathbb{E}\big[\|\hat v^\top J\|_2^2\big]
=
\frac{1}{C}\|J\|_F^2,
$$
等价地
$$
\|J\|_F^2
=
C\,\mathbb{E}\big[\|\hat v^\top J\|_2^2\big].
$$

证毕。

### 7.3 变成自动微分可以直接算的形式

因为
$$
\nabla_x(\hat v^\top z(x))
=
J(x)^\top \hat v,
$$
所以
$$
\|\hat v^\top J(x)\|_2^2
=
\|J(x)^\top \hat v\|_2^2
=
\left\|\nabla_x(\hat v^\top z(x))\right\|_2^2.
$$

于是用 \(n_{\mathrm{proj}}\) 个随机方向 \(\hat v^{(1)},\dots,\hat v^{(n_{\mathrm{proj}})}\) 做 Monte Carlo 估计，就得到
$$
\|J(x)\|_F^2
\approx
\frac{C}{n_{\mathrm{proj}}}
\sum_{\mu=1}^{n_{\mathrm{proj}}}
\left\|
\nabla_x\Big((\hat v^{(\mu)})^\top z(x)\Big)
\right\|_2^2.
$$

这正是训练时真正可以实现的形式。

### 7.4 阅读时要特别注意：论文正文式 (8) 少了一个 \(C\)

论文正文的随机投影近似式写成了
$$
\|J(x)\|_F^2
\approx
\frac{1}{n_{\mathrm{proj}}}
\sum_{\mu=1}^{n_{\mathrm{proj}}}
\left\|
\nabla_x\Big((\hat v^{(\mu)})^\top z(x)\Big)
\right\|_2^2,
$$
但这和上一式的无偏关系并不一致。

按正文式 (7) 以及 Algorithm 1 的写法，正确的无偏估计器应当是
$$
\widehat{\|J(x)\|_F^2}
=
\frac{C}{n_{\mathrm{proj}}}
\sum_{\mu=1}^{n_{\mathrm{proj}}}
\left\|
\nabla_x\Big((\hat v^{(\mu)})^\top z(x)\Big)
\right\|_2^2.
$$

也就是说，**正文式 (8) 大概率少写了一个输出维度 \(C\) 因子**。论文算法伪代码里是带这个因子的，所以实际理解和实现都应以后者为准。

---

## 8. 随机投影估计器的方差推导

论文附录用 Haar 积分给出证明。这里我用一个更直接、但与附录等价的四阶矩推导。

### 8.1 四阶矩公式

仍记 \(\hat v\sim \mathrm{Unif}(S^{C-1})\)。其四阶矩满足
$$
\mathbb{E}[\hat v_i\hat v_j\hat v_k\hat v_\ell]
=
\frac{
\delta_{ij}\delta_{k\ell}
+
\delta_{ik}\delta_{j\ell}
+
\delta_{i\ell}\delta_{jk}
}{C(C+2)}.
$$

这是球面对称分布的标准结果，也与论文附录里的 Haar 平均等价。

### 8.2 二次型平方的期望

令
$$
A:=J J^\top\in\mathbb{R}^{C\times C},
$$
它是对称半正定矩阵。

考虑
$$
Q:=\hat v^\top A\hat v.
$$

则
$$
Q^2
=
\sum_{i,j,k,\ell}
A_{ij}A_{k\ell}\hat v_i\hat v_j\hat v_k\hat v_\ell.
$$

取期望并代入四阶矩公式：
$$
\mathbb{E}[Q^2]
=
\frac{1}{C(C+2)}
\sum_{i,j,k,\ell}
A_{ij}A_{k\ell}
\big(
\delta_{ij}\delta_{k\ell}
+
\delta_{ik}\delta_{j\ell}
+
\delta_{i\ell}\delta_{jk}
\big).
$$

三项分别计算：

第一项：
$$
\sum_{i,j,k,\ell}
A_{ij}A_{k\ell}\delta_{ij}\delta_{k\ell}
=
\sum_i A_{ii}\sum_k A_{kk}
=
(\operatorname{Tr}A)^2.
$$

第二项：
$$
\sum_{i,j,k,\ell}
A_{ij}A_{k\ell}\delta_{ik}\delta_{j\ell}
=
\sum_{i,j}A_{ij}A_{ij}
=
\|A\|_F^2.
$$

第三项：
$$
\sum_{i,j,k,\ell}
A_{ij}A_{k\ell}\delta_{i\ell}\delta_{jk}
=
\sum_{i,j}A_{ij}A_{ji}
=
\operatorname{Tr}(A^2),
$$
而因为 \(A\) 对称，
$$
\|A\|_F^2=\operatorname{Tr}(A^2).
$$

所以
$$
\mathbb{E}[Q^2]
=
\frac{(\operatorname{Tr}A)^2+2\operatorname{Tr}(A^2)}{C(C+2)}.
$$

### 8.3 论文附录中的方差公式

论文真正用的估计器是
$$
\widehat{S}_1:=C\,Q=C\,\hat v^\top A\hat v.
$$

它的均值是
$$
\mathbb{E}[\widehat{S}_1]
=
\operatorname{Tr}(A)
=
\|J\|_F^2.
$$

方差则为
$$
\operatorname{Var}(\widehat{S}_1)
=
C^2\mathbb{E}[Q^2]-(\operatorname{Tr}A)^2.
$$

把上式代入可得
$$
\operatorname{Var}(\widehat{S}_1)
=
\frac{2C}{C+2}\operatorname{Tr}(A^2)
-
\frac{2}{C+2}(\operatorname{Tr}A)^2.
$$

再还原成 \(J\)：
$$
\operatorname{Var}(\widehat{S}_1)
=
\frac{2C}{C+2}\operatorname{Tr}(J J^\top J J^\top)
-
\frac{2}{C+2}\|J\|_F^4.
$$

这和论文附录给出的结果完全一致。

### 8.4 相对方差上界

因为 \(A\succeq 0\)，其特征值非负，所以
$$
\operatorname{Tr}(A^2)\le (\operatorname{Tr}A)^2.
$$

因此
$$
\frac{\operatorname{Var}(\widehat{S}_1)}{\mathbb{E}[\widehat{S}_1]^2}
\le
\frac{2C}{C+2}-\frac{2}{C+2}
=
2\frac{C-1}{C+2}
<2.
$$

这说明单个随机方向的相对方差虽然不一定特别小，但它是 **维度无关的有界常数**，不会随着模型输出维度暴涨而失控。

如果用 \(n_{\mathrm{proj}}\) 个独立投影做平均，则
$$
\operatorname{Var}\!\left(\frac{1}{n_{\mathrm{proj}}}\sum_{\mu=1}^{n_{\mathrm{proj}}}\widehat{S}_\mu\right)
=
\frac{1}{n_{\mathrm{proj}}}\operatorname{Var}(\widehat{S}_1),
$$
所以标准差按
$$
O(n_{\mathrm{proj}}^{-1/2})
$$
下降。

再对 mini-batch 平均，误差还会再有大约
$$
O(|\mathcal{B}|^{-1/2})
$$
的抑制。这就是论文为什么敢在实践里默认只用一个随机投影。

---

## 9. 训练时到底在算什么

把前面的推导都拼起来，单个 batch 上的 Jacobian 正则项近似可以写成
$$
\widehat{\mathcal{R}}_{\mathrm{JR}}(\mathcal{B})
=
\frac{\lambda_{\mathrm{JR}}}{2|\mathcal{B}|}
\sum_{\alpha\in\mathcal{B}}
\frac{C}{n_{\mathrm{proj}}}
\sum_{\mu=1}^{n_{\mathrm{proj}}}
\left\|
\nabla_{x^\alpha}\Big((\hat v_\alpha^{(\mu)})^\top z(x^\alpha)\Big)
\right\|_2^2.
$$

实现流程就是：

1. 正常前向，得到 batch logits \(z(x^\alpha)\)。
2. 给每个样本采一个或几个随机方向 \(\hat v_\alpha^{(\mu)}\in S^{C-1}\)。
3. 把整个 batch 的 \((\hat v_\alpha^{(\mu)})^\top z(x^\alpha)\) 求和成一个标量。
4. 对输入 \(x^\alpha\) 求梯度，得到对应的 \(J^\top \hat v\)。
5. 把这些输入梯度的平方范数加总起来，再乘上 \(C\)。
6. 最后再对参数 \(\theta\) 反传。

论文报告：在 MNIST + LeNet' 上，\(n_{\mathrm{proj}}=1\) 时训练开销大约只是普通训练的 \(1.3\times\)；在较大的模型上也大体仍是常数倍额外成本，而不是 \(C\) 倍。

---

## 10. FGSM 公式为什么会自然冒出来

论文正文写了 FGSM 攻击
$$
\delta_i
=
\varepsilon_{\mathrm{FGSM}}
\cdot
\operatorname{sign}\left(
\sum_{c=1}^C
\frac{\partial \mathcal{L}_{\mathrm{super}}}{\partial z_c}
J_{c,i}
\right).
$$

这其实就是链式法则。

监督损失记为
$$
\mathcal{L}_{\mathrm{super}}(z(x),y).
$$

则对输入的梯度为
$$
\nabla_x \mathcal{L}_{\mathrm{super}}
=
J(x)^\top \nabla_z \mathcal{L}_{\mathrm{super}}.
$$

分量写开就是
$$
\frac{\partial \mathcal{L}_{\mathrm{super}}}{\partial x_i}
=
\sum_{c=1}^C
\frac{\partial \mathcal{L}_{\mathrm{super}}}{\partial z_c}
\frac{\partial z_c}{\partial x_i}
=
\sum_{c=1}^C
\frac{\partial \mathcal{L}_{\mathrm{super}}}{\partial z_c}J_{c,i}.
$$

而 FGSM 的定义就是
$$
\delta
=
\varepsilon_{\mathrm{FGSM}}
\operatorname{sign}\big(\nabla_x \mathcal{L}_{\mathrm{super}}\big),
$$
所以分量形式就是论文的那一式。

这一步也解释了为什么 Jacobian regularization 会帮忙：它直接压小了 \(\nabla_x \mathcal{L}_{\mathrm{super}}\) 里最核心的那一块 \(J^\top\)。

当然，FGSM 只是一种一阶近似攻击；论文真正更看重的是：Jacobian 小意味着局部决策边界更远，这对白噪声、PGD、CW 都会带来几何上的好处，但不会自动给出严格认证鲁棒性。

---

## 11. 附录里的闭式精确梯度：cyclopropagation

这部分是整篇论文最技术化、也最容易读晕的地方。论文主体实际主打的是“随机投影 + autograd”，但附录还给了一个对 MLP 的闭式梯度公式。下面我把它改写成更清楚的矩阵形式。

先说明一个记号问题：主文里 \(z(x)\) 表示最终输出 logits；而附录里 \(z^{(\ell)}\) 表示第 \(\ell\) 层激活后的状态。为了和附录保持一致，下面沿用附录的层内记号。如果分类器最终层确实要输出 logits，那么可把最后一层激活取成恒等映射，即
$$
\sigma^{(L)}(u)=u,
\qquad
D^{(L)}=I,
\qquad
H^{(L)}=0.
$$

### 11.1 多层感知机的记号

设网络共有 \(L\) 层，定义
$$
z^{(0)}=x,
$$
$$
\hat z^{(\ell)}=W^{(\ell)}z^{(\ell-1)}+b^{(\ell)},
\qquad
\ell=1,\dots,L,
$$
$$
z^{(\ell)}=\sigma(\hat z^{(\ell)}),
\qquad
\ell=1,\dots,L.
$$

令
$$
D^{(\ell)}:=\operatorname{diag}\big(\sigma'(\hat z^{(\ell)})\big),
\qquad
H^{(\ell)}:=\operatorname{diag}\big(\sigma''(\hat z^{(\ell)})\big).
$$

则第 \(\ell\) 层的 layerwise Jacobian 为
$$
J^{(\ell)}
:=
\frac{\partial z^{(\ell)}}{\partial z^{(\ell-1)}}
=
D^{(\ell)}W^{(\ell)}.
$$

总的输入输出 Jacobian 是
$$
J
=
J^{(L)}J^{(L-1)}\cdots J^{(1)}.
$$

定义正则项
$$
\mathcal{R}_{\mathrm{JR}}
:=
\frac{1}{2}\|J\|_F^2
=
\frac{1}{2}\operatorname{Tr}(J^\top J).
$$

### 11.2 先求微分：\(d\mathcal{R}_{\mathrm{JR}}=\operatorname{Tr}(J^\top dJ)\)

因为
$$
\mathcal{R}_{\mathrm{JR}}=\frac{1}{2}\operatorname{Tr}(J^\top J),
$$
对它求微分：
$$
d\mathcal{R}_{\mathrm{JR}}
=
\frac{1}{2}\operatorname{Tr}(dJ^\top J + J^\top dJ)
=
\operatorname{Tr}(J^\top dJ),
$$
这里用了迹的对称性：
$$
\operatorname{Tr}(dJ^\top J)=\operatorname{Tr}(J^\top dJ).
$$

如果只看第 \(\ell\) 层的变化，那么
$$
dJ
=
J_{>\ell}\, dJ^{(\ell)}\, J_{<\ell},
$$
其中
$$
J_{<\ell}:=J^{(\ell-1)}\cdots J^{(1)},
\qquad
J_{>\ell}:=J^{(L)}\cdots J^{(\ell+1)}.
$$
边界处约定
$$
J_{<1}:=I_{n_0},
\qquad
J_{>L}:=I_{n_L},
$$
这样 \(\ell=1\) 与 \(\ell=L\) 时公式仍然成立。

代回去：
$$
d\mathcal{R}_{\mathrm{JR}}
=
\operatorname{Tr}\big(J^\top J_{>\ell}\, dJ^{(\ell)}\, J_{<\ell}\big).
$$

利用迹的循环不变性：
$$
d\mathcal{R}_{\mathrm{JR}}
=
\operatorname{Tr}\big(J_{<\ell}J^\top J_{>\ell}\, dJ^{(\ell)}\big).
$$

因此定义
$$
\Omega^{(\ell)}
:=
J_{<\ell}J^\top J_{>\ell},
$$
就有
$$
d\mathcal{R}_{\mathrm{JR}}
=
\operatorname{Tr}\big(\Omega^{(\ell)} dJ^{(\ell)}\big).
$$

这就是论文附录里那个最核心的中间量。

### 11.3 把 \(dJ^{(\ell)}\) 展开

因为
$$
J^{(\ell)}=D^{(\ell)}W^{(\ell)},
$$
所以
$$
dJ^{(\ell)}
=
dD^{(\ell)}W^{(\ell)} + D^{(\ell)}dW^{(\ell)}.
$$

又因为
$$
D^{(\ell)}=\operatorname{diag}(\sigma'(\hat z^{(\ell)})),
$$
所以
$$
dD^{(\ell)}
=
\operatorname{diag}\big(\sigma''(\hat z^{(\ell)})\odot d\hat z^{(\ell)}\big).
$$

这里 \(\odot\) 是逐元素乘法。

### 11.4 先得到对 pre-activation 的梯度

定义
$$
q^{(\ell)}:=\frac{\partial \mathcal{R}_{\mathrm{JR}}}{\partial \hat z^{(\ell)}}
=
\frac{\partial \mathcal{R}_{\mathrm{JR}}}{\partial b^{(\ell)}}.
$$

为什么它同时也是对偏置的梯度？因为
$$
\hat z^{(\ell)}=W^{(\ell)}z^{(\ell-1)}+b^{(\ell)},
$$
对 \(b^{(\ell)}\) 的微分就是
$$
d\hat z^{(\ell)}=db^{(\ell)}.
$$

先看由 \(dD^{(\ell)}\) 带来的“直接项”：
$$
\operatorname{Tr}\big(\Omega^{(\ell)} dD^{(\ell)} W^{(\ell)}\big)
=
\operatorname{Tr}\big(W^{(\ell)}\Omega^{(\ell)} dD^{(\ell)}\big).
$$

由于 \(dD^{(\ell)}\) 是对角矩阵，
$$
\operatorname{Tr}\big(W^{(\ell)}\Omega^{(\ell)} dD^{(\ell)}\big)
=
\sum_j \big[W^{(\ell)}\Omega^{(\ell)}\big]_{jj}\,
\sigma''(\hat z_j^{(\ell)})\, d\hat z_j^{(\ell)}.
$$

所以直接项对应的梯度是
$$
q_{\mathrm{dir}}^{(\ell)}
=
\sigma''(\hat z^{(\ell)})\odot
\operatorname{diag}\big(W^{(\ell)}\Omega^{(\ell)}\big).
$$

再看“递推项”。因为
$$
\hat z^{(\ell+1)}=W^{(\ell+1)}z^{(\ell)}+b^{(\ell+1)},
\qquad
z^{(\ell)}=\sigma(\hat z^{(\ell)}),
$$
所以
$$
d\hat z^{(\ell+1)}
=
W^{(\ell+1)}D^{(\ell)} d\hat z^{(\ell)}+\cdots
$$

其中“\(\cdots\)”表示与 \(d\hat z^{(\ell)}\) 无关的项。

于是由更深层回传来的项满足
$$
\big(q^{(\ell+1)}\big)^\top d\hat z^{(\ell+1)}
=
\big(q^{(\ell+1)}\big)^\top W^{(\ell+1)}D^{(\ell)} d\hat z^{(\ell)},
$$
因此
$$
q_{\mathrm{ind}}^{(\ell)}
=
D^{(\ell)}(W^{(\ell+1)})^\top q^{(\ell+1)}.
$$

两部分合起来，得到更清楚的递推式：
$$
q^{(\ell)}
=
D^{(\ell)}(W^{(\ell+1)})^\top q^{(\ell+1)}
+
\sigma''(\hat z^{(\ell)})\odot
\operatorname{diag}\big(W^{(\ell)}\Omega^{(\ell)}\big),
$$
这个式子对 \(1\le \ell<L\) 直接成立；当 \(\ell=L\) 时，第一项不存在，只保留直接项
$$
q^{(L)}
=
\sigma''(\hat z^{(L)})\odot
\operatorname{diag}\big(W^{(L)}\Omega^{(L)}\big).
$$

如果想把它统一写成一个式子，也可以额外约定 \(q^{(L+1)}:=0\)，并把“第一项”在 \(\ell=L\) 时按 \(0\) 处理。这和论文附录中的偏置梯度递推是等价的，只是这里避免了逐元素除以 \(\sigma'(\hat z)\) 的写法，读起来更直接。

### 11.5 对权重的梯度

由
$$
\hat z^{(\ell)}=W^{(\ell)}z^{(\ell-1)}+b^{(\ell)}
$$
可知
$$
d\hat z^{(\ell)}=dW^{(\ell)}z^{(\ell-1)}+\cdots
$$

于是从 \(q^{(\ell)}\) 这一项得到
$$
\big(q^{(\ell)}\big)^\top d\hat z^{(\ell)}
=
\big(q^{(\ell)}\big)^\top dW^{(\ell)} z^{(\ell-1)},
$$
对应梯度
$$
q^{(\ell)}(z^{(\ell-1)})^\top.
$$

另外，前面 \(D^{(\ell)}dW^{(\ell)}\) 还有一项直接贡献：
$$
\operatorname{Tr}\big(\Omega^{(\ell)} D^{(\ell)} dW^{(\ell)}\big)
=
\operatorname{Tr}\big((D^{(\ell)}(\Omega^{(\ell)})^\top)^\top dW^{(\ell)}\big),
$$
所以它对应的梯度是
$$
D^{(\ell)}(\Omega^{(\ell)})^\top.
$$

最终得到
$$
\frac{\partial \mathcal{R}_{\mathrm{JR}}}{\partial W^{(\ell)}}
=
q^{(\ell)}(z^{(\ell-1)})^\top
+
D^{(\ell)}(\Omega^{(\ell)})^\top.
$$

分量形式就是
$$
\frac{\partial \mathcal{R}_{\mathrm{JR}}}{\partial w_{j_\ell,j_{\ell-1}}^{(\ell)}}
=
q_{j_\ell}^{(\ell)} z_{j_{\ell-1}}^{(\ell-1)}
+
\sigma'(\hat z_{j_\ell}^{(\ell)})
\Omega_{j_{\ell-1},j_\ell}^{(\ell)},
$$
这与论文附录给出的权重梯度公式一致。

### 11.6 这部分该怎么理解

可以把它理解成：

- 第一项 \(q^{(\ell)}(z^{(\ell-1)})^\top\) 是“通过 pre-activation 影响 Jacobian regularizer”的普通链式法则项；
- 第二项 \(D^{(\ell)}(\Omega^{(\ell)})^\top\) 是“当前层 Jacobian 本身对权重的直接依赖”带来的额外项。

附录把这个过程叫做 **cyclopropagation**，因为为了计算 \(\Omega^{(\ell)}\)，信息流需要围绕整张网络“绕一圈”。

### 11.7 这套闭式公式的局限

这套推导依赖 \(\sigma'\) 与 \(\sigma''\) 的存在，因此对平滑激活函数最自然；而对 ReLU 这类在折点不可二阶可导的激活，闭式写法会变得更微妙。也正因此，论文主体强调的通用做法仍然是：

- 用随机投影把 \(\|J\|_F^2\) 变成输入梯度范数；
- 再交给自动微分系统统一处理。

从工程上看，这比显式实现 cyclopropagation 更稳妥，也更容易推广到卷积网络、残差网络等复杂结构。

---

## 12. 论文实验到底说明了什么

### 12.1 MNIST 干净精度几乎不掉，但 Jacobian 显著变小

论文主表给出的 LeNet' on MNIST（全训练集）结果里：

- 无正则：测试精度 \(98.9\%\)，\(\|J\|_F\approx 32.9\)；
- \(L^2\)：测试精度 \(99.2\%\)，\(\|J\|_F\approx 4.6\)；
- Dropout：测试精度 \(98.6\%\)，\(\|J\|_F\approx 21.5\)；
- Jacobian：测试精度 \(99.0\%\)，\(\|J\|_F\approx 1.1\)；
- 全部结合：测试精度 \(99.1\%\)，\(\|J\|_F\approx 1.2\)。

最关键信息不是精度略升还是略降，而是：

- Jacobian 正则几乎不伤害 clean accuracy；
- 但能把输入输出 Jacobian 压到非常小；
- 且和 \(L^2\)、dropout、adversarial training 可以叠加。

### 12.2 对跨域泛化有一点帮助，但不是论文最强结论

MNIST 训练、USPS 测试时：

- 无正则：\(80.4\%\)；
- Jacobian：\(81.3\%\)；
- All combined：\(85.7\%\)。

所以它对 domain shift 可能有帮助，但这个证据相对初步，不像鲁棒性实验那么强。

### 12.3 真正的核心结果是鲁棒性

论文最强调三件事：

- 白噪声下，Jacobian regularization 是最强的单独正则；
- 在 MNIST 上，Jacobian regularization 单独使用就能优于一个 FGSM 式 adversarial training baseline；
- 在 CIFAR-10、ImageNet 上，白噪声和 CW 攻击下的收益也大体保留。

但也要注意作者自己承认的边界：

- PGD 在 CIFAR 上的结果并不总是稳定占优；
- ImageNet 结果是 preliminary 的，而且只有单次实验；
- 这篇论文讨论的是经验鲁棒性提升，不是 certified robustness。

---

## 13. 我对这篇论文的几点阅读判断

### 13.1 这篇论文最有价值的部分不是“想到惩罚 Jacobian”

惩罚输入梯度或 Jacobian 的思路并不是这篇论文首创。它最有价值的地方在于：

- 把 full input-output Jacobian 的计算成本从“随类别数线性增长”降到了“常数级随机投影”；
- 证明了这个随机估计器是无偏的，并且相对方差有维度无关上界；
- 用实验说明这种近似在实际训练里几乎不伤效果。

也就是说，真正的贡献是 **把一个原来很贵的正则项做成了实用工具**。

### 13.2 它控制的是局部平均敏感度，不是最坏方向的严格界

Frobenius 范数
$$
\|J\|_F^2=\sum_{c,i}J_{c,i}^2
$$
衡量的是所有输入-输出偏导数的平方和。

它当然会控制谱范数，因为
$$
\|J\|_2\le \|J\|_F,
$$
但它不是直接最小化最坏方向上的局部 Lipschitz 常数。因此：

- 它很适合作为可微、便宜、泛化性强的训练正则；
- 但它不等于“给每个样本都做了严格认证”。

### 13.3 论文的几何解释是对的，但更准确地说是“在训练样本附近”对

论文图里画出的 decision cells 变大，非常直观；但从数学上看，这一切都是围绕
$$
z(x+\delta)\approx z(x)+J(x)\delta
$$
的局部一阶分析建立的。

所以更准确的说法应该是：

- 它显著改善了训练样本附近的局部几何；
- 这种局部几何改善在测试时通常能迁移成更好的经验鲁棒性；
- 但离开训练分布很远的区域，没有同等强度的理论保证。

### 13.4 正文式 (8) 的小笔误值得注意

如果只照着正文式 (8) 抄代码，很容易少掉一个 \(C\) 因子。

不过由于 Algorithm 1 写对了，实际实现时一般不会出问题。阅读论文时只要记住：
$$
\widehat{\|J\|_F^2}
=
\frac{C}{n_{\mathrm{proj}}}\sum_{\mu}\|J^\top \hat v^{(\mu)}\|_2^2
$$
才是和正文式 (7) 一致的无偏估计器。

---

## 14. 一页总结

如果只保留这篇论文最重要的四行，那就是：

1. 对小扰动 \(\delta\)，
   $$
   z(x+\delta)\approx z(x)+J(x)\delta.
   $$
2. 因此局部稳定性由 \(J(x)\) 控制，最自然的正则就是
   $$
   \frac{\lambda_{\mathrm{JR}}}{2}\|J(x)\|_F^2.
   $$
3. 精确计算太贵，但有无偏随机近似
   $$
   \|J(x)\|_F^2
   =
   C\,\mathbb{E}_{\hat v\sim S^{C-1}}
   \left[\left\|\nabla_x(\hat v^\top z(x))\right\|_2^2\right].
   $$
4. 这会在实践里显著缩小输入输出 Jacobian，并通常增大训练样本附近的局部分类 margin，从而提升对白噪声和对抗扰动的鲁棒性。

---

## 参考资料

- 论文摘要页：<https://arxiv.org/abs/1908.02729>
- 论文源码：<https://arxiv.org/e-print/1908.02729>
- 官方实现：<https://github.com/facebookresearch/jacobian_regularizer>
