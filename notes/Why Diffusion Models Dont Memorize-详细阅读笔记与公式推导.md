---
tags: [public]
category: 研究笔记
---

# Why Diffusion Models Don't Memorize：详细阅读笔记与公式推导

论文信息：

- 标题：*Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training*
- 作者：Tony Bonnaire, Raphaël Urfin, Giulio Biroli, Marc Mézard
- arXiv：<https://arxiv.org/abs/2505.17638>
- 版本：arXiv 页面显示首版提交于 2025 年 5 月 23 日，2025 年 10 月 28 日更新到 v2

这篇论文研究的核心问题非常直接：

> 为什么扩散模型明明可以把训练集记住，但在实际训练中又经常先表现出很好的泛化，而不是立刻记忆训练样本？

作者给出的答案是：

- 对固定模型容量 \(p\)，训练会自然分成两个时间尺度：
  - 一个较早的泛化时间 \(\tau_{\mathrm{gen}}\)，模型在这个阶段就已经能生成质量很高的新样本；
  - 一个更晚的记忆时间 \(\tau_{\mathrm{mem}}\)，只有继续训练到这个时间以后，模型才开始显著复现训练样本。
- 在过参数化但还足以记忆的 regime 下，
$$
\tau_{\mathrm{gen}} \approx \text{常数},\qquad
\tau_{\mathrm{mem}}\propto n,
$$
其中 \(n\) 是训练集大小。

因此，随着 \(n\) 增大，会出现越来越宽的早停窗口
$$
\tau\in[\tau_{\mathrm{gen}},\tau_{\mathrm{mem}}],
$$
模型在这段时间里已经会生成，但还没有开始记忆。这就是文中所谓的 **implicit dynamical regularization**。

![论文总览：训练动力学中的泛化窗口](../../assets/papers/why-diffusion-models-dont-memorize/summary-dynamics.png)

*图 1（左）：作者对训练过程的总览。蓝线表示生成质量的改善，红线表示记忆比例；核心现象就是先到达 \(\tau_{\mathrm{gen}}\)，很久以后才到达 \(\tau_{\mathrm{mem}}\)。*

![论文总览：n-p 相图](../../assets/papers/why-diffusion-models-dont-memorize/summary-phase-diagram.png)

*图 1（右）：\((n,p)\) 相图。作者把现象分成 memorization、architectural regularization 和 dynamical regularization 三个区域。*

---

## 0. 严谨性说明

这篇论文的“推导”其实分成三层：

1. **完全标准且严格的代数推导**
   例如 OU 前向过程的显式解、score matching 极小化器、随机特征网络梯度流闭式解、时间尺度与谱的关系。

2. **高维随机矩阵 / Gaussian equivalence 层面的渐近推导**
   这部分在随机特征文献里是标准套路，论文也给了相对完整的附录推导。

3. **Replica saddle-point 推导**
   作者自己在附录里也明确说了：这一步不是纯数学意义下的严格证明，而是统计物理里非常经典的启发式方法；最终结果也可以从他们引用的更严格随机矩阵路线得到。

所以这份笔记会做到两点：

- 对所有主公式，我都把中间代数链条完整写出来；
- 对于 replica 这类启发式步骤，我会明确标注“这是论文采用的物理学推导，而不是严格概率论证明”。

---

## 1. 论文主结论先讲清楚

作者提出的整体图景可以压缩成下面这四句话：

1. 扩散模型如果真的把 score matching 的经验损失训练到全局极小值，那么它学到的是 **经验 score**，最终会记忆训练集。
2. 但实际训练时，优化动力学并不会一开始就到达那个“记忆解”，而会先停留在一个更平滑、更接近总体 score 的区域。
3. 这个平滑解对应“会生成、但还不记忆”的阶段，其持续时间随 \(n\) 增大而变长。
4. 只有训练足够久以后，模型才会逐渐拟合经验 score 的高频、样本依赖部分，于是记忆出现。

换句话说，这篇论文的主角不是“模型结构正则化”，而是“**训练时间本身就是一种隐式正则化**”。

---

## 1.5 读前预备知识：符号、矩阵谱与随机矩阵

这一节不是论文原文内容，而是为了把后面的推导彻底讲顺。你可以把它当成“进入正文前的工具箱”。

### 1.5.1 这篇论文里最重要的符号表

先把最常见的记号集中说明一下。很多地方之所以读着费劲，不是公式难，而是符号在脑子里没有固定下来。

- \(d\)：数据维度。  
  如果图像是 \(32\times 32\) 灰度图，那么可以理解成 \(d=1024\)。

- \(n\)：训练样本数。

- \(p\)：模型里可训练特征的维度。在随机特征模型里，它就是隐藏层宽度；在真实网络里，它可以粗略对应“模型容量”。

- \(\psi_n=n/d\)、\(\psi_p=p/d\)：高维极限里最常见的两个无量纲比例。

- \(\mathbf x_0\)：原始数据点。

- \(\mathbf x_t\)：扩散到时间 \(t\) 后的 noisy 样本。

- \(\Delta_t=1-e^{-2t}\)：前向 OU 过程产生的噪声方差系数。

- \(P_0\)：真实数据分布；\(P_t\)：加噪后的分布。

- \(\mathbf s(\mathbf x,t)=\nabla_{\mathbf x}\log P_t(\mathbf x)\)：时间 \(t\) 的真实 score。

- \(\mathbf s_{\mathrm{emp}}(\mathbf x,t)\)：把总体分布 \(P_0\) 换成训练集经验分布以后得到的经验 score。

- \(\mathbf\Sigma=\mathbb E[\mathbf x\mathbf x^\top]\)：数据协方差矩阵。

- \(\mathbf U\)：随机特征学习动力学里最关键的 \(p\times p\) 相关矩阵；训练的时间尺度全部由它的特征值控制。

- \(\mathbf V\)：把网络特征和噪声 \(\boldsymbol\xi\) 联系起来的矩阵。

- \(\rho(\lambda)\)：\(\mathbf U\) 的谱密度；可以粗糙理解成“有多少特征值落在 \(\lambda\) 附近”。

- \(\rho_1,\rho_2\)：论文发现的两个主要谱团块。后面会看到：
  - \(\rho_2\) 对应快模态，先学会，主导泛化；
  - \(\rho_1\) 对应慢模态，后学会，主导记忆。

### 1.5.2 “矩阵的谱”到底是什么意思

如果 \(M\in\mathbb R^{p\times p}\) 是对称矩阵，它可以正交对角化：
$$
M = Q\Lambda Q^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_p).
$$

这里：

- \(\lambda_1,\dots,\lambda_p\) 是特征值；
- \(Q=[\mathbf v_1,\dots,\mathbf v_p]\) 的列向量是单位正交特征向量；
- 矩阵作用在每个特征方向上，只是做一个数乘：
$$
M\mathbf v_i=\lambda_i\mathbf v_i.
$$

所以分析一个对称矩阵，核心就是分析：

1. 特征值的大小；
2. 每个特征值对应的方向在干什么。

对动力系统来说尤其如此。因为如果
$$
\dot{\mathbf a}=-M\mathbf a,
$$
那么沿着第 \(i\) 个特征方向的衰减速度就是
$$
e^{-\lambda_i t}.
$$
这就是为什么谱和“学习快慢”天然联系在一起。

### 1.5.3 经验谱分布、Stieltjes 变换与 resolvent

对一个 \(p\times p\) 对称矩阵 \(M\)，定义它的经验谱分布
$$
\mu_M
:=
\frac1p\sum_{i=1}^p \delta_{\lambda_i(M)}.
$$

意思很简单：把每个特征值都当成一颗点质量，再平均。

如果 \(p\) 很大，我们关心的不是每个单独特征值，而是整体分布长什么样。这时最方便的工具就是 **resolvent**：
$$
G_M(z):=(M-zI)^{-1},
\qquad z\in\mathbb C\setminus \operatorname{Sp}(M).
$$

以及它的归一化迹，也就是 **Stieltjes 变换**：
$$
m_M(z)
:=
\frac1p\operatorname{Tr}(M-zI)^{-1}
=
\frac1p\sum_{i=1}^p \frac{1}{\lambda_i-z}.
$$

你可以记住两个最重要的事实：

1. 如果 \(\lambda_0\) 附近堆了很多特征值，那么 \(m_M(z)\) 在 \(z\approx\lambda_0\) 时会有明显响应。
2. 如果 \(\mu_M\) 在某一点有一个 delta 峰，\(m_M(z)\) 会出现一个极点。

更准确地说，如果
$$
\mu_M = f\,\delta_{\lambda_0}+\text{(其他连续部分)},
$$
那么
$$
m_M(z)=\frac{f}{\lambda_0-z}+\text{(一个在 \(z=\lambda_0\) 附近有界的项)}.
$$

这正是后面论文用来识别 \(s_t^2\) 处 delta 峰的原因。

### 1.5.4 为什么取虚部就能“看见”谱密度

如果把 \(z\) 取成
$$
z=\lambda+i\varepsilon,\qquad \varepsilon>0,
$$
则
$$
\frac{1}{x-(\lambda+i\varepsilon)}
=
\frac{x-\lambda}{(x-\lambda)^2+\varepsilon^2}
+
i\frac{\varepsilon}{(x-\lambda)^2+\varepsilon^2}.
$$

注意虚部：
$$
\operatorname{Im}\frac{1}{x-(\lambda+i\varepsilon)}
=
\frac{\varepsilon}{(x-\lambda)^2+\varepsilon^2}.
$$

当 \(\varepsilon\downarrow0\) 时，它会变成以 \(\lambda\) 为中心的尖峰，极限上等价于 \(\pi\delta(x-\lambda)\)。因此
$$
\rho(\lambda)
=
\lim_{\varepsilon\downarrow0}
\frac{1}{\pi}\operatorname{Im}m_M(\lambda+i\varepsilon)
$$
就能把谱密度恢复出来。

这就是为什么随机矩阵文献里几乎总是先求 resolvent / Stieltjes 变换，而不是直接求特征值密度。

### 1.5.5 bulk、atom、support 这些词到底是什么意思

随机矩阵里有三个词你会反复看到：

- **support**：分布真正有质量的区间。
- **bulk**：一整片连续特征值云，通常表示很多特征值形成的连续团块。
- **atom / delta peak**：某个点上有宏观权重的尖峰，也就是
$$
f\,\delta(\lambda-\lambda_0).
$$

所以论文说
$$
\rho(\lambda)
=
\left(1-\frac{1+\psi_n}{\psi_p}\right)\delta(\lambda-s_t^2)
+
\frac{\psi_n}{\psi_p}\rho_1(\lambda)
+
\frac{1}{\psi_p}\rho_2(\lambda)
$$
时，意思就是：

- 有一大批特征值几乎就卡在 \(s_t^2\) 这个点上；
- 还有两个连续的特征值团块，分别叫 \(\rho_1\) 和 \(\rho_2\)。

### 1.5.6 Marchenko--Pastur 定律为什么总会冒出来

如果 \(X\in\mathbb R^{d\times n}\) 的列是 iid 高斯向量，那么样本协方差矩阵
$$
\frac{XX^\top}{n}
$$
在 \(d,n\to\infty\)、\(n/d\to\psi_n\) 时，其特征值不会乱跑，而是集中到一个确定的极限分布，这就是 Marchenko--Pastur 定律。

最经典的情形是总体协方差是单位阵。此时 bulk 支撑大致在
$$
\left(1-\sqrt{\frac{d}{n}}\right)^2
\quad\text{到}\quad
\left(1+\sqrt{\frac{d}{n}}\right)^2
$$
之间。

你不需要死记它的密度公式，但一定要抓住它背后的物理意义：

- 随机样本协方差不会只有一个特征值尺度；
- 它会形成一个明确的 bulk；
- bulk 的左右边界通常能直接决定学习动力学里“最快 / 最慢”模态的时间尺度。

后面论文中
$$
\lambda_\pm
=
s_t^2+v_t^2\left(1\pm\sqrt{\frac{\psi_p}{\psi_n}}\right)^2
$$
这个边界形式，本质上就是一个经过平移和缩放的 Marchenko--Pastur 型边界。

### 1.5.7 rank 为什么能决定 delta 峰的权重

还有一个后面会大量用到的简单线性代数事实：

如果矩阵 \(M\in\mathbb R^{p\times p}\) 的秩最多是 \(r\)，那么它至少有
$$
p-r
$$
个零特征值。

更一般地，如果
$$
M=s_t^2 I + \text{(一个秩不超过 \(r\) 的扰动)},
$$
那么至少会有
$$
p-r
$$
个特征值等于 \(s_t^2\)。

这就是论文里 delta 峰为什么能通过“自由度计数”直接读出来：

- \(W\) 带来至多 \(d\) 个非平凡方向；
- 数据项带来至多 \(n\) 个非平凡方向；
- 剩下的 \(p-d-n\) 个方向什么也没学到，只看到 \(s_t^2 I\)。

### 1.5.8 论文里真正想证明的东西，其实可以先用一句话记住

如果把训练写成线性动力系统，那么每个特征模态的学习速度都正比于对应特征值的大小。

因此：

- 大特征值对应快模态，先学会；
- 小特征值对应慢模态，后学会。

而论文的全部内容，本质上就是说明：

1. “总体结构”对应一团较大的特征值；
2. “训练集特异的高频细节”对应一团较小的特征值；
3. 所以模型会先学会泛化，再学会记忆。

理解了这一点，后面所有公式都是在把这句话严格化。

---

## 2. 扩散模型基本设定：从前向 OU 到 score learning

### 2.1 前向过程

论文使用的连续时间前向过程是 Ornstein--Uhlenbeck SDE：
$$
\mathrm d \mathbf x = -\mathbf x(t)\,\mathrm dt + \mathrm d \mathbf B(t),
$$
其中 \(\mathrm d\mathbf B(t)\) 表示方差为 \(2\,\mathrm dt\) 的布朗增量。

这个 SDE 的显式解是
$$
\mathbf x_t = e^{-t}\mathbf x_0 + \sqrt{\Delta_t}\,\boldsymbol\xi,
\qquad
\Delta_t:=1-e^{-2t},
\qquad
\boldsymbol\xi\sim\mathcal N(0,\mathbf I_d).
$$

这是 OU 过程的标准解。推导如下。

对
$$
\mathrm d\mathbf x_t + \mathbf x_t\,\mathrm dt = \mathrm d\mathbf B_t
$$
乘积分因子 \(e^t\)，得到
$$
\mathrm d(e^t\mathbf x_t)=e^t\,\mathrm d\mathbf B_t.
$$

从 \(0\) 积分到 \(t\)：
$$
e^t\mathbf x_t-\mathbf x_0=\int_0^t e^s\,\mathrm d\mathbf B_s.
$$

所以
$$
\mathbf x_t=e^{-t}\mathbf x_0+e^{-t}\int_0^t e^s\,\mathrm d\mathbf B_s.
$$

由于右边第二项是零均值高斯，其协方差为
$$
2e^{-2t}\int_0^t e^{2s}\,\mathrm ds\;\mathbf I_d
=
2e^{-2t}\cdot \frac{e^{2t}-1}{2}\,\mathbf I_d
=
(1-e^{-2t})\mathbf I_d
=
\Delta_t\mathbf I_d,
$$
故可写成
$$
\mathbf x_t=e^{-t}\mathbf x_0+\sqrt{\Delta_t}\,\boldsymbol\xi.
$$

### 2.2 反向过程

若记前向过程中时间 \(t\) 的密度为 \(P_t(\mathbf x)\)，score 定义为
$$
\mathbf s(\mathbf x,t)=\nabla_{\mathbf x}\log P_t(\mathbf x),
$$
则反向 SDE 写为
$$
-\mathrm d \mathbf x
=
\big[\mathbf x(t)+2\mathbf s(\mathbf x,t)\big]\mathrm dt
+
\mathrm d\mathbf B(t).
$$

这就是标准 score-based diffusion 的反向生成公式。

### 2.3 把 \(P_t\) 明确写成卷积形式

为了后面不跳步，我们把“\(P_t\) 是什么”写得再明确一点。

如果原始数据 \(\mathbf x_0\sim P_0\)，且
$$
\mathbf x_t=e^{-t}\mathbf x_0+\sqrt{\Delta_t}\,\boldsymbol\xi,
\qquad
\boldsymbol\xi\sim\mathcal N(0,I_d),
$$
那么条件分布
$$
P(\mathbf x_t=\mathbf y\mid \mathbf x_0=\mathbf x_0)
$$
是一个高斯密度：
$$
\frac{1}{(2\pi\Delta_t)^{d/2}}
\exp\!\left(
-\frac{\|\mathbf y-e^{-t}\mathbf x_0\|_2^2}{2\Delta_t}
\right).
$$

因此 \(P_t\) 就是把 \(P_0\) 用这个高斯核做卷积：
$$
P_t(\mathbf y)
=
\int_{\mathbb R^d}
P_0(\mathbf x_0)\,
\frac{1}{(2\pi\Delta_t)^{d/2}}
\exp\!\left(
-\frac{\|\mathbf y-e^{-t}\mathbf x_0\|_2^2}{2\Delta_t}
\right)
\mathrm d\mathbf x_0.
$$

这个式子很重要，因为：

- 如果把 \(P_0\) 换成经验分布 \(\frac1n\sum_\nu \delta_{\mathbf x^\nu}\)，就直接得到经验密度 \(P_t^{\mathrm{emp}}\)；
- 对这个积分式直接求梯度，就能得到 Tweedie 型公式和经验 score 的显式表达。

所以你可以把扩散模型里时间 \(t\) 的密度理解成：

> 先把所有原始样本缩小到 \(e^{-t}\mathbf x_0\) 的位置，再在它们周围糊上一层方差为 \(\Delta_t\) 的高斯云。

---

## 3. 为什么 score matching 的极小化器就是真实 score

论文用的训练目标是
$$
\hat{\mathbf s}(\mathbf x,t)
=
\arg\min_{\mathbf s}
\mathbb E_{\mathbf x_0\sim P_0,\boldsymbol\xi\sim\mathcal N(0,\mathbf I_d)}
\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s(\mathbf x_t,t)+\boldsymbol\xi
\right\|_2^2
\right].
$$

这里
$$
\mathbf x_t=e^{-t}\mathbf x_0+\sqrt{\Delta_t}\,\boldsymbol\xi.
$$

这个式子看起来像 denoising score matching，但如果只记结论很容易糊。下面把极小化器推导一遍。

### 3.1 条件化到 \(\mathbf x_t\)

定义泛函
$$
\mathcal J[\mathbf s]
:=
\mathbb E\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s(\mathbf x_t,t)+\boldsymbol\xi
\right\|_2^2
\right].
$$

对 \(\mathbf y=\mathbf x_t\) 条件化：
$$
\mathcal J[\mathbf s]
=
\mathbb E_{\mathbf y}
\left[
\mathbb E\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s(\mathbf y,t)+\boldsymbol\xi
\right\|_2^2
\mid \mathbf x_t=\mathbf y
\right]
\right].
$$

对固定 \(\mathbf y\)，内层期望等于
$$
\Delta_t\|\mathbf s(\mathbf y,t)\|_2^2
+
2\sqrt{\Delta_t}\,\mathbf s(\mathbf y,t)\cdot
\mathbb E[\boldsymbol\xi\mid \mathbf x_t=\mathbf y]
+
\mathbb E[\|\boldsymbol\xi\|_2^2\mid \mathbf x_t=\mathbf y].
$$

最后一项与 \(\mathbf s\) 无关，因此对 \(\mathbf s\) 的点态最优条件是
$$
2\Delta_t\,\mathbf s^\star(\mathbf y,t)
+
2\sqrt{\Delta_t}\,\mathbb E[\boldsymbol\xi\mid \mathbf x_t=\mathbf y]
=0.
$$

故
$$
\boxed{
\mathbf s^\star(\mathbf y,t)
=
-\frac{1}{\sqrt{\Delta_t}}
\mathbb E[\boldsymbol\xi\mid \mathbf x_t=\mathbf y]
}
$$

### 3.2 把 \(\mathbb E[\boldsymbol\xi\mid \mathbf x_t]\) 改写成 score

因为
$$
\mathbf x_t=e^{-t}\mathbf x_0+\sqrt{\Delta_t}\,\boldsymbol\xi,
$$
所以
$$
\boldsymbol\xi
=
\frac{\mathbf x_t-e^{-t}\mathbf x_0}{\sqrt{\Delta_t}}.
$$

对 \(\mathbf x_t=\mathbf y\) 条件化：
$$
\mathbb E[\boldsymbol\xi\mid \mathbf x_t=\mathbf y]
=
\frac{\mathbf y-e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]}{\sqrt{\Delta_t}}.
$$

代回上式可得
$$
\mathbf s^\star(\mathbf y,t)
=
-\frac{1}{\Delta_t}
\left(
\mathbf y-e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]
\right).
$$

另一方面，\(P_t\) 是 \(P_0\) 与高斯核卷积后的密度。对高斯卷积做求导可得 Tweedie 型恒等式
$$
\nabla_{\mathbf y}\log P_t(\mathbf y)
=
-\frac{1}{\Delta_t}
\left(
\mathbf y-e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]
\right).
$$

因此
$$
\boxed{
\mathbf s^\star(\mathbf y,t)=\nabla_{\mathbf y}\log P_t(\mathbf y)
}
$$

这说明 score matching 确实在学真实 score。

### 3.3 Tweedie 型恒等式不跳步推导

上面我直接用了
$$
\nabla_{\mathbf y}\log P_t(\mathbf y)
=
-\frac{1}{\Delta_t}
\left(
\mathbf y-e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]
\right),
$$
这里我们把它完整推一遍。

从上一节的卷积公式出发：
$$
P_t(\mathbf y)
=
\int
P_0(\mathbf x_0)\,
K_t(\mathbf y,\mathbf x_0)\,
\mathrm d\mathbf x_0,
$$
其中高斯核
$$
K_t(\mathbf y,\mathbf x_0)
:=
\frac{1}{(2\pi\Delta_t)^{d/2}}
\exp\!\left(
-\frac{\|\mathbf y-e^{-t}\mathbf x_0\|_2^2}{2\Delta_t}
\right).
$$

对 \(\mathbf y\) 求梯度：
$$
\nabla_{\mathbf y}P_t(\mathbf y)
=
\int
P_0(\mathbf x_0)\,
\nabla_{\mathbf y}K_t(\mathbf y,\mathbf x_0)\,
\mathrm d\mathbf x_0.
$$

而
$$
\nabla_{\mathbf y}K_t(\mathbf y,\mathbf x_0)
=
K_t(\mathbf y,\mathbf x_0)\cdot
\left(
-\frac{\mathbf y-e^{-t}\mathbf x_0}{\Delta_t}
\right).
$$

所以
$$
\nabla_{\mathbf y}P_t(\mathbf y)
=
\int
P_0(\mathbf x_0)\,
K_t(\mathbf y,\mathbf x_0)\,
\left(
-\frac{\mathbf y-e^{-t}\mathbf x_0}{\Delta_t}
\right)
\mathrm d\mathbf x_0.
$$

两边除以 \(P_t(\mathbf y)\)：
$$
\nabla_{\mathbf y}\log P_t(\mathbf y)
=
\frac{
\int
P_0(\mathbf x_0)\,
K_t(\mathbf y,\mathbf x_0)\,
\left(
-\frac{\mathbf y-e^{-t}\mathbf x_0}{\Delta_t}
\right)
\mathrm d\mathbf x_0
}{
\int
P_0(\mathbf x_0)\,
K_t(\mathbf y,\mathbf x_0)\,
\mathrm d\mathbf x_0
}.
$$

而 Bayes 公式告诉我们：
$$
P(\mathbf x_0\mid \mathbf x_t=\mathbf y)
\propto
P_0(\mathbf x_0)K_t(\mathbf y,\mathbf x_0).
$$

因此上式正好就是对后验分布取期望：
$$
\nabla_{\mathbf y}\log P_t(\mathbf y)
=
\mathbb E\left[
-\frac{\mathbf y-e^{-t}\mathbf x_0}{\Delta_t}
\;\middle|\;
\mathbf x_t=\mathbf y
\right].
$$

把 \(\mathbf y\) 提出期望号外面，就得到
$$
\boxed{
\nabla_{\mathbf y}\log P_t(\mathbf y)
=
-\frac{1}{\Delta_t}
\left(
\mathbf y-e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]
\right)
}
$$

这个恒等式有一个非常重要的直觉：

- score 不是一个神秘黑盒；
- 它就是“当前点 \(\mathbf y\) 与后验均值 \(e^{-t}\mathbb E[\mathbf x_0\mid \mathbf x_t=\mathbf y]\) 之间的偏差”除以 \(\Delta_t\)。

也就是说，score 本质上在告诉你：

> 如果我现在在 \(\mathbf y\)，从贝叶斯角度看，最可能对应的干净数据位置在哪里？我应该往哪个方向拉回去？

---

## 4. 经验损失的极小化器为什么会导致记忆

实际训练时，不是对总体分布 \(P_0\) 做期望，而是对训练集
$$
\{\mathbf x^\nu\}_{\nu=1}^n
$$
做经验平均：
$$
\mathcal L_t(\theta;\{\mathbf x^\nu\}_{\nu=1}^n)
=
\frac{1}{n}
\sum_{\nu=1}^n
\mathbb E_{\boldsymbol\xi}
\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s_\theta(\mathbf x_t^\nu,t)+\boldsymbol\xi
\right\|_2^2
\right],
$$
其中
$$
\mathbf x_t^\nu=e^{-t}\mathbf x^\nu+\sqrt{\Delta_t}\,\boldsymbol\xi.
$$

它的全局最优解是 **经验 score**
$$
\mathbf s_{\mathrm{emp}}(\mathbf x,t)
=
\nabla_{\mathbf x}\log P_t^{\mathrm{emp}}(\mathbf x),
$$
其中
$$
P_t^{\mathrm{emp}}(\mathbf x)
=
\frac{1}{n(2\pi\Delta_t)^{d/2}}
\sum_{\nu=1}^n
\exp\!\left(
-\frac{\|\mathbf x-e^{-t}\mathbf x^\nu\|_2^2}{2\Delta_t}
\right).
$$

### 4.1 把经验 score 写显式

先对 \(P_t^{\mathrm{emp}}\) 求梯度：
$$
\nabla_{\mathbf x}P_t^{\mathrm{emp}}(\mathbf x)
=
\frac{1}{n(2\pi\Delta_t)^{d/2}}
\sum_{\nu=1}^n
\exp\!\left(
-\frac{\|\mathbf x-e^{-t}\mathbf x^\nu\|_2^2}{2\Delta_t}
\right)
\left(
-\frac{\mathbf x-e^{-t}\mathbf x^\nu}{\Delta_t}
\right).
$$

所以
$$
\mathbf s_{\mathrm{emp}}(\mathbf x,t)
=
\frac{
\sum_{\nu=1}^n
\omega_\nu(\mathbf x,t)\,
\dfrac{e^{-t}\mathbf x^\nu-\mathbf x}{\Delta_t}
}{
\sum_{\nu=1}^n \omega_\nu(\mathbf x,t)
},
$$
其中
$$
\omega_\nu(\mathbf x,t)
:=
\exp\!\left(
-\frac{\|\mathbf x-e^{-t}\mathbf x^\nu\|_2^2}{2\Delta_t}
\right).
$$

这个式子非常关键。它说明经验 score 是一个由训练样本诱导出的“吸引场”。

当 \(t\) 很小、\(\Delta_t\) 很小时：

- 每个高斯核都非常尖；
- \(\mathbf s_{\mathrm{emp}}\) 在每个训练样本附近都会出现局部吸引；
- 反向生成过程容易被训练样本吸过去。

所以如果模型真的学到经验 score，它最终就会记忆训练集。

这也是论文的起点：**问题不在于经验 score 不会记忆，而在于实际训练为何没有立刻学到它。**

### 4.2 小噪声极限下，经验 score 为什么一定像“弹簧力”

为了更清楚地看到经验 score 为什么会记忆，我们看一个局部极限。

假设 \(\mathbf x\) 非常接近某个训练样本 \(\mathbf x^\mu\) 的缩放位置 \(e^{-t}\mathbf x^\mu\)，并且比起其他训练样本，它对应的高斯权重远大于别的项：
$$
\omega_\mu(\mathbf x,t)\gg \omega_\nu(\mathbf x,t),
\qquad \nu\neq \mu.
$$

这时分子和分母都由 \(\mu\) 这一项主导，因此
$$
\mathbf s_{\mathrm{emp}}(\mathbf x,t)
\approx
\frac{
\omega_\mu(\mathbf x,t)\dfrac{e^{-t}\mathbf x^\mu-\mathbf x}{\Delta_t}
}{
\omega_\mu(\mathbf x,t)
}
=
\frac{e^{-t}\mathbf x^\mu-\mathbf x}{\Delta_t}.
$$

也就是说，在训练样本附近，
$$
\boxed{
\mathbf s_{\mathrm{emp}}(\mathbf x,t)
\approx
-\frac{1}{\Delta_t}
\big(\mathbf x-e^{-t}\mathbf x^\mu\big)
}
$$

这就是一个标准的线性回复力，也就是“弹簧力”：

- 如果 \(\mathbf x\) 在训练样本右边，score 就把你往左拉；
- 如果 \(\mathbf x\) 在训练样本左边，score 就把你往右拉；
- 拉回强度正比于偏离量；
- 比例系数是 \(1/\Delta_t\)。

所以当 \(t\) 很小时，\(\Delta_t\) 很小，回复力会非常大：
$$
\frac{1}{\Delta_t}\to\infty.
$$

这就解释了为什么小噪声时经验 score 会极强地吸附训练样本。

### 4.3 一维直觉：经验 score = 一堆局部吸引井叠加

如果你更喜欢一维图像化理解，可以把
$$
P_t^{\mathrm{emp}}(x)
=
\frac1n\sum_{\nu=1}^n \mathcal N(x;e^{-t}x^\nu,\Delta_t)
$$
想成在数轴上放了 \(n\) 个高斯包。

那么
$$
s_{\mathrm{emp}}(x,t)=\partial_x \log P_t^{\mathrm{emp}}(x)
$$
就是这个“高斯包混合密度”的对数导数。

在每个训练样本附近，它都像一个把粒子往中心拉的向量场；远离所有样本时，不同高斯包的贡献互相抵消，场会变弱。

所以经验 score 的向量场像是：

- 在每个训练点附近有一个稳定吸引井；
- 井与井之间可能存在分界面；
- 反向生成时，轨道容易掉进某个训练样本的 basin of attraction。

这就是论文说“如果真的学到经验 score，就必然会记忆”的几何图像。

---

## 5. 论文的实验发现：两阶段训练动力学

### 5.1 记忆指标

论文把一个生成样本 \(\mathbf x_\tau\) 判为“memorized”的标准是
$$
\mathbb E_{\mathbf x_\tau}
\left[
\frac{
\|\mathbf x_\tau-\mathbf a^{\mu_1}\|_2
}{
\|\mathbf x_\tau-\mathbf a^{\mu_2}\|_2
}
\right]
<k,
$$
其中：

- \(\mathbf a^{\mu_1}\) 是训练集里最近邻；
- \(\mathbf a^{\mu_2}\) 是第二近邻；
- 论文用 \(k=\frac13\)。

这个指标的直觉是：

- 如果生成样本真的是新样本，它通常不会离某个训练样本极端接近；
- 如果生成样本几乎就是训练图的复现，那么它到最近邻的距离会远小于到第二近邻的距离。

### 5.2 在真实 U-Net 上的发现

论文在 CelebA 灰度 \(32\times 32\) 上训练 U-Net DDPM，得到如下经验规律：

1. 在固定模型容量 \(p\approx 4\times 10^6\) 时，
$$
\tau_{\mathrm{gen}}\approx 10^5 \text{ steps},
$$
几乎不随 \(n\) 变化。

2. 记忆出现时间近似满足
$$
\tau_{\mathrm{mem}}\propto n.
$$

3. 这个结论不是“每个样本被重复看得更少”造成的。因为就算使用 full-batch 更新 \(B=n\)，仍然有
$$
\tau_{\mathrm{mem}}\propto n.
$$

4. 若改变模型宽度 \(W\)，则作者观察到
$$
\tau_{\mathrm{gen}}\propto W^{-1},\qquad
\tau_{\mathrm{mem}}\propto nW^{-1}.
$$

所以在真实模型上，经验上有一个非常干净的相图：

- 模型越宽，学习越快，也越早开始记忆；
- 数据集越大，记忆时间越晚；
- 因而会出现一个随 \(n\) 变宽的早停泛化窗口。

![CelebA 上记忆时间随训练集大小线性推迟](../../assets/papers/why-diffusion-models-dont-memorize/celeba-memorization-vs-n.png)

*关键实验图：在固定 U-Net 容量下，FID 很快达到稳定，而记忆比例的上升明显滞后；插图里的时间重标度显示 \(\tau_{\mathrm{mem}}\propto n\)。*

![CelebA 上时间尺度随模型宽度变化](../../assets/papers/why-diffusion-models-dont-memorize/celeba-width-scaling.png)

*关键实验图：模型越宽，\(\tau_{\mathrm{gen}}\) 和 \(\tau_{\mathrm{mem}}\) 都越早发生；作者总结为 \(\tau_{\mathrm{gen}}\propto W^{-1}\)、\(\tau_{\mathrm{mem}}\propto nW^{-1}\)。*

---

## 6. 理论模型：固定扩散时间下的随机特征 score 网络

为了把上面的现象解析出来，作者引入一个固定扩散时间 \(t\) 的随机特征网络：
$$
\mathbf s_{\mathbf A}(\mathbf x)
=
\frac{\mathbf A}{\sqrt p}
\sigma\!\left(\frac{\mathbf W\mathbf x}{\sqrt d}\right),
$$
其中

- \(\mathbf W\in\mathbb R^{p\times d}\) 是随机高斯第一层，训练时冻结；
- \(\mathbf A\in\mathbb R^{d\times p}\) 是第二层权重，训练；
- \(\sigma\) 是逐坐标非线性。

记高维比例
$$
\psi_p=\frac{p}{d},\qquad
\psi_n=\frac{n}{d},
$$
并取
$$
d,p,n\to\infty,\qquad \psi_p,\psi_n \text{ 固定}.
$$

这个模型不再是“同一个网络同时拟合全部扩散时间”，而是固定 \(t\) 单独研究。这当然比真实 DDPM 简化，但它保留了论文想抓住的核心机制：**score learning 的谱分解与时间尺度分离。**

### 6.1 把每个对象的尺寸都写出来

这里最容易让人迷糊的是矩阵尺寸。我们把它们全部写清楚。

- 输入数据
$$
\mathbf x\in\mathbb R^d.
$$

- 冻结的随机第一层
$$
\mathbf W\in\mathbb R^{p\times d}.
$$
所以
$$
\frac{\mathbf W\mathbf x}{\sqrt d}\in\mathbb R^p.
$$

- 非线性特征向量
$$
\sigma\!\left(\frac{\mathbf W\mathbf x}{\sqrt d}\right)\in\mathbb R^p.
$$

- 可训练第二层
$$
\mathbf A\in\mathbb R^{d\times p}.
$$
因此
$$
\frac{\mathbf A}{\sqrt p}\sigma\!\left(\frac{\mathbf W\mathbf x}{\sqrt d}\right)\in\mathbb R^d,
$$
正好是一个 \(d\) 维向量，和 score 的维度一致。

- 训练相关矩阵
$$
\mathbf U\in\mathbb R^{p\times p},
\qquad
\mathbf V\in\mathbb R^{p\times d}.
$$

所以
$$
\mathbf V^\top\in\mathbb R^{d\times p},
\qquad
\mathbf V^\top \mathbf U^{-1}\in\mathbb R^{d\times p},
$$
和 \(\mathbf A/\sqrt p\) 的尺寸正好一致。

这一步虽然看起来“基础”，但非常重要。很多论文里矩阵推导之所以读不下去，不是因为算不动，而是因为中间一步压根没确认矩阵乘法是否合法。

### 6.2 为什么这个模型虽然简化，但仍然抓住了本质

这个随机特征模型做了两件极大的简化：

1. 第一层权重不训练；
2. 只固定一个扩散时间 \(t\)。

但它保留了两件对本文最重要的结构：

- 训练目标依然是 score matching；
- 训练动力学依然可以写成“线性系统 + 某个数据相关相关矩阵 \(\mathbf U\)”。

论文真正想解释的不是 U-Net 的每一层到底怎么学，而是：

> 为什么存在一批快模态先把总体结构学出来，另一批慢模态很久以后才把训练集细节学出来？

只要这个核心结构还在，简化模型就仍然有解释力。

---

## 7. 随机特征网络的训练 / 测试损失

固定扩散时间 \(t\) 后，训练损失写成
$$
\mathcal L_{\mathrm{train}}
=
\frac{1}{d}\cdot
\frac{1}{n}
\sum_{\nu=1}^n
\mathbb E_{\boldsymbol\xi}
\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s_{\mathbf A}(\mathbf x_t^\nu)
+
\boldsymbol\xi
\right\|_2^2
\right].
$$

测试损失和真实 score 误差定义为
$$
\mathcal L_{\mathrm{test}}
=
\frac{1}{d}
\mathbb E_{\mathbf x,\boldsymbol\xi}
\left[
\left\|
\sqrt{\Delta_t}\,\mathbf s_{\mathbf A}(\mathbf x_t)+\boldsymbol\xi
\right\|_2^2
\right],
$$
$$
\mathcal E_{\mathrm{score}}
=
\frac{1}{d}\mathbb E_{\mathbf x}
\left[
\left\|
\mathbf s_{\mathbf A}(\mathbf x)-\nabla\log P_{\mathbf x}(\mathbf x)
\right\|_2^2
\right].
$$

其中 \(\mathcal L_{\mathrm{gen}}=\mathcal L_{\mathrm{test}}-\mathcal L_{\mathrm{train}}\) 衡量过拟合。

---

## 8. 梯度流：为什么训练时间尺度完全由 \(\mathbf U\) 的谱决定

### 8.1 定义两个矩阵 \(\mathbf U,\mathbf V\)

作者定义
$$
\mathbf U
=
\frac{1}{n}\sum_{\nu=1}^n
\mathbb E_{\boldsymbol\xi}
\left[
\sigma\!\left(\frac{\mathbf W\mathbf x_t^\nu}{\sqrt d}\right)
\sigma\!\left(\frac{\mathbf W\mathbf x_t^\nu}{\sqrt d}\right)^{\!\top}
\right],
$$
$$
\mathbf V
=
\frac{1}{n}\sum_{\nu=1}^n
\mathbb E_{\boldsymbol\xi}
\left[
\sigma\!\left(\frac{\mathbf W\mathbf x_t^\nu}{\sqrt d}\right)
\boldsymbol\xi^{\top}
\right].
$$

### 8.2 先把训练损失展开

设
$$
\mathbf z^\nu:=\sigma\!\left(\frac{\mathbf W\mathbf x_t^\nu}{\sqrt d}\right).
$$

则
$$
\mathbf s_{\mathbf A}(\mathbf x_t^\nu)=\frac{\mathbf A}{\sqrt p}\mathbf z^\nu.
$$

把平方展开：
$$
\left\|
\sqrt{\Delta_t}\frac{\mathbf A}{\sqrt p}\mathbf z^\nu+\boldsymbol\xi
\right\|_2^2
=
\Delta_t\,\mathbf z^{\nu\top}\frac{\mathbf A^\top\mathbf A}{p}\mathbf z^\nu
+
2\sqrt{\Delta_t}\,\boldsymbol\xi^\top\frac{\mathbf A}{\sqrt p}\mathbf z^\nu
+
\|\boldsymbol\xi\|_2^2.
$$

对 \(\nu\) 和 \(\boldsymbol\xi\) 取平均后：
$$
\mathcal L_{\mathrm{train}}(\mathbf A)
=
1
+
\frac{\Delta_t}{d}\operatorname{Tr}
\left(
\frac{\mathbf A^\top\mathbf A}{p}\mathbf U
\right)
+
\frac{2\sqrt{\Delta_t}}{d}
\operatorname{Tr}
\left(
\frac{\mathbf A}{\sqrt p}\mathbf V
\right).
$$

这里常数项 \(1\) 来自 \(\frac1d\mathbb E\|\boldsymbol\xi\|_2^2=1\)。

### 8.3 对 \(\mathbf A\) 求梯度

利用矩阵求导公式
$$
\nabla_{\mathbf A}\operatorname{Tr}(\mathbf A^\top\mathbf A\mathbf U)
=
2\mathbf A\mathbf U
\quad(\mathbf U=\mathbf U^\top),
$$
得到
$$
\nabla_{\mathbf A}\mathcal L_{\mathrm{train}}(\mathbf A)
=
\frac{2\Delta_t}{d}\frac{\mathbf A}{p}\mathbf U
+
\frac{2\sqrt{\Delta_t}}{d\sqrt p}\mathbf V^\top.
$$

### 8.4 梯度流方程

论文把离散 GD
$$
\mathbf A^{(k+1)}
=
\mathbf A^{(k)}-\eta\nabla_{\mathbf A}\mathcal L_{\mathrm{train}}(\mathbf A^{(k)})
$$
在 \(\eta\to0\) 极限下写成连续时间，并把训练时间缩放为
$$
\tau=\frac{k\eta}{d^2}.
$$

于是梯度流为
$$
\dot{\mathbf A}(\tau)
=
-d^2\nabla_{\mathbf A}\mathcal L_{\mathrm{train}}(\mathbf A(\tau))
=
-2\Delta_t\frac{d}{p}\mathbf A\mathbf U
-\frac{2d\sqrt{\Delta_t}}{\sqrt p}\mathbf V^\top.
$$

由于
$$
\frac{d}{p}=\frac{1}{\psi_p},
$$
这个式子本质上是一个线性矩阵 ODE。

### 8.5 闭式解

设
$$
\mathbf B(\tau):=\frac{\mathbf A(\tau)}{\sqrt p},
$$
则
$$
\dot{\mathbf B}
=
-\frac{2\Delta_t}{\psi_p}\mathbf B\mathbf U
-\frac{2\sqrt{\Delta_t}}{\psi_p}\mathbf V^\top.
$$

这是标准的一阶线性 ODE。稳态解 \(\mathbf B_\star\) 满足
$$
\mathbf B_\star\mathbf U
=
-\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top,
$$
所以
$$
\mathbf B_\star
=
-\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}.
$$

因此通解为
$$
\boxed{
\frac{\mathbf A(\tau)}{\sqrt p}
=
-\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}
+
\left(
\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}
+
\frac{\mathbf A_0}{\sqrt p}
\right)
\exp\!\left(-\frac{2\Delta_t}{\psi_p}\mathbf U\,\tau\right)
}
$$

若初始化 \(\mathbf A_0=0\)，则
$$
\frac{\mathbf A(\tau)}{\sqrt p}
=
\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}
\left(
\exp\!\left(-\frac{2\Delta_t}{\psi_p}\mathbf U\,\tau\right)-\mathbf I
\right).
$$

### 8.6 关键结论：训练时间尺度 = \(\mathbf U\) 特征值的倒数

把 \(\mathbf U\) 对角化：
$$
\mathbf U=\sum_{\lambda\in \operatorname{Sp}(\mathbf U)}\lambda\,\mathbf v_\lambda\mathbf v_\lambda^\top.
$$

则矩阵指数变成
$$
\exp\!\left(-\frac{2\Delta_t}{\psi_p}\mathbf U\tau\right)
=
\sum_\lambda
\exp\!\left(-\frac{2\Delta_t}{\psi_p}\lambda\tau\right)
\mathbf v_\lambda\mathbf v_\lambda^\top.
$$

因此每个特征模态 \(\lambda\) 的松弛时间是
$$
\tau_\lambda
\asymp
\frac{\psi_p}{\Delta_t\,\lambda}.
$$

所以整篇理论部分都归结为一个问题：

> \(\mathbf U\) 的谱长什么样？

### 8.7 用一维线性 ODE 理解“特征值越小学得越慢”

如果你觉得矩阵指数还是抽象，我们先退回一维。

考虑最简单的 ODE：
$$
\dot a(\tau)=-\lambda\big(a(\tau)-a_\star\big),
$$
其中 \(a_\star\) 是目标值，\(\lambda>0\)。

它的解是
$$
a(\tau)=a_\star + (a(0)-a_\star)e^{-\lambda \tau}.
$$

这里 \(\lambda\) 越大，指数衰减越快，系统越快接近稳态；\(\lambda\) 越小，系统越慢。

矩阵情形只不过是把这个一维故事拆成很多彼此正交的特征方向：

- 在第 \(i\) 个特征方向上，系统就像
$$
\dot a_i(\tau)=-\frac{2\Delta_t}{\psi_p}\lambda_i\big(a_i(\tau)-a_{i,\star}\big),
$$
- 因此时间尺度就是
$$
\tau_i\sim \frac{\psi_p}{\Delta_t\lambda_i}.
$$

所以“学习速度由谱决定”本质上只是：

> 每个特征方向都是一个独立的一维指数松弛，而小特征值对应慢松弛。

---

## 9. Gaussian equivalence：把 \(\mathbf U\) 化成可分析的随机矩阵

这一部分是论文的技术核心。

### 9.1 定义几个系数

记
$$
\sigma_{\mathbf x}^2:=\frac{\operatorname{Tr}(\mathbf\Sigma)}{d},
\qquad
\mathbf\Sigma:=\mathbb E[\mathbf x\mathbf x^\top],
$$
以及
$$
\Gamma_t^2=e^{-2t}\sigma_{\mathbf x}^2+\Delta_t.
$$

作者定义
$$
b_t
=
\mathbb E_{u,v}
\Big[
v\,\sigma(e^{-t}\sigma_{\mathbf x}u+\sqrt{\Delta_t}v)
\Big],
$$
$$
a_t
=
\mathbb E_{u,v}
\left[
\sigma(e^{-t}\sigma_{\mathbf x}u+\sqrt{\Delta_t}v)
\frac{u}{e^{-t}\sigma_{\mathbf x}}
\right],
$$
$$
v_t^2
=
\mathbb E_{u,v,w}
\Big[
\sigma(e^{-t}\sigma_{\mathbf x}u+\sqrt{\Delta_t}v)
\sigma(e^{-t}\sigma_{\mathbf x}u+\sqrt{\Delta_t}w)
\Big]
-a_t^2e^{-2t}\sigma_{\mathbf x}^2,
$$
$$
s_t^2
=
\mathbb E_u[\sigma(\Gamma_tu)^2]
-a_t^2e^{-2t}\sigma_{\mathbf x}^2-v_t^2-b_t^2.
$$

这里 \(u,v,w\) 都是独立标准高斯。

这些量的物理意义分别是：

- \(a_t\)：和“数据方向相关”的线性 signal 分量；
- \(v_t\)：样本依赖但与整体协方差无关的随机涨落；
- \(b_t\)：与噪声 \(\boldsymbol\xi\) 相关的分量；
- \(s_t^2\)：剩余各向同性噪声地板。

### 9.2 用 Stein 引理理解 \(a_t,b_t\)

虽然论文正文没展开，但这里有个很漂亮的关系。

记
$$
Y=e^{-t}\sigma_{\mathbf x}u+\sqrt{\Delta_t}v.
$$
则 \(Y\sim \mathcal N(0,\Gamma_t^2)\)，因此可写成
$$
Y=\Gamma_t z,\qquad z\sim\mathcal N(0,1).
$$

定义
$$
\mu_1(t):=\mathbb E_z[\sigma(\Gamma_t z)z].
$$

由 Stein 引理
$$
\mathbb E[zf(z)]=\mathbb E[f'(z)]
$$
可得
$$
\mathbb E[\sigma'(Y)]
=
\frac{1}{\Gamma_t}\mu_1(t).
$$

于是
$$
\mathbb E[u\sigma(Y)]
=
e^{-t}\sigma_{\mathbf x}\,\mathbb E[\sigma'(Y)]
=
e^{-t}\sigma_{\mathbf x}\frac{\mu_1(t)}{\Gamma_t},
$$
所以
$$
a_t=\frac{\mu_1(t)}{\Gamma_t}.
$$

同理
$$
\mathbb E[v\sigma(Y)]
=
\sqrt{\Delta_t}\,\mathbb E[\sigma'(Y)]
=
\sqrt{\Delta_t}\frac{\mu_1(t)}{\Gamma_t},
$$
因此
$$
b_t=\sqrt{\Delta_t}\frac{\mu_1(t)}{\Gamma_t}.
$$

进而
$$
e^{-2t}\sigma_{\mathbf x}^2a_t^2+b_t^2
=
\frac{e^{-2t}\sigma_{\mathbf x}^2+\Delta_t}{\Gamma_t^2}\mu_1(t)^2
=
\mu_1(t)^2.
$$

这个关系在后面化简谱方程时非常有用。

### 9.3 Gaussian equivalence 结果

论文附录证明，在高维极限下，
$$
\boxed{
\mathbf U
\simeq
\frac{\mathbf G}{\sqrt n}\frac{\mathbf G^\top}{\sqrt n}
+
b_t^2\frac{\mathbf W\mathbf W^\top}{d}
+
s_t^2\mathbf I_p
}
$$
其中
$$
\mathbf G
=
e^{-t}a_t\frac{\mathbf W\mathbf X'}{\sqrt d}
+
v_t\mathbf\Omega,
$$
\(\mathbf X'\) 的列独立服从 \(\mathcal N(0,\mathbf\Sigma)\)，\(\mathbf\Omega\) 是独立高斯矩阵。

对应的总体版本是
$$
\boxed{
\tilde{\mathbf U}
\simeq
\mu_1(t)^2\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
+
\big(\|\sigma\|^2-\mu_1(t)^2\big)\mathbf I_p
}
$$
其中
$$
\mathbf\Sigma_t=e^{-2t}\mathbf\Sigma+\Delta_t\mathbf I_d.
$$

同时
$$
\boxed{
\mathbf V\simeq\tilde{\mathbf V}\simeq
\frac{\mu_1(t)\sqrt{\Delta_t}}{\Gamma_t}\frac{\mathbf W}{\sqrt d}
}
$$

### 9.4 这三个式子最重要的意义

它们把一个看上去很复杂的非线性学习问题，化成了：

- 一个样本协方差型随机矩阵 \(\mathbf G\mathbf G^\top/n\)；
- 一个总体协方差项 \(b_t^2\mathbf W\mathbf W^\top/d\)；
- 一个各向同性噪声项 \(s_t^2\mathbf I_p\)。

也就是说，问题从“分析神经网络训练”变成了“分析某个随机矩阵的谱”。

### 9.5 为什么 Gaussian equivalence 看起来合理

这一步第一次读往往会觉得很“魔法”。其实它背后的想法很朴素：

> 当维度很大时，非线性特征 \(\sigma(\mathbf W\mathbf x/\sqrt d)\) 的二阶统计性质，往往只依赖于少数低阶相关量；因此可以用一个高斯代理模型来复现这些二阶统计。

更具体地说，令
$$
g_i=\frac{\mathbf W_i^\top \mathbf x}{\sqrt d},
$$
其中 \(\mathbf W_i\) 是 \(\mathbf W\) 的第 \(i\) 行。高维下，\(g_i\) 近似高斯。

如果 \(g_i,g_j\) 的相关系数很小，那么对
$$
\mathbb E[\sigma(g_i)\sigma(g_j)]
$$
做 Hermite 展开或 Mehler 展开，主导项通常只取决于：

1. 对角上的方差；
2. 非对角上的线性相关。

因此 \(\sigma(g)\) 可以被近似分解成：

- 一个沿着原始信号方向的线性部分；
- 一个与样本有关但各向同性的高斯噪声部分；
- 再加上一个对角平移项。

这就是为什么 \(\mathbf U\) 的 GEP 结果会长成
$$
\frac{\mathbf G\mathbf G^\top}{n}+b_t^2\frac{\mathbf W\mathbf W^\top}{d}+s_t^2I
$$
这种“样本协方差 + 总体协方差 + 各向同性噪声”的结构。

如果你只记一个核心直觉，那就是：

> GEP 不是说“原模型真的等于高斯模型”，而是说“对于我们关心的谱和二阶统计，高维极限下它们可以用同一个高斯代理来替代”。

---

## 10. 从 GEP 到谱方程：三元 saddle-point 方程组

作者定义三个 Stieltjes 型量：
$$
q(z)=\frac{1}{p}\operatorname{Tr}(\mathbf U-z\mathbf I_p)^{-1},
$$
$$
r(z)=\frac{1}{p}\operatorname{Tr}\!\left(
\mathbf\Sigma^{1/2}\mathbf W^\top(\mathbf U-z\mathbf I_p)^{-1}\mathbf W\mathbf\Sigma^{1/2}
\right),
$$
$$
s(z)=\frac{1}{p}\operatorname{Tr}\!\left(
\mathbf W^\top(\mathbf U-z\mathbf I_p)^{-1}\mathbf W
\right).
$$

再定义
$$
\hat s(q)=b_t^2\psi_p+\frac{1}{q},
$$
$$
\hat r(r,q)
=
\frac{\psi_p a_t^2e^{-2t}}
{
1+\dfrac{a_t^2e^{-2t}\psi_p}{\psi_n}r
+
\dfrac{\psi_p v_t^2}{\psi_n}q
}.
$$

则论文给出如下方程组：
$$
s
=
\int \mathrm d\rho_{\mathbf\Sigma}(\lambda)
\frac{1}{\hat s(q)+\lambda \hat r(r,q)},
$$
$$
r
=
\int \mathrm d\rho_{\mathbf\Sigma}(\lambda)
\frac{\lambda}{\hat s(q)+\lambda \hat r(r,q)},
$$
$$
\psi_p(s_t^2-z)
+
\frac{\psi_pv_t^2}
{
1+\dfrac{a_t^2e^{-2t}\psi_p}{\psi_n}r
+
\dfrac{\psi_p v_t^2}{\psi_n}q
}
+
\frac{1-\psi_p}{q}
-\frac{s}{q^2}
=0.
$$

最后由 Sokhotski--Plemelj 公式恢复谱密度：
$$
\rho(\lambda)
=
\lim_{\varepsilon\downarrow0}
\frac{1}{\pi}\operatorname{Im}q(\lambda+i\varepsilon).
$$

### 10.1 先解释 \(q,r,s\) 这三个量各自到底在测什么

第一次看到
$$
q(z),\ r(z),\ s(z)
$$
很多人会疑惑：为什么不是只写一个 \(q(z)\)，还要多写 \(r,s\)？

原因是：

- \(q(z)\) 是最标准的 resolvent trace，只看整体谱；
- \(r(z)\) 和 \(s(z)\) 是“带有方向权重的 resolvent trace”，它们不只看特征值，还看 resolvent 和 \(\mathbf W,\mathbf\Sigma\) 这些特殊方向的耦合。

更具体地说：

$$
q(z)=\frac1p\operatorname{Tr}(U-zI)^{-1}
$$
只统计所有特征值；

$$
s(z)=\frac1p\operatorname{Tr}\big(W^\top(U-zI)^{-1}W\big)
$$
测的是 resolvent 在 \(W\) 张成的方向上有多大；

$$
r(z)=\frac1p\operatorname{Tr}\big(\Sigma^{1/2}W^\top(U-zI)^{-1}W\Sigma^{1/2}\big)
$$
则进一步把数据协方差 \(\Sigma\) 的方向信息也编码了进去。

所以从数学上说，这三者共同构成一个闭合的自洽系统；从物理上说，它们分别编码了：

- 纯谱信息；
- 随机特征方向信息；
- 数据协方差方向信息。

### 10.2 这个方程怎么来的

论文附录的套路是：

1. 写
$$
q(z)=\frac{1}{p}\operatorname{Tr}(\mathbf U-z\mathbf I)^{-1}
=
2\partial_z
\frac{1}{p}\log\det(\mathbf U-z\mathbf I)^{-1/2}.
$$

2. 用高斯积分把 \(\det^{-1/2}\) 变成配分函数；
3. 用 replica trick 处理 \(\log\det\)；
4. 对 \(\mathbf X,\mathbf W,\mathbf\Omega\) 做平均；
5. 引入 order parameters \(Q,R,S\)；
6. 做 replica-symmetric saddle point；
7. 对 \(q,r,s,\hat r,\hat s\) 求驻点，得到上面的方程。

这部分论文附录给了完整公式链，但它毕竟是 replica 方法，所以我这里不把它包装成纯数学严格证明。真正重要的是：

- 方程组的结构非常清楚；
- 后面两个时间尺度都是直接从这个方程组的不同标度解中读出来的。

### 10.3 如果不想碰 replica，最少该记住什么

如果你暂时不想深入 replica trick，那最少要记住下面三点：

1. 对称随机矩阵的谱问题，通常会先转成 resolvent / Stieltjes 变换问题；
2. 在高维极限里，这些变换往往满足自洽方程；
3. 只要能得到自洽方程，谱边界、delta 峰、多个 bulk 的权重和位置，都可以从这些方程里读出来。

换句话说，论文真正依赖 replica 的地方，不是最终结论的“物理直觉”，而是具体自洽方程的推导捷径。后面谱分解与时间尺度结论本身，是完全可以独立理解的。

### 10.4 把 \(q(z)\) 当成“平均后的 \(\frac{1}{\lambda-z}\)”来看

这一点非常值得单独说，因为它是后面所有谱结论的直觉核心。

设
$$
\mathbf U = Q\operatorname{diag}(\lambda_1,\dots,\lambda_p)Q^\top.
$$
那么
$$
(\mathbf U-zI)^{-1}
=
Q\operatorname{diag}\!\left(
\frac{1}{\lambda_1-z},\dots,\frac{1}{\lambda_p-z}
\right)Q^\top.
$$

于是
$$
q(z)
=
\frac1p\operatorname{Tr}(\mathbf U-zI)^{-1}
=
\frac1p\sum_{i=1}^p \frac{1}{\lambda_i-z}.
$$

所以你完全可以把 \(q(z)\) 理解成：

> 把每个特征值 \(\lambda_i\) 先代入函数 \(\frac{1}{\lambda-z}\)，再做平均。

这句话虽然朴素，但几乎已经包含了全部谱分析直觉。

因为：

- 如果很多特征值都落在某个区间附近，那么 \(q(z)\) 在那个区间附近会有连续响应；
- 如果在某一点 \(\lambda_0\) 上堆了宏观数量级的特征值，那么 \(q(z)\) 里会出现
$$
\frac{\text{权重}}{\lambda_0-z}
$$
这样的极点；
- 如果 \(z=\lambda+i\varepsilon\) 穿过某个 bulk，那么
$$
\frac{1}{\lambda_i-(\lambda+i\varepsilon)}
$$
会出现虚部，于是谱密度非零。

因此后面你看到作者“求 \(q(z)\) 的自洽方程”，本质上就是：

> 不直接盯着 \(p\) 个特征值，而是用一个复函数 \(q(z)\) 把整个谱一次性打包起来。

![随机特征模型中的训练/测试误差曲线](../../assets/papers/why-diffusion-models-dont-memorize/rf-train-test.png)

*随机特征实验图：即使在这个简化模型里，也能清楚看到 train/test loss 与 score error 的双时间尺度结构，这正是后面谱分析要解释的对象。*

---

## 11. 两个谱团块：\(\rho_1\) 控制记忆，\(\rho_2\) 控制泛化

这是全文最关键的一步。

### 11.1 定理的形式

在强过参数化 / 大数据极限下，作者得到谱分解：

#### Regime I：\(\psi_p>\psi_n\gg1\)
$$
\rho(\lambda)
=
\left(1-\frac{1+\psi_n}{\psi_p}\right)\delta(\lambda-s_t^2)
+
\frac{\psi_n}{\psi_p}\rho_1(\lambda)
+
\frac{1}{\psi_p}\rho_2(\lambda).
$$

#### Regime II：\(\psi_n>\psi_p\gg1\)
$$
\rho(\lambda)
=
\left(1-\frac{1}{\psi_p}\right)\rho_1(\lambda)
+
\frac{1}{\psi_p}\rho_2(\lambda).
$$

其中：

- \(\rho_1\) 是第一个 bulk；
- \(\rho_2\) 是第二个 bulk；
- 还有一个在 \(s_t^2\) 处的 delta 峰（只在 \(\psi_p>\psi_n+1\) 时出现）。

### 11.2 先看 delta 峰

如果某个向量 \(\mathbf v\in\mathbb R^p\) 同时满足
$$
\mathbf\Omega^\nu\cdot \mathbf v=0,\qquad \nu=1,\dots,n,
$$
以及
$$
\mathbf W^\top\mathbf v=0,
$$
那么在 GEP 近似下
$$
\mathbf U\mathbf v=s_t^2\mathbf v.
$$

这意味着 \(\lambda=s_t^2\) 是特征值。

约束一共有 \(n+d\) 个，而向量维度是 \(p\)，所以只要
$$
p\ge n+d,
$$
就会有核空间，其维数近似是
$$
p-n-d.
$$

故对应的谱质量约为
$$
\frac{p-n-d}{p}
=
1-\frac{n}{p}-\frac{d}{p}
=
1-\frac{\psi_n}{\psi_p}-\frac{1}{\psi_p}.
$$

这就是上面 delta 峰的权重。

而且这部分特征向量满足 \(\mathbf W^\top\mathbf v=0\)，对训练和测试损失都没有贡献，所以它们和泛化 / 记忆动力学无关。

### 11.3 这里其实就是线性代数里的 rank-nullity 定理

上面那段关于 delta 峰的讨论，完全可以用最基础的线性代数语言来理解。

定义线性映射
$$
T:\mathbb R^p\to \mathbb R^{n+d},
\qquad
T(\mathbf v)
=
\begin{bmatrix}
\mathbf\Omega^1\!\cdot\!\mathbf v\\
\vdots\\
\mathbf\Omega^n\!\cdot\!\mathbf v\\
\mathbf W^\top \mathbf v
\end{bmatrix}.
$$

那么“同时满足”
$$
\mathbf\Omega^\nu\cdot\mathbf v=0,\qquad
\mathbf W^\top \mathbf v=0
$$
其实就是在说
$$
T(\mathbf v)=0,
$$
也就是 \(\mathbf v\in \ker(T)\)。

线性代数里最基本的 rank-nullity 定理说：
$$
\dim\ker(T)+\operatorname{rank}(T)=p.
$$

而 \(\operatorname{rank}(T)\le n+d\)，因此
$$
\dim\ker(T)\ge p-(n+d).
$$

这就说明：只要 \(p\) 比 \(n+d\) 大很多，就一定存在大量非零向量 \(\mathbf v\) 同时落在这两个约束的核空间里。

对这些向量，经验项和总体协方差项都“看不见”它们，于是只剩各向同性项 \(s_t^2I\) 在起作用：
$$
\mathbf U\mathbf v=s_t^2\mathbf v.
$$

因此 \(\lambda=s_t^2\) 处会积累起一个宏观比例的特征值，这就是 delta 峰。

---

## 12. 第一团块 \(\rho_1\)：慢模态，控制记忆时间

### 12.1 做标度假设

在过参数化 regime 下，作者对 saddle-point 方程做 ansatz：
$$
q=\mathcal O(1),\qquad
r=\mathcal O\!\left(\frac{1}{\psi_p}\right),\qquad
s=\mathcal O\!\left(\frac{1}{\psi_p}\right).
$$

于是
$$
\hat s=b_t^2\psi_p+\frac{1}{q}\sim \mathcal O(\psi_p),
$$
$$
\hat r
=
\frac{\psi_p a_t^2e^{-2t}}
{1+\frac{\psi_p v_t^2}{\psi_n}q+\mathcal O(1)}
\sim \mathcal O(\psi_p).
$$

而 \(s,r\) 都因分母是 \(\mathcal O(\psi_p)\) 而是 \(\mathcal O(1/\psi_p)\)。

将这些数量级代回最后一个方程，保留主导项，得到
$$
(s_t^2-z)
+
\frac{v_t^2}{1+\dfrac{\psi_p}{\psi_n}v_t^2 q}
-\frac{1}{q}
=0.
$$

### 12.2 化成二次方程

记
$$
c:=\frac{\psi_p}{\psi_n}.
$$

则
$$
(s_t^2-z)+\frac{v_t^2}{1+cv_t^2q}-\frac{1}{q}=0.
$$

乘以 \(q(1+cv_t^2q)\)：
$$
\big((s_t^2-z)q-1\big)(1+cv_t^2q)+v_t^2q=0.
$$

展开得
$$
cv_t^2(s_t^2-z)q^2
+
\big((s_t^2-z)+v_t^2-cv_t^2\big)q
-1=0.
$$

这是关于 \(q\) 的二次方程。

如果你想更熟悉一点，也可以把它写成
$$
Aq^2+Bq+C=0
$$
的标准形式，其中
$$
A=cv_t^2(s_t^2-z),\qquad
B=(s_t^2-z)+v_t^2-cv_t^2,\qquad
C=-1.
$$

于是根就是
$$
q(z)=\frac{-B\pm\sqrt{B^2-4AC}}{2A}.
$$

这里最重要的不是把根真的完整背下来，而是要看到：

- 只要平方根里面是正数，\(q(z)\) 就是实的；
- 一旦平方根里面变成负数，\(q(z)\) 就会出现虚部；
- 而谱密度正是从 \(q\) 的虚部读出来的。

### 12.3 谱边界 = 判别式为零

设 \(z=\lambda+i0\)。当二次方程判别式为负时，\(q\) 出现虚部，于是谱密度非零。

判别式为
$$
\Delta(\lambda)
=
\big((s_t^2-\lambda)+v_t^2(1-c)\big)^2
+
4cv_t^2(s_t^2-\lambda).
$$

令 \(x=\lambda-s_t^2\)，则
$$
\Delta(\lambda)
=
\big(-x+v_t^2(1-c)\big)^2-4cv_t^2x.
$$

解 \(\Delta(\lambda)=0\) 可得边界
$$
\boxed{
\lambda_\pm
=
s_t^2+v_t^2\left(1\pm\sqrt{\frac{\psi_p}{\psi_n}}\right)^2
}
$$

所以第一团块 \(\rho_1\) 的支撑区间是
$$
\boxed{
\operatorname{supp}(\rho_1)
=
\left[
s_t^2+v_t^2\left(1-\sqrt{\frac{\psi_p}{\psi_n}}\right)^2,\;
s_t^2+v_t^2\left(1+\sqrt{\frac{\psi_p}{\psi_n}}\right)^2
\right]
}
$$

如果你以前学过一元二次函数，这里其实就是同一件事：

- 判别式 \(>0\)：两个实根，对应 resolvent 没有虚部；
- 判别式 \(=0\)：碰到边界；
- 判别式 \(<0\)：根变成共轭复数对，对应谱密度出现。

### 12.3.1 把判别式一步一步化到论文里的边界公式

这一段论文原文写得比较快，我们把代数一步一步写开。

从
$$
Aq^2+Bq+C=0
$$
出发，判别式是
$$
\Delta = B^2-4AC.
$$

代入
$$
A=cv_t^2(s_t^2-\lambda),\qquad
B=(s_t^2-\lambda)+v_t^2(1-c),\qquad
C=-1
$$
得到
$$
\Delta(\lambda)
=
\big((s_t^2-\lambda)+v_t^2(1-c)\big)^2
+4cv_t^2(s_t^2-\lambda).
$$

现在令
$$
x=\lambda-s_t^2,
\qquad
s_t^2-\lambda=-x,
$$
则
$$
\Delta(\lambda)
=
\big(-x+v_t^2(1-c)\big)^2-4cv_t^2x.
$$

把平方展开：
$$
\Delta(\lambda)
=
x^2-2xv_t^2(1-c)+v_t^4(1-c)^2-4cv_t^2x.
$$

把 \(x\) 的系数合并：
$$
-2v_t^2(1-c)-4cv_t^2
=
-2v_t^2(1+c),
$$
因此
$$
\Delta(\lambda)
=
x^2-2v_t^2(1+c)x+v_t^4(1-c)^2.
$$

这又是一个关于 \(x\) 的二次式。令 \(\Delta(\lambda)=0\)，得到
$$
x
=
v_t^2(1+c)\pm v_t^2\sqrt{(1+c)^2-(1-c)^2}.
$$

而
$$
(1+c)^2-(1-c)^2 = 4c,
$$
所以
$$
x
=
v_t^2(1+c)\pm 2v_t^2\sqrt c
=
v_t^2(1\pm \sqrt c)^2.
$$

最后把 \(x=\lambda-s_t^2\) 代回去，就得到
$$
\lambda_\pm
=
s_t^2 + v_t^2(1\pm \sqrt c)^2
=
s_t^2 + v_t^2\left(1\pm \sqrt{\frac{\psi_p}{\psi_n}}\right)^2.
$$

这就是论文里的边界公式。

### 12.4 为什么它对应记忆时间

当 \(p\gg n\) 时，
$$
\frac{\psi_p}{\psi_n}\gg1,
$$
所以 \(\rho_1\) 的特征值尺度是
$$
\lambda_{\mathrm{mem}}\sim \frac{\psi_p}{\psi_n}.
$$

于是对应时间尺度
$$
\tau_{\mathrm{mem}}
\asymp
\frac{\psi_p}{\Delta_t\,\lambda_{\mathrm{mem}}}
\sim
\frac{\psi_p}{\Delta_t}\cdot\frac{\psi_n}{\psi_p}
=
\frac{\psi_n}{\Delta_t}.
$$

由于 \(\psi_n=n/d\)，在固定 \(d\) 下就是
$$
\boxed{
\tau_{\mathrm{mem}}\propto n
}
$$

这就是论文最核心的理论结论。

---

## 13. 第二团块 \(\rho_2\)：快模态，控制泛化时间

### 13.1 第二种标度 ansatz

为了抓到第二个 bulk，作者改用
$$
q=\mathcal O\!\left(\frac{1}{\psi_p}\right),\qquad
r=\mathcal O\!\left(\frac{1}{\psi_p}\right).
$$

于是
$$
\hat s=b_t^2\psi_p+\frac{1}{q},
\qquad
\hat r\simeq \psi_p a_t^2e^{-2t}.
$$

把它代回方程后，可化成
$$
q
=
-\left(
z'
-\int
\frac{\rho_{\mu_1^2(t)\mathbf\Sigma_t}(\mu)\,\mu\,\mathrm d\mu}
{1+\psi_p q\mu}
\right)^{-1},
$$
其中
$$
z'=z-s_t^2-v_t^2.
$$

这个方程正是 Bai--Silverstein 型样本协方差谱方程，因此它对应的 bulk 就是总体矩阵
$$
\tilde{\mathbf U}
=
\mu_1^2(t)\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
+
(s_t^2+v_t^2)\mathbf I_p
$$
的样本协方差谱。

### 13.2 为什么它和 \(n\) 无关

注意这里出现的是总体协方差 \(\mathbf\Sigma_t\)，而不是有限样本数据集本身。

因此第二个 bulk \(\rho_2\)：

- 反映的是总体结构；
- 和具体这次训练集抽到了哪些样本无关；
- 因而天然对应“population / generalization”层面的模态。

### 13.3 它的特征值尺度

由于
$$
\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
$$
是一个 \(p\times p\) 的 Wishart 型对象，且 \(p>d\)，其 bulk 特征值规模是
$$
\mathcal O(\psi_p).
$$

### 13.3.1 什么叫 Wishart 型矩阵

如果
$$
X\in\mathbb R^{p\times d}
$$
的每一列都是均值为零的随机向量，那么
$$
XX^\top
$$
就是最典型的样本协方差型矩阵。

当列向量是高斯时，这类矩阵的分布就叫 Wishart 分布；即使不是严格高斯，只要满足合适条件，它的谱行为通常也仍然接近 Wishart / Marchenko--Pastur 类。

因此
$$
\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
$$
之所以被称作 Wishart 型，是因为你可以把
$$
\mathbf Y:=\mathbf W\mathbf\Sigma_t^{1/2}
$$
看成一个“带协方差结构的随机设计矩阵”，于是
$$
\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
=
\frac{\mathbf Y\mathbf Y^\top}{d}
$$
就是样本协方差矩阵的标准形式。

从直觉上说：

- \(\mathbf\Sigma_t\) 决定总体上哪些方向方差大、哪些方向方差小；
- \(\mathbf W\) 把这些方向随机投影到 \(p\) 维特征空间；
- 最后得到的 Gram / 协方差矩阵的谱，就描述了“总体结构在随机特征空间里长什么样”。

### 13.3.2 如果 \(\mathbf\Sigma_t=I\)，它就退化成最熟悉的 Marchenko--Pastur 情形

为了和你熟悉的线性代数对象接上，我们看最简单的各向同性情形：
$$
\mathbf\Sigma_t = I_d.
$$

此时
$$
\frac{\mathbf W\mathbf\Sigma_t\mathbf W^\top}{d}
=
\frac{\mathbf W\mathbf W^\top}{d}.
$$

如果 \(\mathbf W\) 的元素是 iid 高斯，那么这就是教科书级别的 Wishart 矩阵。

在 \(p,d\to\infty\) 且 \(p/d\to \psi_p\) 时，它的非零特征值会集中在一个 Marchenko--Pastur 区间里，尺度大约是
$$
\big(1\pm \sqrt{\psi_p}\big)^2.
$$

所以当 \(\psi_p\) 很大时，典型特征值大小确实就是
$$
\mathcal O(\psi_p).
$$

这正是作者说
$$
\lambda_{\mathrm{gen}}\sim \psi_p
$$
的最朴素来源。

也就是说，\(\rho_2\) 并不神秘，它本质上就是：

> 一个带总体协方差 \(\Sigma_t\) 的 Wishart 型矩阵所产生的“大尺度 bulk”。

于是第二个时间尺度是
$$
\tau_{\mathrm{gen}}
\asymp
\frac{\psi_p}{\Delta_t\,\lambda_{\mathrm{gen}}}
\sim
\frac{\psi_p}{\Delta_t\,\psi_p}
=
\frac{1}{\Delta_t}.
$$

因此
$$
\boxed{
\tau_{\mathrm{gen}}=\mathcal O(1)
}
$$

更重要的是，它 **不依赖 \(n\)**。

![随机特征模型中的谱分解示意](../../assets/papers/why-diffusion-models-dont-memorize/rf-spectrum.png)

*理论图：\(\mathbf U\) 的谱清楚分成靠近零处的慢 bulk 与更大尺度上的快 bulk。论文正是用这两个 bulk 来解释“先泛化、后记忆”。*

---

## 14. 这两个时间尺度为什么对应“先泛化，后记忆”

现在我们可以把整篇论文最重要的逻辑链写成一行：

$$
\rho_2:\ \lambda=\mathcal O(\psi_p)
\Longrightarrow
\tau_{\mathrm{gen}}=\mathcal O(1),
$$
$$
\rho_1:\ \lambda=\mathcal O(\psi_p/\psi_n)
\Longrightarrow
\tau_{\mathrm{mem}}=\mathcal O(\psi_n)\propto n.
$$

解释非常直接：

1. **大特征值模态先学会**  
   这些模态来自 \(\rho_2\)，描述的是总体 / 低频 / 平滑部分，所以模型很快获得不错的生成能力。

2. **小特征值模态后学会**  
   这些模态来自 \(\rho_1\)，它们携带更多样本依赖、高频、数据集特定的信息，所以只有很长时间以后才会被拟合。

3. **记忆需要高频样本依赖模态**  
   所以记忆不会一开始发生，而是在很久以后才发生。

4. **数据越多，慢模态越慢**  
   因此记忆时间正比于 \(n\)。

这就是所谓的 **implicit dynamical regularization**：

> 不是模型根本不具备记忆能力，而是训练动力学先把模型带到了一个只学到总体结构的区域；如果适时停止，就能利用这个时间顺序本身获得泛化。

---

## 15. 论文附录里的一个重要补充：快时间上训练 / 测试损失几乎一致

论文还证明了，在
$$
1\ll \tau\ll \psi_n
$$
的时间窗中，
$$
\mathcal L_{\mathrm{train}}
\simeq
\mathcal L_{\mathrm{test}}
\simeq
1-\mathcal O(\Delta_t).
$$

直觉上，这是因为：

- 在这个时间段，\(\rho_2\) 那批大特征值对应的模态已经几乎松弛完；
- \(\rho_1\) 那批小特征值对应的模态还没怎么动；
- 而 \(\rho_2\) 本身对应的是总体协方差主导的方向，因此 train/test 基本一致。

作者用
$$
\|\mathbf U-\tilde{\mathbf U}\|_{\mathrm{op}}
=
\mathcal O\!\left(\frac{\psi_p}{\sqrt{\psi_n}}\right)
$$
来控制这件事，说明在第二个 bulk 对应的子空间里，经验矩阵 \(\mathbf U\) 和总体矩阵 \(\tilde{\mathbf U}\) 相差很小，因此训练和测试损失差距在大 \(n\) 下消失。

这等于从另一个角度支持了“先学总体，再学样本细节”的图景。

---

## 16. 论文实验和理论是怎样对应上的

现在把真实 U-Net 实验和 RF 理论一一对上。

### 16.1 真实实验

在真实 U-Net / CelebA 上，作者观察到：
$$
\tau_{\mathrm{gen}}\approx \text{常数},\qquad
\tau_{\mathrm{mem}}\propto n.
$$

同时模型宽度 \(W\) 增大时，
$$
\tau_{\mathrm{gen}}\propto W^{-1},\qquad
\tau_{\mathrm{mem}}\propto nW^{-1}.
$$

### 16.2 理论模型

在 RF 模型里，作者证明：
$$
\tau_{\mathrm{gen}}\sim \frac{1}{\Delta_t},
\qquad
\tau_{\mathrm{mem}}\sim \frac{\psi_n}{\Delta_t}.
$$

若把模型容量增大理解为增大 \(\psi_p\) 或提高有效谱尺度，那么快模态和慢模态的松弛都整体加快，于是与实验中的 \(W^{-1}\) 趋势一致。

### 16.3 统一图景

实验和理论共同支持的图景是：

$$
\text{快模态（population-like）}
\;\to\;
\text{高质量生成}
\;\to\;
\text{慢模态（dataset-specific）}
\;\to\;
\text{记忆}.
$$

---

## 17. 我对这篇论文的理解：它真正解释了什么

我觉得这篇论文最有价值的地方，不是简单地说“早停有用”，而是把为什么有用拆成了谱上的两类模态：

### 17.1 它解释了“为什么记忆不是立刻发生”

以前很多经验现象只告诉我们：

- 小数据时会记忆；
- 大数据时不太容易记忆；
- 早停能改善。

但这篇论文进一步说：

- 经验 score 的高频部分更难学；
- 这些高频部分携带训练集特异性；
- 高维训练动力学天然先学光滑部分，后学尖锐部分。

这就把“先泛化、后记忆”解释成了一个训练动力学上的 **谱偏置**。

### 17.2 它把“数据集变大”解释成“慢模态更慢”

不是因为样本被看得更少，而是因为：

- 随 \(n\) 增大，经验 score 里样本特定的那部分更细、更高频；
- 这些对应更小的有效特征值；
- 所以需要更长训练时间才能被拟合。

这比“数据多所以不容易过拟合”要具体得多。

### 17.3 它把 architectural regularization 和 dynamical regularization 区分开了

两种机制不同：

1. **Architectural regularization**  
   模型本身不够表达经验 score，哪怕 \(\tau\to\infty\) 也记不住。

2. **Dynamical regularization**  
   模型其实足以记住，但在有限训练时间内先学到的是泛化解。

这两个概念在实践里经常被混在一起，这篇论文把它们明确分开，是很有价值的。

---

## 18. 这篇论文的局限

### 18.1 理论模型是固定 \(t\) 的，不是完整 time-conditioned DDPM

真实 DDPM 用一个共享网络同时覆盖全部扩散时间 \(t\)，而理论部分是每个 \(t\) 单独分析。这当然抓住了核心机制，但仍然是简化模型。

### 18.2 随机特征网络和真实 U-Net 之间仍有距离

随机特征模型没有卷积归纳偏置、层级特征、attention 等结构。它能解释“为什么有两个时间尺度”，但不能完整刻画工业级扩散模型。

### 18.3 Replica 推导不是严格概率论证明

这点必须坦诚。附录中的 saddle-point 推导是统计物理传统路线，经验上常常正确，但它不是一篇纯数学论文。

### 18.4 论文给出了缩放律，但没有完全闭式地给出所有常数

例如 \(\tau_{\mathrm{gen}}\)、\(\tau_{\mathrm{mem}}\) 的前因子如何依赖数据分布、激活函数、优化器、条件输入等，论文只做了部分分析。

---

## 19. 一页纸总结

这篇论文的整个逻辑链，可以压缩成下面几步：

### 第一步：经验 score 一定会记忆

经验训练集对应的 noisy density 是
$$
P_t^{\mathrm{emp}}(\mathbf x)
=
\frac{1}{n(2\pi\Delta_t)^{d/2}}
\sum_{\nu=1}^n
\exp\!\left(
-\frac{\|\mathbf x-e^{-t}\mathbf x^\nu\|^2}{2\Delta_t}
\right),
$$
其极小化器
$$
\mathbf s_{\mathrm{emp}}=\nabla\log P_t^{\mathrm{emp}}
$$
在小噪声下会把反向生成轨道吸向训练样本。

### 第二步：实际训练的时间尺度由 \(\mathbf U\) 的谱决定

随机特征模型的梯度流闭式解为
$$
\frac{\mathbf A(\tau)}{\sqrt p}
=
-\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}
+
\left(
\frac{1}{\sqrt{\Delta_t}}\mathbf V^\top\mathbf U^{-1}
+
\frac{\mathbf A_0}{\sqrt p}
\right)
\exp\!\left(-\frac{2\Delta_t}{\psi_p}\mathbf U\tau\right),
$$
所以模态时间尺度是
$$
\tau_\lambda\asymp \frac{\psi_p}{\Delta_t\lambda}.
$$

### 第三步：\(\mathbf U\) 有两个彼此分离的谱团块

在过参数化区，
$$
\rho(\lambda)
=
\left(1-\frac{1+\psi_n}{\psi_p}\right)\delta(\lambda-s_t^2)
+
\frac{\psi_n}{\psi_p}\rho_1(\lambda)
+
\frac{1}{\psi_p}\rho_2(\lambda).
$$

其中
$$
\operatorname{supp}(\rho_1)
=
\left[
s_t^2+v_t^2\left(1-\sqrt{\frac{\psi_p}{\psi_n}}\right)^2,\;
s_t^2+v_t^2\left(1+\sqrt{\frac{\psi_p}{\psi_n}}\right)^2
\right],
$$
而 \(\rho_2\) 的特征值规模是 \(\mathcal O(\psi_p)\)。

### 第四步：于是自动出现两个训练时间尺度

因为
$$
\lambda_{\rho_2}\sim \psi_p
\quad\Longrightarrow\quad
\tau_{\mathrm{gen}}\sim \mathcal O(1),
$$
而
$$
\lambda_{\rho_1}\sim \frac{\psi_p}{\psi_n}
\quad\Longrightarrow\quad
\tau_{\mathrm{mem}}\sim \mathcal O(\psi_n)\propto n.
$$

所以：
$$
\boxed{
\tau_{\mathrm{gen}}\approx \text{常数},
\qquad
\tau_{\mathrm{mem}}\propto n
}
$$

这就解释了为什么扩散模型在会记忆之前，先有一个越来越宽的泛化窗口。

---

## 20. 最后的理解

如果只用一句话概括这篇论文，我会这样说：

> 扩散模型不是“不会记忆”，而是“记忆解对应的谱模态更慢，因此训练动力学会先到达泛化解，再在更久以后才逐渐接近记忆解”。

我认为这是这篇论文最重要的贡献：它把“早停有效”从经验事实，提升成了一个有明确随机矩阵 / 谱分解支撑的理论图景。

---

## 21. 参考论文与链接

- 主论文 arXiv：<https://arxiv.org/abs/2505.17638>
- 相关背景：Jonathan Ho, Ajay Jain, Pieter Abbeel, *Denoising Diffusion Probabilistic Models*：<https://arxiv.org/abs/2006.11239>
- 相关背景：Yang Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*：<https://arxiv.org/abs/2011.13456>
