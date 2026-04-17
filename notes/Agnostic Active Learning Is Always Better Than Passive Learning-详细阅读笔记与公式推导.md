---
tags: [public]
category: 研究笔记
---

# Agnostic Active Learning Is Always Better Than Passive Learning：详细阅读笔记与公式推导

## 0. 论文信息

- 题目：Agnostic Active Learning Is Always Better Than Passive Learning
- 作者：Steve Hanneke
- 会议：NeurIPS 2025 Oral
- OpenReview：<https://openreview.net/forum?id=XPe55Uffd7>
- PDF：<https://openreview.net/pdf?id=XPe55Uffd7>

一句话概括：这篇论文解决了 agnostic active learning 里一个持续二十年的核心开放问题。它证明了，对任意概念类 \( \mathcal C \)，agnostic active learning 的最优一阶 query complexity 的领先项总是

$$
\Theta\!\left(\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)\right),
$$

而 passive learning 的对应领先项总是

$$
\Theta\!\left(\frac{\beta}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)\right),
$$

因此当 \( \varepsilon \ll \beta \ll 1 \) 时，active learning 总能比 passive learning 少一个数量级为 \( \beta \) 的因子。

---

## 1. 我对全文主线的理解

这篇论文最关键的贡献，不是简单提出一个新的 query rule，而是把 agnostic active learning 的困难重新表述成一个“方差隔离”问题。

作者的核心观察是：

1. passive learning 的一阶复杂度之所以是
   \[
   \frac{\beta}{\varepsilon^2}(d+\log(1/\delta)),
   \]
   本质上来自误差估计里的方差项，而这个方差大小正比于最优风险 \( \beta \)。
2. 传统 disagreement-based active learning 往往只能说“去 disagreement region 里问标签”，但这个 region 可能太大，甚至大到使得 query complexity 重新退化回 passive learning 的量级。
3. 真正该问的，不是整个 disagreement region，而是其中“误差估计方差真的高”的部分。
4. 只要能把这部分高方差区域 \(\Delta\) 隔离出来，并证明它的边缘质量满足
   \[
   P_X(\Delta)=O(\beta),
   \]
   那么在这个区域上做精细估计的 query 成本就会自动降到
   \[
   O\!\left(\frac{\beta^2}{\varepsilon^2}(d+\log(1/\delta))\right).
   \]
5. 其余区域由于 version space 的直径已经足够小，只需要支付 realizable-like 的 lower-order term。

这就是论文提出的算法原则：

$$
\text{AVID} = \text{Adaptive Variance Isolation by Disagreements}.
$$

---

## 2. 问题设定与基本定义

### 2.1 基本分类设定

设实例空间为 \( \mathcal X \)，概念类为

$$
\mathcal C \subseteq \{0,1\}^{\mathcal X},
$$

其 VC 维记为

$$
d = \operatorname{VC}(\mathcal C).
$$

未知数据分布为

$$
P \ \text{on}\ \mathcal X\times\{0,1\}.
$$

对任意分类器 \(h:\mathcal X\to\{0,1\}\)，定义其风险

$$
\operatorname{er}_P(h)
:=
P\big(\{(x,y):h(x)\neq y\}\big).
$$

最佳类内风险记为

$$
\beta_P^*
:=
\inf_{h\in\mathcal C}\operatorname{er}_P(h).
$$

论文中用参数 \( \beta \) 来描述一个噪声预算，表示我们讨论的 minimax 复杂度是对所有满足

$$
\beta_P^* \le \beta
$$

的分布 \(P\) 取最坏情形。

### 2.2 active learning 与 passive learning 的复杂度

#### passive learning

给定 \(n\) 个 i.i.d. 标注样本

$$
(X_1,Y_1),\dots,(X_n,Y_n)\sim P,
$$

被动学习器输出 \( \hat h \)。

最优 passive sample complexity 记为

$$
M_P(\varepsilon,\delta;\beta,\mathcal C),
$$

它是最小的 \(n\)，使得存在某个 passive learner，对于所有满足 \( \beta_P^*\le\beta \) 的分布 \(P\)，都有

$$
\Pr\!\left(
\operatorname{er}_P(\hat h)\le \beta_P^*+\varepsilon
\right)
\ge 1-\delta.
$$

#### active learning

active learner 看到一串 i.i.d. 未标注样本 \(X_1,X_2,\dots\)，可以自适应选择要查询哪些 \(Y_i\)，最终输出 \(\hat h\)。

最优 active query complexity 记为

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta),
$$

它是最小的 query 数 \(Q\)，使得存在 active learner，对所有满足 \( \beta_P^*\le\beta \) 的 \(P\)，以至少 \(1-\delta\) 的概率：

$$
\operatorname{er}_P(\hat h)\le \beta_P^*+\varepsilon,
\qquad
\#\text{queries}\le Q.
$$

### 2.3 star number

论文的第二个主结果会引入 star number。定义为

$$
s(\mathcal C)
:=
\sup\Big\{
n\in\mathbb N:
\exists x_1,\dots,x_n\in\mathcal X,\ 
\exists h_0,h_1,\dots,h_n\in\mathcal C,
\ \forall i,j,\ 
h_i(x_j)\neq h_0(x_j)\iff i=j
\Big\}.
$$

直觉上，它衡量的是这样一种 realizable-active-learning 最难结构：

- 有一堆点 \(x_1,\dots,x_n\)
- 大多数假设都和默认假设 \(h_0\) 一样
- 只有某个未知的 \(h_i\) 会在唯一一个点 \(x_i\) 上翻转标签

这会迫使 active learner 基本逐个排查这些点。

---

## 3. passive learning 的复杂度为什么是 \( \beta/\varepsilon^2 + 1/\varepsilon \)

这一节是理解全文的基础。因为 active learning 的领先项，正是把这里的 \( \beta \) 再省掉了一个。

### 3.1 从 ERM 的 excess risk 出发

取 \(h^*\in\mathcal C\) 为一个近似最优分类器，使得

$$
\operatorname{er}_P(h^*)\le \beta_P^*+\frac{\varepsilon}{100}.
$$

为简化推导，下文先把 \(h^*\) 当成达到 infimum 的最优解来写，即

$$
\operatorname{er}_P(h^*)=\beta_P^*.
$$

对任意 \(h\in\mathcal C\)，记 excess risk 为

$$
\Delta(h)
:=
\operatorname{er}_P(h)-\operatorname{er}_P(h^*).
$$

再定义损失差随机变量

$$
Z_h(x,y)
:=
\mathbf 1[h(x)\neq y]-\mathbf 1[h^*(x)\neq y].
$$

则

$$
\mathbb E[Z_h] = \Delta(h).
$$

同时，由于 \(Z_h\in\{-1,0,1\}\)，并且只有在 \(h(x)\neq h^*(x)\) 时它才可能非零，所以

$$
Z_h^2(x,y)=\mathbf 1[h(x)\neq h^*(x)].
$$

因此

$$
\operatorname{Var}(Z_h)
\le
\mathbb E[Z_h^2]
=
P_X(h\neq h^*)
$$

而二分类里有经典不等式

$$
P_X(h\neq h^*)
\le
\operatorname{er}_P(h)+\operatorname{er}_P(h^*)
=
\Delta(h)+2\beta_P^*.
\tag{3.1}
$$

所以

$$
\operatorname{Var}(Z_h)\le \Delta(h)+2\beta_P^*.
\tag{3.2}
$$

这一步就是“\(\beta\) 出现在一阶复杂度里”的根源。

### 3.2 Uniform Bernstein 型控制

记经验风险最小化器为

$$
\hat h
\in
\arg\min_{h\in\mathcal C}\hat{\operatorname{er}}_n(h).
$$

对 VC 类，结合 symmetrization、localization 和 Bernstein 型集中不等式，可以得到：存在常数 \(c_1,c_2>0\)，使得以至少 \(1-\delta\) 的概率，对所有 \(h\in\mathcal C\) 同时成立

$$
\left|
\big(\hat{\operatorname{er}}_n(h)-\hat{\operatorname{er}}_n(h^*)\big)
-\Delta(h)
\right|
\le
c_1\sqrt{\frac{(\Delta(h)+\beta_P^*)(d+\log(1/\delta))}{n}}

+c_2\frac{d+\log(1/\delta)}{n}.
\tag{3.3}
$$

把 \(h=\hat h\) 代入，并利用 ERM 的定义

$$
\hat{\operatorname{er}}_n(\hat h)-\hat{\operatorname{er}}_n(h^*)\le 0,
$$

得到

$$
\Delta(\hat h)
\le
c_1\sqrt{\frac{(\Delta(\hat h)+\beta_P^*)(d+\log(1/\delta))}{n}}

+c_2\frac{d+\log(1/\delta)}{n}.
\tag{3.4}
$$

### 3.3 解这个 fixed-point 不等式

记

$$
\Gamma := d+\log(1/\delta).
$$

若我们想保证 \( \Delta(\hat h)\le \varepsilon \)，只需证明当 \( \Delta(\hat h)\ge \varepsilon \) 时，式 (3.4) 不可能成立。

假设 \( \Delta(\hat h)\ge \varepsilon \)。则

$$
c_1\sqrt{\frac{\Delta(\hat h)\Gamma}{n}}
\le
c_1\sqrt{\frac{\Gamma}{n\varepsilon}}\ \Delta(\hat h).
$$

所以只要

$$
n\ge 16c_1^2\frac{\Gamma}{\varepsilon},
\tag{3.5}
$$

就有

$$
c_1\sqrt{\frac{\Delta(\hat h)\Gamma}{n}}
\le
\frac14\Delta(\hat h).
\tag{3.6}
$$

另一方面，只要

$$
n\ge 16c_1^2\frac{\beta_P^*\Gamma}{\varepsilon^2},
\tag{3.7}
$$

就有

$$
c_1\sqrt{\frac{\beta_P^*\Gamma}{n}}
\le
\frac{\varepsilon}{4}
\le
\frac14\Delta(\hat h).
\tag{3.8}
$$

再只要

$$
n\ge 4c_2\frac{\Gamma}{\varepsilon},
\tag{3.9}
$$

就有

$$
c_2\frac{\Gamma}{n}
\le
\frac{\varepsilon}{4}
\le
\frac14\Delta(\hat h).
\tag{3.10}
$$

把 (3.6)(3.8)(3.10) 代回 (3.4)，得到右边至多为

$$
\frac34\Delta(\hat h),
$$

与 (3.4) 矛盾。

因此，只要

$$
n
\gtrsim
\frac{\beta_P^*}{\varepsilon^2}\Gamma
+
\frac{1}{\varepsilon}\Gamma,
$$

就有 \( \Delta(\hat h)\le\varepsilon \)。

也就是说，passive learning 的样本复杂度满足

$$
M_P(\varepsilon,\delta;\beta,\mathcal C)
=
\Theta\!\left(
\frac{\beta}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
+
\frac{d+\log(1/\delta)}{\varepsilon}
\right),
\tag{3.11}
$$

忽略对数细节时可写成

$$
\tilde\Theta\!\left(\frac{\beta d}{\varepsilon^2}+\frac{d}{\varepsilon}\right).
$$

### 3.4 这一步的真正含义

被动学习里，决定领先项的不是“平均误差本身”，而是“估计误差差值的方差”。  
式 (3.2) 告诉我们，这个方差的一阶量级是 \( \beta \)。所以 passive learning 必须付出 \( \beta/\varepsilon^2 \) 的代价。

active learning 如果想严格优于 passive，就必须做到下面这件事：

> 不是把所有地方都平均地采样，而是把方差真的大的地方单独拎出来，只在那里密集查询。

这正是 AVID 的出发点。

---

## 4. 论文的主定理

### 4.1 主结论一：对所有概念类，active 的领先项总是更优

论文的第一主结果可以写成：

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\tilde O\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
+
\frac{d}{\varepsilon}
\right),
\tag{4.1}
$$

并且存在匹配的下界

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\Omega\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right).
\tag{4.2}
$$

更强的是：对每个 \(d\)，都存在某个 VC 维为 \(d\) 的类，使得

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\Omega\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
+
\frac{d}{\varepsilon}
\right).
\tag{4.3}
$$

于是，一阶领先项已经被完全刻画。

### 4.2 主结论二：star number 精炼 lower-order term

令 \(s=s(\mathcal C)\)。论文的第二主结果是

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
O\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right)
+
\tilde O\!\left(
\big(s\wedge\tfrac1\varepsilon\big)d
\right),
\tag{4.4}
$$

并且

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\Omega\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
+
s\wedge\tfrac1\varepsilon
\right).
\tag{4.5}
$$

这说明：

1. 领先项对所有类都是最优的，统一是
   \[
   \Theta\!\left(\frac{\beta^2}{\varepsilon^2}(d+\log(1/\delta))\right).
   \]
2. realizable-like 的 lower-order term 则由 star number 控制。
3. 一般上界和下界之间还差一个 \(d\) 因子，但这已经是“接近最优”的统一刻画。

### 4.3 与 passive learning 的直接比较

passive 的领先项：

$$
\frac{\beta}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big).
$$

active 的领先项：

$$
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big).
$$

二者的比值是

$$
\frac{\beta^2/\varepsilon^2}{\beta/\varepsilon^2}
=
\beta.
\tag{4.6}
$$

因此在 far-from-realizable regime

$$
\varepsilon \ll \beta \ll 1
$$

里，active learning 必然比 passive learning 更省标签，而且这个结论对任意概念类都成立。

---

## 5. 为什么传统 disagreement-based 分析还不够

论文解决的问题，恰好卡在过去理论的短板上。

传统思路一般是：

1. 维护一个 version space \(V\)
2. 只在 disagreement region
   \[
   \operatorname{DIS}(V)
   :=
   \{x:\exists f,g\in V,\ f(x)\neq g(x)\}
   \]
   上查询标签

这在 realizable setting 非常自然，但在 agnostic setting 会出现一个致命问题：

- 有些点虽然在 \(\operatorname{DIS}(V)\) 里
- 但这些 disagreement 对“区分谁更接近 Bayes / 最优类内分类器”其实没有那么高的信息量
- 于是 learner 还是会在太大的区域里花掉太多查询

这就是先前上界里经常出现 disagreement coefficient、splitting index 等附加因子的原因。

论文的突破点是：

> 不要把整个 disagreement region 当成“都一样难”。  
> 应该继续把它切开，只把真正高方差、真正难估的那一块拿出来单独处理。

---

## 6. AVID 算法到底在做什么

### 6.1 两个对象：version space 与高方差区域

AVID 在每个精度层级 \(k\) 维护：

- 一个候选集合 \(V_{k-1}\subseteq\mathcal C\)
- 一个已经“剥离”出来的高方差区域 \(\Delta_{i_k}\subseteq\mathcal X\)

并令当前目标精度为

$$
\varepsilon_k \asymp C^{\,1-k},
\qquad 1<C<2.
$$

算法要同时维持两类不变量：

#### 不变量 A：在 \(\mathcal X\setminus\Delta_{i_k}\) 上，version space 直径已经很小

对所有 \(f,g\in V_{k-1}\)，都有

$$
P_X\big(\{f\neq g\}\setminus \Delta_{i_k}\big)\le \varepsilon_k.
\tag{6.1}
$$

也就是说，在“外面”这块区域里，候选假设们已经几乎不分歧了。

#### 不变量 B：被剥离出来的 \(\Delta_{i_k}\) 本身质量很小

AVID 最关键的新结论是

$$
P_X(\Delta_{i_k})\le 5\beta_P^*.
\tag{6.2}
$$

这是全文最重要的一步。因为它直接把 hardest region 的质量压到了 \(O(\beta)\)。

### 6.2 两种样本预算

在每个精度层级 \(k\)，算法做两种事：

#### 1. 在外部容易区域上做 disagreement-based 查询

取一个未标注样本块 \(S_k^{(1)}\)，大小大致为

$$
m_k
\asymp
\frac{d\log(1/\varepsilon_k)+\log(1/\delta_k)}{\varepsilon_k}.
\tag{6.3}
$$

只对其中落在

$$
D_{k-1}\setminus\Delta_{i_k},
\qquad D_{k-1}:=\operatorname{DIS}(V_{k-1}),
$$

的点查询标签。

#### 2. 在内部困难区域上做高精度误差估计

对 \(\Delta_{i_k}\) 再单独拿一块未标注样本 \(S_k^{(2)}\)，大小设计为

$$
m_k'
\asymp
\frac{p_k}{\varepsilon_k^2}\Big(d+\log\tfrac1{\delta_k}\Big),
\qquad
p_k \approx P_X(\Delta_{i_k}).
\tag{6.4}
$$

然后只对 \(S_k^{(2)}\) 中真正落在 \(\Delta_{i_k}\) 的样本查询标签。

这里最关键的一点是：

- \(m_k'\) 是用来做“整体误差贡献”估计的未标注样本数
- 真正需要问标签的，只是其中进入 \(\Delta_{i_k}\) 的那部分

所以标签数会再乘一次 \(p_k\)，从而得到 \(p_k^2/\varepsilon_k^2\) 这一结构，也就是领先项里出现 \(\beta^2\) 而不是 \(\beta\) 的根本原因。

---

## 7. 最关键的证明：为什么 \(P_X(\Delta)\le O(\beta)\)

这是全篇我觉得最值得反复看的一段。

### 7.1 记号

设算法在构造 \(\Delta\) 的过程中形成一串递增集合

$$
\varnothing=\Delta_0\subseteq \Delta_1\subseteq\cdots,
$$

每次新增一块

$$
A_i := \Delta_{i+1}\setminus \Delta_i.
$$

论文中每个 \(A_i\) 都来自某个候选对 \(f,g\) 的 disagreement：

$$
A_i = \{f\neq g\}\setminus \Delta_i.
\tag{7.1}
$$

目标是证明：对任意 \(h_0\in\mathcal C\)，每次新增的区域都会“切掉” \(h_0\) 的一块错误区，即

$$
P\big(\operatorname{ER}(h_0)\cap A_i\big)
\ge
c\,P_X(A_i)
\tag{7.2}
$$

其中 \(c>0\) 是常数。把它对所有 \(A_i\) 累加起来，就能推出 \(P_X(\Delta)\lesssim \beta_P^*\)。

### 7.2 第一步：在新增区域里，\(f\) 或 \(g\) 至少有一个会犯很多错

由于标签只有 \(0/1\)，在 \(A_i=\{f\neq g\}\setminus\Delta_i\) 上，\(f\) 和 \(g\) 中必有一个出错。更准确地，

$$
\operatorname{ER}(f)\cap A_i
\quad\text{与}\quad
\operatorname{ER}(g)\cap A_i
$$

是互斥且并起来覆盖 \(A_i\) 的。

因此至少有一个 \(f'\in\{f,g\}\) 满足

$$
P\big(\operatorname{ER}(f')\cap A_i\big)
\ge
\frac12 P_X(A_i).
\tag{7.3}
$$

这一步完全是几何/组合事实。

### 7.3 第二步：构造一个 hybrid classifier

取任意 \(h_0\in\mathcal C\)。论文构造一个“拼接”分类器 \(h'\)，它在新加入的区域 \(A_i\) 上模仿 \(h_0\)，在其他地方模仿 \(f'\)。

抽象地写就是

$$
h'
=
f'\mathbf 1_{\mathcal X\setminus A_i}
+
h_0\mathbf 1_{A_i},
\tag{7.4}
$$

更严格地说，论文把 \(\Delta_{i_k}\) 与当前辅助候选集一起并入定义里，以保证 \(h'\) 确实落在分析需要的那个扩展候选集合里。

这个构造的意义是：

- \(f'\) 和 \(h'\) 只在 \(A_i\) 上不同
- 所以二者真风险之差恰好就是“谁在 \(A_i\) 上犯错更多”

于是有

$$
\operatorname{er}_P(f')-\operatorname{er}_P(h')
=
P\big(\operatorname{ER}(f')\cap A_i\big)
-
P\big(\operatorname{ER}(h_0)\cap A_i\big).
\tag{7.5}
$$

### 7.4 第三步：由经验最优性推出真风险差不能太大

AVID 在当前层级会选一个经验上表现最好的 \(\hat h_k\)，并把 version space 缩成那些经验风险不比 \(\hat h_k\) 差太多的函数。

于是，论文中的统一偏差控制引理给出：对当前分析里出现的可比较函数，真风险差不会比经验风险差大太多。把它应用到 \(f'\) 和 \(h'\) 上，可得

$$
\operatorname{er}_P(f')-\operatorname{er}_P(h')
\le
O(\varepsilon_k).
\tag{7.6}
$$

论文证明里的具体形式是

$$
\operatorname{er}_P(f')-\operatorname{er}_P(h')
\le
\frac{5\varepsilon_k}{4C'}.
\tag{7.7}
$$

其中 \(C'>1\) 是算法常数。

### 7.5 第四步：把三者合起来

将 (7.3)(7.5)(7.7) 合并：

$$
P\big(\operatorname{ER}(h_0)\cap A_i\big)
\ge
\frac12 P_X(A_i)-\frac{5\varepsilon_k}{4C'}.
\tag{7.8}
$$

而算法只有在当前 disagreement 片段“仍然足够大”时才会继续把它加入 \(\Delta\)。因此 \(P_X(A_i)\) 至少是 \( \Theta(\varepsilon_k) \) 量级，\(\frac{5\varepsilon_k}{4C'}\) 可以被吸收进前项。

论文最终得到常数形式

$$
P\big(\operatorname{ER}(h_0)\cap A_i\big)
\ge
\frac15 P_X(A_i).
\tag{7.9}
$$

### 7.6 第五步：求和

由于各个 \(A_i\) 两两不交，

$$
P\big(\operatorname{ER}(h_0)\cap \Delta_i\big)
=
\sum_{j<i} P\big(\operatorname{ER}(h_0)\cap A_j\big)
\ge
\frac15 \sum_{j<i} P_X(A_j)
=
\frac15 P_X(\Delta_i).
\tag{7.10}
$$

于是

$$
P_X(\Delta_i)
\le
5\,P\big(\operatorname{ER}(h_0)\cap \Delta_i\big).
\tag{7.11}
$$

对任意 \(h_0\in\mathcal C\) 成立，故取 infimum：

$$
P_X(\Delta_i)
\le
5\inf_{h\in\mathcal C}P\big(\operatorname{ER}(h)\cap \Delta_i\big)
\le
5\inf_{h\in\mathcal C}\operatorname{er}_P(h)
=
5\beta_P^*.
\tag{7.12}
$$

这就是全文最核心的不等式。

### 7.7 我对这一步的直觉理解

这一步非常漂亮。它不是在说“\(\Delta\) 小，所以好学”；而是在说：

> 任何一次把某块区域加入 \(\Delta\)，都意味着这块区域里，任意参考分类器 \(h_0\) 都必须已经背负了与其质量成正比的一部分错误。

于是 \(\Delta\) 不可能无限长大，否则会逼得最优分类器也在 \(\Delta\) 上犯掉超过 \(\beta\) 的错误，与 \(\beta_P^*=\inf_h \operatorname{er}_P(h)\) 矛盾。

---

## 8. 为什么 leading term 会变成 \( \beta^2/\varepsilon^2 \)

这节是全文第二个最关键的地方。

### 8.1 在困难区内估计误差贡献

设

$$
p_k := P_X(\Delta_{i_k}).
$$

由上一节知道

$$
p_k \le 5\beta_P^* \le 5\beta.
\tag{8.1}
$$

为了把 \(\Delta_{i_k}\) 内的误差贡献估到精度 \(O(\varepsilon_k)\)，算法取未标注样本数

$$
m_k'
\asymp
\frac{p_k}{\varepsilon_k^2}
\Big(d+\log\tfrac1{\delta_k}\Big).
\tag{8.2}
$$

这正对应于估计

$$
P\big(\operatorname{ER}(h)\cap\Delta_{i_k}\big)
$$

的 Bernstein 方差大小为 \(O(p_k)\)。

### 8.2 但真正要问标签的只有进入 \(\Delta_{i_k}\) 的样本

在这 \(m_k'\) 个未标注样本里，只有落入 \(\Delta_{i_k}\) 的点才需要被查询标签，所以 query 数大约是

$$
q_k^{\text{hard}}
\asymp
p_k m_k'
\asymp
\frac{p_k^2}{\varepsilon_k^2}
\Big(d+\log\tfrac1{\delta_k}\Big).
\tag{8.3}
$$

再代入 \(p_k\le 5\beta\)，得到

$$
q_k^{\text{hard}}
\lesssim
\frac{\beta^2}{\varepsilon_k^2}
\Big(d+\log\tfrac1{\delta_k}\Big).
\tag{8.4}
$$

这就是为什么主动学习的领先项里会出现 \( \beta^2 \)。

### 8.3 对所有层级求和

由于 \(\varepsilon_k\) 按几何序列递减，且最后一层满足 \(\varepsilon_N\asymp \varepsilon\)，于是

$$
\sum_{k=1}^{N}\frac1{\varepsilon_k^2}
=
\Theta\!\left(\frac1{\varepsilon^2}\right).
\tag{8.5}
$$

因此困难区域贡献总 query complexity 为

$$
\sum_k q_k^{\text{hard}}
\lesssim
\frac{\beta^2}{\varepsilon^2}
\Big(d+\log\tfrac1\delta\Big).
\tag{8.6}
$$

这正是主定理的 leading term。

---

## 9. 为什么 easy region 只留下 lower-order term

### 9.1 版本空间在 easy region 外的直径已经很小

由 AVID 的不变量 (6.1)，在 \(\mathcal X\setminus\Delta_{i_k}\) 上，\(V_{k-1}\) 的直径已经缩到 \(\varepsilon_k\)。

这意味着，如果我们把条件分布写成

$$
P_k(A)
:=
\frac{P_X(A\setminus \Delta_{i_k})}{P_X(\mathcal X\setminus\Delta_{i_k})},
\tag{9.1}
$$

那么 \(V_{k-1}\) 实际上落在某个以 \(h^*\) 为中心、半径为 \(O(\varepsilon_k)\) 的球里。

### 9.2 用 disagreement coefficient / star number 控制外部 disagreement

论文利用了一个已知事实：对任意概念类，

$$
\theta \le s,
\tag{9.2}
$$

即 disagreement coefficient 被 star number 上界。

于是可以推出

$$
P_X(D_{k-1}\setminus\Delta_{i_k})
\le
s\,\varepsilon_k.
\tag{9.3}
$$

当 \(s=\infty\) 时这当然要截断到 \(1\)，所以更准确地写是

$$
P_X(D_{k-1}\setminus\Delta_{i_k})
\le
\min\{1,s\varepsilon_k\}.
\tag{9.4}
$$

### 9.3 因而外部 query 数每层只需 realizable-like 成本

外部样本块大小是

$$
m_k
\asymp
\frac{d\log(1/\varepsilon_k)+\log(1/\delta_k)}{\varepsilon_k}.
\tag{9.5}
$$

只查询其中落入 \(D_{k-1}\setminus\Delta_{i_k}\) 的点，因此每层外部 query 数满足

$$
q_k^{\text{easy}}
\lesssim
\min\{1,s\varepsilon_k\}\,m_k
\lesssim
\min\left\{\frac1{\varepsilon_k},s\right\}
\Big(d\log\tfrac1{\varepsilon_k}+\log\tfrac1{\delta_k}\Big).
\tag{9.6}
$$

求和后得到

$$
\sum_k q_k^{\text{easy}}
\lesssim
\min\left\{
\frac1\varepsilon,\,
s\log\tfrac1\varepsilon
\right\}
\Big(d\log\tfrac1\varepsilon+\log\tfrac1\delta\Big).
\tag{9.7}
$$

也就是 theorem 5 里的 lower-order term：

$$
\tilde O\!\left(\big(s\wedge\tfrac1\varepsilon\big)d\right).
\tag{9.8}
$$

### 9.4 这一步的直觉

active learning 在 easy region 上并没有做什么“神奇”的事情。它只是利用：

- 版本空间已经很窄
- 所以 disagreement 已经很稀薄
- 在这时继续问标签，本质上就是 realizable active learning 的老问题

因此 lower-order term 由 star number 接管，是很自然的。

---

## 10. 全部复杂度拼起来

把困难区和容易区两部分相加：

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
\lesssim
\sum_k q_k^{\text{hard}}
+
\sum_k q_k^{\text{easy}}.
\tag{10.1}
$$

由 (8.6) 与 (9.7) 得

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
O\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right)
+
O\!\left(
\min\left\{
\frac1\varepsilon,\,
s\log\tfrac1\varepsilon
\right\}
\Big(d\log\tfrac1\varepsilon+\log\tfrac1\delta\Big)
\right).
\tag{10.2}
$$

把 polylog 合并进 \( \tilde O \) 记号里，就是

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
O\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right)
+
\tilde O\!\left(
\big(s\wedge\tfrac1\varepsilon\big)d
\right),
\tag{10.3}
$$

与论文主定理一致。

---

## 11. 下界告诉我们的事情

### 11.1 领先项已经完全 sharp

论文结合已有下界证明：

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\Omega\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right).
\tag{11.1}
$$

因此领先项没有任何附加复杂度量可以再改进掉了。

### 11.2 lower-order term 在一般类上还没完全闭合

论文给出的一般 lower bound 是

$$
\Omega\!\left(s\wedge\tfrac1\varepsilon\right),
\tag{11.2}
$$

而一般 upper bound 是

$$
\tilde O\!\left(\big(s\wedge\tfrac1\varepsilon\big)d\right).
\tag{11.3}
$$

因此中间还差一个 \(d\)。

作者的观点是：

- 这已经是“几乎最优”的统一刻画
- 若想把 lower-order term 也完全 sharp 地刻画出来，可能需要比 \(d\) 和 \(s\) 更细的新复杂度量

### 11.3 一些 regime 的含义

#### 1. far-from-realizable：\( \varepsilon \ll \beta \ll 1 \)

领先项支配，总复杂度为

$$
\Theta\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right),
$$

严格优于 passive。

#### 2. nearly realizable：\( \beta\lesssim \varepsilon \)

此时 leading term 变弱，lower-order term 开始主导，active 的收益可能明显缩小。

#### 3. very high noise：\( \beta=\Theta(1) \)

此时 active 的领先项和 passive 的领先项只差常数级，active 的收益也会消失。

---

## 12. 论文没有实验，所有“结果”都是理论结果

这篇论文是纯理论工作，没有数值实验、benchmark 或 ablation。

因此它的“实验结果”应理解为三类理论结果：

1. 上界：AVID 达到统一的 query complexity 上界。
2. 下界：已有 lower bound 在领先项上对所有类都是 sharp。
3. 精炼：lower-order term 可以用 star number 精炼，并在很多类上接近最优。

换句话说，这篇论文的“结果部分”不是图表，而是 theorem + lemma + lower bound 三件套。

---

## 13. 我觉得最值得记住的 5 个公式

### 1. passive learning 的一阶复杂度

$$
M_P(\varepsilon,\delta;\beta,\mathcal C)
=
\Theta\!\left(
\frac{\beta}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
+
\frac{d+\log(1/\delta)}{\varepsilon}
\right).
$$

### 2. active learning 的统一领先项

$$
Q_{\mathcal C}^a(\varepsilon,\delta;\beta)
=
\Theta\!\left(
\frac{\beta^2}{\varepsilon^2}\Big(d+\log\tfrac1\delta\Big)
\right)
\quad
\text{(领先项意义下)}.
$$

### 3. AVID 最关键的不等式

$$
P_X(\Delta)\le 5\beta_P^*.
$$

### 4. 困难区每层 query 代价

$$
q_k^{\text{hard}}
\asymp
P_X(\Delta_k)\,m_k'
\asymp
\frac{P_X(\Delta_k)^2}{\varepsilon_k^2}
\Big(d+\log\tfrac1{\delta_k}\Big).
$$

### 5. 容易区每层 query 代价

$$
q_k^{\text{easy}}
\lesssim
\min\left\{\frac1{\varepsilon_k},s\right\}
\Big(d\log\tfrac1{\varepsilon_k}+\log\tfrac1{\delta_k}\Big).
$$

---

## 14. 我自己的总结

我觉得这篇论文真正漂亮的地方，是它把“active learning 为什么有可能永远优于 passive”这件事说得非常结构化：

1. passive 的瓶颈来自方差，而不是只来自偏差。
2. agnostic 场景下，方差高的部分不一定等于整个 disagreement region。
3. 只要把高方差区域 \(\Delta\) 从 disagreement 里继续剥出来，并证明 \(P_X(\Delta)=O(\beta)\)，就能把 query complexity 的领先项压到 \( \beta^2/\varepsilon^2 \)。
4. 剩下的部分则退化成 realizable active learning 的 lower-order 问题，由 star number 控制。

所以这篇论文的贡献可以压缩成一句话：

> agnostic active learning 的本质，不是“更聪明地在 disagreement 里问”，而是“先把 disagreement 里真正高方差的那部分单独隔离出来”。

这也是为什么作者把算法原则命名为 AVID，而不是把它描述成又一个 disagreement-based 变体。

---

## 15. 进一步阅读时我会特别关注的点

如果后面要继续深挖，我觉得最值得继续跟的 4 个方向是：

1. 能不能把 theorem 3 的 lower-order term 里的 \(d\) 因子去掉。
2. 能不能给出 matching upper/lower bound 所需的新复杂度量，而不只用 \(d\) 和 \(s\)。
3. 能不能把 AVID 的“方差隔离”原则迁移到更一般的 selective labeling / bandit feedback / partial monitoring 问题里。
4. 能不能把这种思路转译到现代深度学习的 pool-based active learning heuristic 上，得到更 principled 的 acquisition design。
