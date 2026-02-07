# 数学原理与控制原理详解

本项目旨在通过数据驱动的预测控制（Data-driven Predictive Control）解决自由锻造液压机的控制问题。项目包含了基于物理模型的模型预测控制（MPC）、监督学习（Supervised Learning）神经网络控制器，以及无监督（混合）学习（Unsupervised/Hybrid Learning）控制器。

## 1. 物理模型与数学原理

该项目使用一个非线性连续时间系统来模拟液压机。模型包含五个状态变量和一个控制输入。

### 1.1 状态变量与控制输入

状态变量 $x \in \mathbb{R}^5$ 定义如下：
- $y$: 上模具的位移（变形量） [m]
- $\dot{y}$: 上模具的速度（变形速度） [m/s]
- $p_1$: 工作缸内的压力 [Pa]
- $p_2$: 回程缸内的压力 [Pa]
- $z$: 伺服阀阀芯的位移 [m]

控制输入 $u \in \mathbb{R}^1$：
- $u$: 施加在伺服阀上的电压/张力 [ad] (无量纲或归一化单位)

### 1.2 动力学方程

系统的动力学由以下微分方程组描述：

$$
\begin{aligned}
\dot{y} &= v \\
\dot{v} &= \frac{1}{M} \left( \frac{3}{4}\pi D_1^2 p_1 - \frac{1}{2}\pi D_2^2 p_2 - B v - F_t - F_{d\_article} \right) + G \\
\dot{p}_1 &= \frac{K_B}{V_1} \left( \frac{q_{vPB}}{3} - A_1 v - K_{L\_1} p_1 \right) \\
\dot{p}_2 &= \frac{K_B}{V_2} \left( -\frac{q_{vAT}}{2} + A_2 v - K_{L\_2} p_2 \right) \\
\dot{z} &= -\frac{z}{T_1} + \frac{u}{T_1}
\end{aligned}
$$

其中：
- $v = \dot{y}$ 是速度。
- $M$ 是运动部件的质量。
- $B$ 是粘滞阻尼系数。
- $D_1, D_2$ 分别是工作柱塞和回程柱塞的直径。
- $A_1, A_2$ 分别是工作柱塞和回程柱塞的有效面积。
- $G$ 是重力加速度。
- $K_B$ 是体积模量。
- $V_1, V_2$ 是腔体体积，随位移 $y$ 变化：$V_1 = V_{1\_0}/2 + A_1 y$, $V_2 = V_{2\_0}/2 - A_2 y$。
- $K_{L\_1}, K_{L\_2}$ 是泄漏系数。
- $T_1$ 是伺服阀的时间常数。

### 1.3 力学分析

**摩擦力 $F_t$**:
$$
F_t = \begin{cases}
\frac{F_T \cdot v}{0.5} & \text{if } |v| \le 0.5 \\
F_T & \text{otherwise}
\end{cases}
$$

**变形力 $F_{d\_article}$**:
变形力基于材料特性和几何形状计算。使用 C45 40 碳钢的材料常数。
$$
F_{d\_article} = K_d \cdot A_d \cdot M_0 \cdot \exp(M_1 T) \cdot \epsilon^{M_2} \cdot \dot{\epsilon}^{M_3} \cdot \exp(M_4/\epsilon)
$$
当 $y > 0$ 且 $\dot{y} \ge 0$ 时生效，否则为 0。

其中：
- $\epsilon = \ln(H_0 / (H_0 - y))$ 是应变。
- $\dot{\epsilon} = \dot{y} / (H_0 - y)$ 是应变率。
- $K_d$ 是考虑摩擦和热传递的影响系数。
- $A_d$ 是锻件与模具的接触面积。

### 1.4 伺服阀流量方程

流量 $q_{vPB}$ 和 $q_{vAT}$ 取决于阀芯位移 $z$ 和压力差：

$$
\begin{aligned}
q_{vPB\_work} &= \pi D z C_D \sqrt{\frac{2}{\rho} |P_S - p_1|} \cdot \text{sign}(P_S - p_1) \\
q_{vAT\_work} &= \pi D z C_D \sqrt{\frac{2}{\rho} |p_2 - P_T|} \cdot \text{sign}(p_2 - P_T) \\
q_{vPB\_return} &= \pi D z C_D \sqrt{\frac{2}{\rho} |p_1 - P_T|} \cdot \text{sign}(p_1 - P_T) \\
q_{vAT\_return} &= \pi D z C_D \sqrt{\frac{2}{\rho} |P_S - p_2|} \cdot \text{sign}(P_S - p_2)
\end{aligned}
$$

当 $z \ge 0$ 时，使用 `_work` 流量；否则使用 `_return` 流量。

---

## 2. 控制原理：模型预测控制 (MPC)

MPC 用于生成最优控制序列，使系统状态跟踪参考轨迹。

### 2.1 优化问题

在每个时间步 $k$，MPC 求解以下优化问题：

$$
\min_{u_{0:N-1}} \sum_{k=0}^{N-1} l(x_k, u_k) + m(x_N)
$$

受限于：
- 系统动力学方程 $x_{k+1} = f(x_k, u_k)$
- 状态约束 $x_{\min} \le x_k \le x_{\max}$ (如压力 $p_1, p_2 \ge 0$)
- 控制输入约束 $u_{\min} \le u_k \le u_{\max}$

### 2.2 代价函数

代价函数设计为跟踪参考速度 `ref`：

$$
l(x_k, u_k) = (\dot{y}_k - \text{ref}_k)^2 + \lambda u_k^2
$$

其中：
- $(\dot{y}_k - \text{ref}_k)^2$ 惩罚速度跟踪误差。
- $\lambda u_k^2$ 是正则化项，惩罚控制输入的大小（代码中 `mpc.set_rterm(u = 0.02)`）。

---

## 3. 监督学习 (Supervised Learning)

目标是训练一个神经网络控制器 $\pi_{\theta}(x)$ 来模仿 MPC 的行为。

### 3.1 数据生成
通过运行 MPC 控制器在不同条件下（随机初始状态、噪声等）的闭环仿真，收集数据集 $\mathcal{D} = \{(x_k, \text{ref}_k, u_k^*)\}$，其中 $u_k^*$ 是 MPC 计算的最优控制输入。

### 3.2 网络结构
使用一个前馈神经网络 (FNN)：
- **输入**: $[\dot{y}, z, \text{ref}]$
- **隐藏层**: 1层，50个神经元，ReLU 激活函数。
- **输出**: $u$ (控制输入)。

### 3.3 训练目标
最小化预测控制输入与 MPC 最优控制输入之间的 L1 损失：
$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(x, \text{ref}, u^*) \in \mathcal{D}} | \pi_{\theta}(x, \text{ref}) - u^* |
$$

---

## 4. 无监督/混合学习 (Unsupervised/Hybrid Learning)

为了提高控制器的鲁棒性和处理约束的能力，采用了无监督/混合学习框架。

### 4.1 代理模型 (Surrogate Model)
首先训练一个 LSTM 神经网络作为植物（Plant）的代理模型，用于预测系统状态的演变。
$$
x_{k+1} \approx f_{\phi}(x_k, u_k)
$$
输入特征包括 $[ \dot{y}, p_1, p_2, z, u ]$。

### 4.2 控制器训练与 MPC 损失
训练控制器 $\pi_{\theta}$ 时，不再仅仅模仿 MPC 的输出，而是最小化基于代理模型预测的未来轨迹的损失，类似于 MPC 的目标函数。

损失函数包含：
- **跟踪误差**: $\sum_{k=1}^N (\dot{\hat{y}}_k - \text{ref}_k)^2$
- **约束惩罚**: 惩罚违反压力约束等情况。
- **物理一致性**: 确保预测状态符合物理规律。

### 4.3 可行性恢复 (Feasibility Recovery)
为了确保神经网络输出的控制量不仅能跟踪参考轨迹，还能满足硬约束（如压力非负），引入了可行性恢复机制。这是一个基于优化的后处理步骤：

寻找 $u$，使得它尽可能接近神经网络的输出 $u_{NN}$，同时满足系统约束：
$$
\min_{u, s} (u - u_{NN})^2 + w \cdot ||s||^2
$$
s.t.
$$
x_{next} = f(x_{current}, u) \\
g(x_{next}) \le s
$$
其中 $s$ 是松弛变量，$g(x)$ 是约束条件。

---

## 总结

本项目结合了传统的基于模型的控制 (MPC) 和现代的数据驱动方法 (Deep Learning)。
1.  **物理建模**提供了系统的基准和数据生成源。
2.  **MPC** 提供了最优控制的基准。
3.  **监督学习** 试图克隆 MPC 的策略，实现快速推理。
4.  **无监督学习** 进一步利用代理模型优化控制器，使其更接近 MPC 的长期优化性能，并处理约束。
