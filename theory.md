# 1D 不连续 Galerkin (DG) / 通量重构 (FR) 方法理论文档

## 1. 引言

本代码实现了一个一维不连续 Galerkin (DG) 和通量重构 (FR) 方法的求解器，支持以下 PDEs：

- **线性平流方程 (LAD)**: $\frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$
- **Burgers 方程**: $\frac{\partial u}{\partial t} + \frac{1}{2}\frac{\partial u^2}{\partial x} = 0$
- **一维 Navier-Stokes 方程**

---

## 2. 不连续 Galerkin (DG) 方法

### 2.1 弱形式

在参考单元 $\xi \in [-1, 1]$ 上，DG 方法的弱形式通过对通量项进行分部积分得到：

$$\int_{-1}^{1} \ell_i \frac{\partial u_h}{\partial t} d\xi - \int_{-1}^{1} F(u_h) \frac{\partial \ell_i}{\partial \xi} d\xi = \left[\ell_i F^*\right]_{j-1/2}^{j+1/2}$$

其中：
- $\ell_i$ 是 Lagrange 基函数
- $F^*$ 是数值通量（黎曼通量）
- $u_h$ 是数值解

**推导过程**：
从强形式 $\int \ell_i (\frac{\partial u}{\partial t} + \frac{\partial F}{\partial x}) d\xi = 0$ 出发，对 $\frac{\partial F}{\partial x}$ 分部积分得：
$$\int \ell_i \frac{\partial F}{\partial x} d\xi = \left[\ell_i F\right] - \int F \frac{\partial \ell_i}{\partial x} d\xi$$

### 2.2 质量矩阵

$$M_{ij} = \int_{-1}^{1} \ell_i \ell_j d\xi$$

对于一阶线性单元 (ORDER=1)，LGL 点为 $\xi = [-1, 1]$：

$$\xi_0 = -1, \quad \xi_1 = 1$$

质量矩阵：

$$M = \begin{bmatrix} \frac{2}{3} & \frac{1}{3} \\ \frac{1}{3} & \frac{2}{3} \end{bmatrix}$$

LGL 权重：$w_0 = 1, \quad w_1 = 1$

### 2.3 微分矩阵 (D 矩阵)

$$D_{ij} = \left.\frac{\partial \ell_j}{\partial \xi}\right|_{\xi_i}$$

对于 ORDER=1：

$$D = \begin{bmatrix} -\frac{1}{2} & \frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{bmatrix}$$

### 2.4 刚度矩阵 (S 矩阵)

$$S_{ij} = \int_{-1}^{1} \ell_i \frac{\partial \ell_j}{\partial \xi} d\xi = M_{ik} D_{kj}$$

对于 ORDER=1：

$$S = \begin{bmatrix} -\frac{1}{2} & \frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{bmatrix}$$

### 2.5 DG 右端项计算

1. **在每个 GLL 点计算数值通量**：
   $$F_i = F(u_h(\xi_i))$$

2. **计算内部贡献**：
   $$\text{rhs\_prediction} = S \times F$$

3. **添加边界通量（黎曼问题）**：
   - 在 $j-1/2$ 处：$F^*_{j-1/2} = F(u_j^R, u_{j-1}^L)$
   - 在 $j+1/2$ 处：$F^*_{j+1/2} = F(u_j^R, u_{j+1}^L)$

4. **求解 $M^{-1}$ 得到最终右端项**

### 2.6 扩散项处理 (BR2 方法)

对于线性平流-扩散方程，使用 BR2 (Bassi-Rebay 2) 方法处理扩散项：

1. **计算单元内梯度**：
   $$\frac{\partial u}{\partial x} = M^{-1} D u$$

2. **计算局部提升算子**：
   $$\text{localLift} = M^{-1} M_{\text{ENT}} \times \text{jump} \times (-\frac{1}{2})$$

3. **全局提升算子累加**

4. **BR2 通量**：
   $$F_{\text{vis}} = \frac{1}{2} \nu \left((\frac{\partial u_L}{\partial x} - \frac{\partial u_L}{\partial x}^{\text{lift}}) + (\frac{\partial u_R}{\partial x} - \frac{\partial u_R}{\partial x}^{\text{lift}})\right)$$

---

## 3. 通量重构 (FR) 方法

### 3.1 基本思想

FR 方法通过"提升"将逐点通量重构为连续通量场，避免了 DG 方法中求解逆矩阵的操作。

### 3.2 FR 步骤

#### 1) 预测步 (Prediction)：计算内部通量贡献

- **LP (Lax-Friedrichs 通量)**：
  $$\text{rhs} = -D \times F / J$$

- **熵修正 (Entropy modification)**：使用 Ismail-Roe 或 Chandrashekar 熵稳定通量

#### 2) 校正步 (Correction)：添加界面通量贡献

$$\delta F_{j-1/2} = F^*_{j-1/2} - F(u_j(\xi=1)) \quad \text{(左边界)}$$

$$\delta F_{j+1/2} = F^*_{j+1/2} - F(u_j(\xi=-1)) \quad \text{(右边界)}$$

$$\text{rhs\_correction} = M^{-1} \times \delta F$$

#### 3) 组合

$$\text{rhs} = \text{rhs\_prediction} + \text{rhs\_correction}$$

### 3.3 熵稳定通量

代码实现了两种熵稳定通量：
- **Ismail-Roe 熵通量**
- **Chandrashekar 熵通量**

---

## 4. 黎曼求解器

### 4.1 Lax-Friedrichs 通量

$$F^* = \frac{1}{2} (F_L + F_R - \alpha (u_R - u_L))$$

其中：
- $\alpha = \max(|a|)$ (线性平流)
- $\alpha = \max(|u|)$ (Burgers)

### 4.2 HLL 通量 (Navier-Stokes)

计算左右波速：

$$\lambda_L = \min(u_L - c_L, u_R - c_R)$$

$$\lambda_R = \max(u_L + c_L, u_R + c_R)$$

其中 $c$ 为声速。

然后根据 $\lambda_L$ 和 $\lambda_R$ 的符号选择适当的通量。

---

## 5. 时间离散

### 5.1 RK1 (Forward Euler)

$$u^{n+1} = u^n + \Delta t \times \text{RHS}(u^n)$$

### 5.2 RK2 (Heun's Method)

$$u^* = u^n + \Delta t \times \text{RHS}(u^n)$$

$$u^{n+1} = \frac{1}{2} u^n + \frac{1}{2} u^* + \frac{1}{2} \Delta t \times \text{RHS}(u^*)$$

### 5.3 RK3 (SSPRK3)

$$u^* = u^n + \Delta t \times \text{RHS}(u^n)$$

$$u^{**} = \frac{3}{4} u^n + \frac{1}{4} u^* + \frac{1}{4} \Delta t \times \text{RHS}(u^*)$$

$$u^{n+1} = \frac{1}{3} u^n + \frac{2}{3} u^{**} + \frac{2}{3} \Delta t \times \text{RHS}(u^{**})$$

### 5.4 新显式格式 (time_scheme_type=1)

使用双时间层格式：

$$u^*_i = u^n_i \quad \text{(第一层)}$$

$$u^{**}_i = -u^{n-1}_i + 2 \cdot u^n_i \quad \text{(第二层)}$$

更新公式包含权重参数 $w$ (config.weight)。

---

## 6. TVD 限制器

使用 **Minmod 限制器**来保持解的单调性：

$$\text{minmod}(a, b, c) = \begin{cases} 0 & \text{如果 } a, b, c \text{ 符号不同} \\ \text{sign}(a) \cdot \min(|a|, |b|, |c|) & \text{如果同号} \end{cases}$$

应用限制器后，解被重构为：

$$u_i(\xi) = u_{\text{avg}} + c1_i \cdot L_i(\xi)$$

---

## 7. 物理模型

### 7.1 线性平流方程 (LAD)

- 守恒变量：$u$
- 通量：$F = a \cdot u$
- 粘性通量：$F_{\text{vis}} = \nu \cdot \frac{\partial u}{\partial x}$

### 7.2 Burgers 方程

- 守恒变量：$u$
- 通量：$F = \frac{1}{2} u^2$

### 7.3 一维 Navier-Stokes

- 守恒变量：$U = [\rho, \rho u, E]^T$
- 原始变量：$V = [\rho, u, p, T]^T$
- 通量：
  - $F_1 = \rho u$
  - $F_2 = \rho u^2 + p$
  - $F_3 = (E + p)u$

其中：
- $\gamma = 1.4$ (GAMMA)
- $R = 1.0$ (气体常数)

---

## 8. 代码结构

### 8.1 配置文件 (config.h)

| 参数 | 说明 |
|------|------|
| x0, x1 | 计算域边界 |
| n_ele | 单元数量 |
| dt | 时间步长 |
| total_time | 总模拟时间 |
| a | 平流系数 (LAD) |
| nu | 扩散系数 (LAD) |
| dg_fr_type | 0: DG, 1: FR |
| limiter_type | 0: 无限制器, 1: TVD |
| time_scheme_type | 0: RK, 1: 新显式格式 |
| enable_entropy_modify | 启用熵修正 |

### 8.2 主要函数

| 函数名 | 功能 |
|--------|------|
| Initialization() | 初始化网格和初值条件 |
| computeElemRhsDG() | 计算 DG 方法右端项 |
| computeElemRhsFR() | 计算 FR 方法右端项 |
| compPredictionLP() | 计算 LP (Lax-Friedrichs) 预测 |
| compPredictionEntropy() | 计算熵稳定预测 |
| compPredictionCR() | CR (中心重构) 预测 |
| timeRK1/RK2/RK3() | Runge-Kutta 时间推进 |
| timeNewExplicitSchemeK1() | 新显式格式 |
| TvdLimiter() | TVD 限制器应用 |
| computeElementGrad() | 计算单元梯度 |
| ComputeElementAvg() | 计算单元平均值 |
| Output() | 输出完整解 |
| OutputAvg() | 输出单元平均解 |

---

## 9. 总结

本代码实现了一个完整的一维 DG/FR 求解器，支持：

- 多种 PDE 模型：线性平流、Burgers、Navier-Stokes
- 两种无网格方法：DG 和 FR
- 多种时间推进格式：RK1, RK2, RK3, 新显式格式
- TVD 限制器保持解的单调性
- BR2 方法处理扩散项
- 熵修正提高数值稳定性

---

*生成日期: 2026-03-05*
