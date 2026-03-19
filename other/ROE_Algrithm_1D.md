# 基于原始变量变换法的特征值与Roe格式推导指南

## 目录
1. [引言](#引言)
2. [理论基础：原始变量变换法](#理论基础原始变量变换法)
3. [完整欧拉方程的推导](#完整欧拉方程的推导)
4. [弱可压缩NS方程的推导](#弱可压缩ns方程的推导)
5. [Roe数值通量格式实现](#roe数值通量格式实现)
6. [代码实现建议](#代码实现建议)
7. [总结](#总结)

---

## 引言

在计算流体力学（CFD）中，Roe格式是一种广泛使用的高分辨率数值格式，用于求解双曲型守恒律方程组。构造Roe格式的核心步骤是：
1. 计算Roe平均状态
2. 求解雅可比矩阵的特征值和右特征向量
3. 进行波强分解
4. 构造数值通量

直接对守恒变量形式的雅可比矩阵进行特征分解通常非常繁琐。**原始变量变换法**通过将问题转换到物理意义更清晰的原始变量空间，大大简化了这一过程。

本文档详细演示两种模型的推导过程：
- **完整欧拉方程**（可压缩流，包含能量方程）
- **弱可压缩NS方程**（人工可压缩性方法，无能量方程，线性状态方程）

---

## 理论基础：原始变量变换法

### 核心思想

对于守恒律方程组：
$$\frac{\partial U}{\partial t} + \frac{\partial F(U)}{\partial x} = 0$$

引入原始变量 $V$（如 $\rho, u, p$），使得：
- $U = U(V)$
- $F = F(V)$

利用链式法则：
$$\frac{\partial U}{\partial t} = \frac{\partial U}{\partial V}\frac{\partial V}{\partial t} = M\frac{\partial V}{\partial t}$$
$$\frac{\partial F}{\partial x} = \frac{\partial F}{\partial V}\frac{\partial V}{\partial x} = \tilde{B}\frac{\partial V}{\partial x}$$

原方程变为：
$$M\frac{\partial V}{\partial t} + \tilde{B}\frac{\partial V}{\partial x} = 0$$

左乘 $M^{-1}$：
$$\frac{\partial V}{\partial t} + \underbrace{(M^{-1}\tilde{B})}_{\tilde{B}_{prim}}\frac{\partial V}{\partial x} = 0$$

### 关键性质

1. **相似变换**：$\tilde{B}_{prim} = M^{-1}\tilde{A}M$，其中 $\tilde{A} = \frac{\partial F}{\partial U}$
2. **特征值相同**：$\tilde{A}$ 和 $\tilde{B}_{prim}$ 拥有相同的特征值 $\lambda_k$
3. **特征向量关系**：
   - 若 $\vec{k}_k$ 是 $\tilde{B}_{prim}$ 的右特征向量
   - 则 $\vec{r}_k = M\vec{k}_k$ 是 $\tilde{A}$ 的右特征向量

### 优势

- $\tilde{B}_{prim}$ 通常具有更简单的结构（更多零元素）
- 物理意义清晰，便于利用物理知识直接写出特征向量
- 避免了对复杂守恒变量雅可比矩阵的直接运算

---

## 完整欧拉方程的推导

### 1. 变量定义

**守恒变量**（3个）：
$$U = \begin{bmatrix} \rho \\ \rho u \\ \rho E \end{bmatrix}$$

**原始变量**（3个）：
$$V = \begin{bmatrix} \rho \\ u \\ p \end{bmatrix}$$

**状态方程**（理想气体）：
$$p = (\gamma-1)\left(\rho E - \frac{1}{2}\rho u^2\right)$$
$$\rho E = \frac{p}{\gamma-1} + \frac{1}{2}\rho u^2$$

**通量向量**：
$$F = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ (\rho E + p)u \end{bmatrix} = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ \left(\frac{\gamma}{\gamma-1}p + \frac{1}{2}\rho u^2\right)u \end{bmatrix}$$

### 2. 变换矩阵 $M$

计算 $M = \frac{\partial U}{\partial V}$：

$$dU = \begin{bmatrix} d\rho \\ d(\rho u) \\ d(\rho E) \end{bmatrix} = \begin{bmatrix} d\rho \\ u d\rho + \rho du \\ \frac{1}{\gamma-1}dp + \frac{1}{2}u^2 d\rho + \rho u du \end{bmatrix}$$

$$M = \begin{bmatrix} 
1 & 0 & 0 \\
u & \rho & 0 \\
\frac{1}{2}u^2 & \rho u & \frac{1}{\gamma-1}
\end{bmatrix}$$

**逆矩阵** $M^{-1}$：

$$M^{-1} = \begin{bmatrix} 
1 & 0 & 0 \\
-\frac{u}{\rho} & \frac{1}{\rho} & 0 \\
\frac{\gamma-1}{2}u^2 & -(\gamma-1)u & \gamma-1
\end{bmatrix}$$

### 3. 原始变量空间的雅可比矩阵

利用物理方程的直接形式（连续性、动量、压力方程）：

$$\frac{\partial V}{\partial t} + \tilde{B}_{prim}\frac{\partial V}{\partial x} = 0$$

其中：
$$\tilde{B}_{prim} = \begin{bmatrix} 
u & \rho & 0 \\
0 & u & \frac{1}{\rho} \\
0 & \gamma p & u
\end{bmatrix}$$

**验证**：该矩阵来自以下方程组：
- 连续性：$\rho_t + u\rho_x + \rho u_x = 0$
- 动量：$u_t + u u_x + \frac{1}{\rho}p_x = 0$
- 压力：$p_t + u p_x + \gamma p u_x = 0$

### 4. 特征值求解

解 $\det(\tilde{B}_{prim} - \lambda I) = 0$：

$$\det\begin{bmatrix} 
u-\lambda & \rho & 0 \\
0 & u-\lambda & \frac{1}{\rho} \\
0 & \gamma p & u-\lambda
\end{bmatrix} = (u-\lambda)\left[(u-\lambda)^2 - \frac{\gamma p}{\rho}\right] = 0$$

定义声速 $c = \sqrt{\frac{\gamma p}{\rho}}$，得到三个特征值：

$$\lambda_1 = u - c \quad \text{(左行声波)}$$
$$\lambda_2 = u \quad \text{(接触间断/熵波)}$$
$$\lambda_3 = u + c \quad \text{(右行声波)}$$

### 5. 原始变量空间的特征向量

#### (1) $\lambda_1 = u - c$（左行声波）

解 $(\tilde{B}_{prim} - \lambda_1 I)\vec{k}_1 = 0$：

$$\begin{bmatrix} 
c & \rho & 0 \\
0 & c & \frac{1}{\rho} \\
0 & \gamma p & c
\end{bmatrix}\begin{bmatrix} k_\rho \\ k_u \\ k_p \end{bmatrix} = 0$$

由第2行：$c k_u + \frac{1}{\rho}k_p = 0 \Rightarrow k_p = -\rho c k_u$

由第1行：$c k_\rho + \rho k_u = 0 \Rightarrow k_\rho = -\frac{\rho}{c}k_u$

取 $k_u = 1$，得：
$$\vec{k}_1 = \begin{bmatrix} -\frac{\rho}{c} \\ 1 \\ -\rho c \end{bmatrix}$$

为简化，令 $k_\rho = 1$，则 $k_u = -\frac{c}{\rho}$，$k_p = c^2$：
$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c}{\rho} \\ c^2 \end{bmatrix}$$

#### (2) $\lambda_2 = u$（接触间断）

解 $(\tilde{B}_{prim} - \lambda_2 I)\vec{k}_2 = 0$：

$$\begin{bmatrix} 
0 & \rho & 0 \\
0 & 0 & \frac{1}{\rho} \\
0 & \gamma p & 0
\end{bmatrix}\begin{bmatrix} k_\rho \\ k_u \\ k_p \end{bmatrix} = 0$$

得：$k_u = 0$，$k_p = 0$，$k_\rho$ 任意。

取 $k_\rho = 1$：
$$\vec{k}_2 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

#### (3) $\lambda_3 = u + c$（右行声波）

类似地，取 $k_\rho = 1$：
$$\vec{k}_3 = \begin{bmatrix} 1 \\ \frac{c}{\rho} \\ c^2 \end{bmatrix}$$

### 6. 映射回守恒变量空间

利用 $\vec{r}_k = M\vec{k}_k$：

#### (1) $\vec{r}_1$（左行声波）

$$\vec{r}_1 = \begin{bmatrix} 
1 & 0 & 0 \\
u & \rho & 0 \\
\frac{1}{2}u^2 & \rho u & \frac{1}{\gamma-1}
\end{bmatrix}\begin{bmatrix} 1 \\ -\frac{c}{\rho} \\ c^2 \end{bmatrix} = \begin{bmatrix} 
1 \\
u - c \\
\frac{1}{2}u^2 - uc + \frac{c^2}{\gamma-1}
\end{bmatrix}$$

利用总焓 $H = \frac{c^2}{\gamma-1} + \frac{1}{2}u^2$：
$$\vec{r}_1 = \begin{bmatrix} 1 \\ u-c \\ H - uc \end{bmatrix}$$

#### (2) $\vec{r}_2$（接触间断）

$$\vec{r}_2 = M\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ u \\ \frac{1}{2}u^2 \end{bmatrix}$$

#### (3) $\vec{r}_3$（右行声波）

$$\vec{r}_3 = \begin{bmatrix} 1 \\ u+c \\ H + uc \end{bmatrix}$$

### 7. 最终结果汇总

**特征值**：
$$\Lambda = \text{diag}(u-c, \quad u, \quad u+c)$$

**原始变量空间的右特征向量矩阵** $P_{prim}$：
$$P_{prim} = \begin{bmatrix}
1 & 1 & 1 \\
-\frac{c}{\rho} & 0 & \frac{c}{\rho} \\
c^2 & 0 & c^2
\end{bmatrix}$$

**逆矩阵** $P_{prim}^{-1}$（用于波强分解）：
$$P_{prim}^{-1} = \begin{bmatrix}
0 & -\frac{\rho}{2c} & \frac{1}{2c^2} \\
1 & 0 & -\frac{1}{c^2} \\
0 & \frac{\rho}{2c} & \frac{1}{2c^2}
\end{bmatrix}$$

**守恒变量空间的右特征向量矩阵** $P = MP_{prim}$：
$$P = \begin{bmatrix}
1 & 1 & 1 \\
u-c & u & u+c \\
H-uc & \frac{1}{2}u^2 & H+uc
\end{bmatrix}$$

**波强计算**：
$$\vec{\alpha} = P_{prim}^{-1}\Delta V, \quad \Delta V = \begin{bmatrix} \Delta\rho \\ \Delta u \\ \Delta p \end{bmatrix}$$

---

## 弱可压缩NS方程的推导

### 1. 模型描述

**假设**：
- 无能量方程
- 线性状态方程：$p = p_0 + c_s^2(\rho - \rho_0)$
- $c_s$ 为给定常数（人工声速）

**守恒变量**（2个）：
$$U = \begin{bmatrix} \rho \\ \rho u \end{bmatrix}$$

**原始变量**（2个）：
$$V = \begin{bmatrix} \rho \\ u \end{bmatrix}$$

**通量向量**：
$$F = \begin{bmatrix} \rho u \\ \rho u^2 + p \end{bmatrix} = \begin{bmatrix} \rho u \\ \rho u^2 + c_s^2\rho + \text{const} \end{bmatrix}$$

### 2. 变换矩阵 $M$

$$M = \frac{\partial U}{\partial V} = \begin{bmatrix} 
1 & 0 \\
u & \rho
\end{bmatrix}$$

**逆矩阵**：
$$M^{-1} = \begin{bmatrix} 
1 & 0 \\
-\frac{u}{\rho} & \frac{1}{\rho}
\end{bmatrix}$$

### 3. 原始变量空间的雅可比矩阵

计算 $\tilde{B} = \frac{\partial F}{\partial V}$：

$$\tilde{B} = \begin{bmatrix} 
u & \rho \\
u^2 + c_s^2 & 2\rho u
\end{bmatrix}$$

计算 $\tilde{B}_{prim} = M^{-1}\tilde{B}$：

$$\tilde{B}_{prim} = \begin{bmatrix} 
1 & 0 \\
-\frac{u}{\rho} & \frac{1}{\rho}
\end{bmatrix}\begin{bmatrix} 
u & \rho \\
u^2 + c_s^2 & 2\rho u
\end{bmatrix} = \begin{bmatrix} 
u & \rho \\
\frac{c_s^2}{\rho} & u
\end{bmatrix}$$

### 4. 特征值求解

解 $\det(\tilde{B}_{prim} - \lambda I) = 0$：

$$\det\begin{bmatrix} 
u-\lambda & \rho \\
\frac{c_s^2}{\rho} & u-\lambda
\end{bmatrix} = (u-\lambda)^2 - c_s^2 = 0$$

得到两个特征值：
$$\lambda_1 = u - c_s \quad \text{(左行波)}$$
$$\lambda_2 = u + c_s \quad \text{(右行波)}$$

**注意**：没有中间接触波（$\lambda=u$），因为去掉了能量方程。

### 5. 原始变量空间的特征向量

#### (1) $\lambda_1 = u - c_s$

解 $(\tilde{B}_{prim} - \lambda_1 I)\vec{k}_1 = 0$：

$$\begin{bmatrix} 
c_s & \rho \\
\frac{c_s^2}{\rho} & c_s
\end{bmatrix}\begin{bmatrix} k_\rho \\ k_u \end{bmatrix} = 0$$

由第1行：$c_s k_\rho + \rho k_u = 0 \Rightarrow k_u = -\frac{c_s}{\rho}k_\rho$

取 $k_\rho = 1$：
$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c_s}{\rho} \end{bmatrix}$$

#### (2) $\lambda_2 = u + c_s$

取 $k_\rho = 1$：
$$\vec{k}_2 = \begin{bmatrix} 1 \\ \frac{c_s}{\rho} \end{bmatrix}$$

### 6. 映射回守恒变量空间

#### (1) $\vec{r}_1$

$$\vec{r}_1 = M\vec{k}_1 = \begin{bmatrix} 
1 & 0 \\
u & \rho
\end{bmatrix}\begin{bmatrix} 1 \\ -\frac{c_s}{\rho} \end{bmatrix} = \begin{bmatrix} 1 \\ u - c_s \end{bmatrix}$$

#### (2) $\vec{r}_2$

$$\vec{r}_2 = M\vec{k}_2 = \begin{bmatrix} 1 \\ u + c_s \end{bmatrix}$$

### 7. 最终结果汇总

**特征值**：
$$\Lambda = \text{diag}(u-c_s, \quad u+c_s)$$

**原始变量空间的右特征向量矩阵** $P_{prim}$：
$$P_{prim} = \begin{bmatrix}
1 & 1 \\
-\frac{c_s}{\rho} & \frac{c_s}{\rho}
\end{bmatrix}$$

**逆矩阵** $P_{prim}^{-1}$：
$$P_{prim}^{-1} = \begin{bmatrix}
\frac{1}{2} & -\frac{\rho}{2c_s} \\
\frac{1}{2} & \frac{\rho}{2c_s}
\end{bmatrix}$$

**守恒变量空间的右特征向量矩阵** $P = MP_{prim}$：
$$P = \begin{bmatrix}
1 & 1 \\
u-c_s & u+c_s
\end{bmatrix}$$

**波强计算**：
$$\vec{\alpha} = P_{prim}^{-1}\Delta V, \quad \Delta V = \begin{bmatrix} \Delta\rho \\ \Delta u \end{bmatrix}$$

$$\alpha_1 = \frac{1}{2}\left(\Delta\rho - \frac{\rho}{c_s}\Delta u\right)$$
$$\alpha_2 = \frac{1}{2}\left(\Delta\rho + \frac{\rho}{c_s}\Delta u\right)$$

---

## Roe数值通量格式实现

### 通用公式

对于任意双曲系统，Roe数值通量为：

$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \frac{1}{2}\sum_{k=1}^{n}|\lambda_k|\alpha_k\vec{r}_k$$

其中：
- $F_L, F_R$：左右状态的物理通量
- $\lambda_k$：Roe平均状态下的特征值
- $\vec{r}_k$：对应的右特征向量
- $\alpha_k$：波强，由 $\vec{\alpha} = P^{-1}\Delta U$ 计算
- $\Delta U = U_R - U_L$

### Roe平均状态构造

#### 完整欧拉方程

$$\tilde{u} = \frac{\sqrt{\rho_L}u_L + \sqrt{\rho_R}u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{H} = \frac{\sqrt{\rho_L}H_L + \sqrt{\rho_R}H_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{c} = \sqrt{(\gamma-1)(\tilde{H} - \frac{1}{2}\tilde{u}^2)}$$
$$\tilde{\rho} = \sqrt{\rho_L \rho_R} \quad \text{(或其他兼容形式)}$$

#### 弱可压缩模型

$$\tilde{u} = \frac{\sqrt{\rho_L}u_L + \sqrt{\rho_R}u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{\rho} = \frac{\rho_L + \rho_R}{2} \quad \text{(算术平均即可)}$$
$$\tilde{c}_s = c_s \quad \text{(常数)}$$

### 算法步骤

#### 步骤1：计算Roe平均状态
使用左右状态 $U_L, U_R$ 计算 $\tilde{\rho}, \tilde{u}, \tilde{H}, \tilde{c}$。

#### 步骤2：计算特征值和特征向量
在Roe平均状态下计算 $\lambda_k$ 和 $\vec{r}_k$。

#### 步骤3：波强分解
计算 $\vec{\alpha} = P^{-1}\Delta U$，其中 $\Delta U = U_R - U_L$。

#### 步骤4：构造数值通量
$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \frac{1}{2}\sum_{k=1}^{n}|\lambda_k|\alpha_k\vec{r}_k$$

---

## 波强 $\alpha_k$ 的详细推导

### 核心思想：利用原始变量空间简化波强计算

直接计算 $P^{-1}$（守恒变量空间的左特征向量矩阵）较为复杂。利用原始变量变换，可以大大简化推导。

**关键关系**：
- $\Delta U = M \Delta V$（守恒变量跳跃与原始变量跳跃的关系）
- $P = M P_{prim}$（特征向量矩阵的关系）
- 因此：$P^{-1} = P_{prim}^{-1} M^{-1}$

**波强公式**：
$$\vec{\alpha} = P^{-1}\Delta U = P_{prim}^{-1} M^{-1} \cdot M \Delta V = P_{prim}^{-1}\Delta V$$

**优势**：
1. $P_{prim}$ 结构简单，容易求逆
2. $\Delta V = [\Delta\rho, \Delta u, \Delta p]^T$ 直接使用原始变量跳跃，物理意义清晰
3. 避免了对复杂矩阵 $P$ 的求逆

---

### 完整欧拉方程的波强计算

#### 原始变量空间的特征向量矩阵

由前文推导，原始变量空间的右特征向量（取 $k_\rho = 1$）：

$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c}{\rho} \\ c^2 \end{bmatrix}, \quad \vec{k}_2 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \vec{k}_3 = \begin{bmatrix} 1 \\ \frac{c}{\rho} \\ c^2 \end{bmatrix}$$

组成矩阵：
$$P_{prim} = \begin{bmatrix} 1 & 1 & 1 \\ -\frac{c}{\rho} & 0 & \frac{c}{\rho} \\ c^2 & 0 & c^2 \end{bmatrix}$$

#### 求解 $P_{prim}^{-1}$

设 $P_{prim}^{-1} = \begin{bmatrix} q_{11} & q_{12} & q_{13} \\ q_{21} & q_{22} & q_{23} \\ q_{31} & q_{32} & q_{33} \end{bmatrix}$，由 $P_{prim}^{-1}P_{prim} = I$：

**第1行**：
- 第2列：$q_{11} \cdot 1 + q_{12} \cdot 0 + q_{13} \cdot 0 = 0 \Rightarrow q_{11} = 0$
- 第1列减第3列：$q_{12} \cdot (-\frac{c}{\rho}) + q_{13} \cdot c^2 = 1$，$q_{12} \cdot \frac{c}{\rho} + q_{13} \cdot c^2 = 0$
- 两式相减得：$q_{12} \cdot (-\frac{2c}{\rho}) = 1 \Rightarrow q_{12} = -\frac{\rho}{2c}$
- 代入：$\frac{\rho}{2c} \cdot \frac{c}{\rho} + q_{13} c^2 = 0 \Rightarrow q_{13} = \frac{1}{2c^2}$

**第2行**：
- 第2列：$q_{21} = 1$
- 第1列减第3列得：$q_{22} = 0$，$q_{23} = -\frac{1}{c^2}$

**第3行**：
- 第2列：$q_{31} = 0$
- 类似得：$q_{32} = \frac{\rho}{2c}$，$q_{33} = \frac{1}{2c^2}$

因此：
$$P_{prim}^{-1} = \begin{bmatrix} 0 & -\frac{\rho}{2c} & \frac{1}{2c^2} \\ 1 & 0 & -\frac{1}{c^2} \\ 0 & \frac{\rho}{2c} & \frac{1}{2c^2} \end{bmatrix}$$

#### 波强公式

设 $\Delta V = [\Delta\rho, \Delta u, \Delta p]^T$，则：
$$\vec{\alpha} = P_{prim}^{-1}\Delta V$$

$$\begin{bmatrix} \alpha_1 \\ \alpha_2 \\ \alpha_3 \end{bmatrix} = \begin{bmatrix} 0 & -\frac{\rho}{2c} & \frac{1}{2c^2} \\ 1 & 0 & -\frac{1}{c^2} \\ 0 & \frac{\rho}{2c} & \frac{1}{2c^2} \end{bmatrix} \begin{bmatrix} \Delta\rho \\ \Delta u \\ \Delta p \end{bmatrix}$$

**结果**：
$$\boxed{\alpha_1 = \frac{\Delta p - \rho c \Delta u}{2c^2}}$$

$$\boxed{\alpha_2 = \Delta\rho - \frac{\Delta p}{c^2}}$$

$$\boxed{\alpha_3 = \frac{\Delta p + \rho c \Delta u}{2c^2}}$$

#### 物理意义

- $\alpha_1$：左行声波强度，与 $\Delta p - \rho c \Delta u$ 成正比（压力波与速度波的组合）
- $\alpha_2$：接触波（熵波）强度，与密度跳跃相关
- $\alpha_3$：右行声波强度，与 $\Delta p + \rho c \Delta u$ 成正比

**注意**：在实际计算中，$\rho, c$ 取Roe平均值 $\tilde{\rho}, \tilde{c}$。

---

### 弱可压缩NS方程的波强计算

#### 原始变量空间

对于弱可压缩模型，原始变量 $V = [\rho, u]^T$（无压力变量）。

特征向量：
$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c_s}{\rho} \end{bmatrix}, \quad \vec{k}_2 = \begin{bmatrix} 1 \\ \frac{c_s}{\rho} \end{bmatrix}$$

组成矩阵：
$$P_{prim} = \begin{bmatrix} 1 & 1 \\ -\frac{c_s}{\rho} & \frac{c_s}{\rho} \end{bmatrix}$$

#### 求解 $P_{prim}^{-1}$

$$P_{prim}^{-1} = \frac{1}{\det(P_{prim})}\begin{bmatrix} \frac{c_s}{\rho} & -1 \\ \frac{c_s}{\rho} & 1 \end{bmatrix} = \frac{1}{\frac{2c_s}{\rho}}\begin{bmatrix} \frac{c_s}{\rho} & -1 \\ \frac{c_s}{\rho} & 1 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & -\frac{\rho}{2c_s} \\ \frac{1}{2} & \frac{\rho}{2c_s} \end{bmatrix}$$

#### 波强公式

$$\begin{bmatrix} \alpha_1 \\ \alpha_2 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & -\frac{\rho}{2c_s} \\ \frac{1}{2} & \frac{\rho}{2c_s} \end{bmatrix} \begin{bmatrix} \Delta\rho \\ \Delta u \end{bmatrix}$$

**结果**：
$$\boxed{\alpha_1 = \frac{1}{2}\left(\Delta\rho - \frac{\rho}{c_s}\Delta u\right)}$$

$$\boxed{\alpha_2 = \frac{1}{2}\left(\Delta\rho + \frac{\rho}{c_s}\Delta u\right)}$$

---

### 展开形式：Roe数值通量

通用形式：
$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \Phi$$

其中耗散项：
$$\Phi = \frac{1}{2}\sum_{k=1}^{n}|\lambda_k|\alpha_k\vec{r}_k$$

#### 完整欧拉方程

$$\Phi = \frac{1}{2}\left[|u-c|\alpha_1\vec{r}_1 + |u|\alpha_2\vec{r}_2 + |u+c|\alpha_3\vec{r}_3\right]$$

右特征向量 $\vec{r}_k = M\vec{k}_k$：
$$\vec{r}_1 = \begin{bmatrix} 1 \\ u-c \\ H-uc \end{bmatrix}, \quad \vec{r}_2 = \begin{bmatrix} 1 \\ u \\ \frac{1}{2}u^2 \end{bmatrix}, \quad \vec{r}_3 = \begin{bmatrix} 1 \\ u+c \\ H+uc \end{bmatrix}$$

**分量形式**：
$$\Phi_\rho = \frac{1}{2}\left[|u-c|\alpha_1 + |u|\alpha_2 + |u+c|\alpha_3\right]$$

$$\Phi_{\rho u} = \frac{1}{2}\left[|u-c|\alpha_1(u-c) + |u|\alpha_2 u + |u+c|\alpha_3(u+c)\right]$$

$$\Phi_{\rho E} = \frac{1}{2}\left[|u-c|\alpha_1(H-uc) + |u|\alpha_2\cdot\frac{1}{2}u^2 + |u+c|\alpha_3(H+uc)\right]$$

#### 弱可压缩NS方程

$$\Phi = \frac{1}{2}\left[|u-c_s|\alpha_1\vec{r}_1 + |u+c_s|\alpha_2\vec{r}_2\right]$$

右特征向量：
$$\vec{r}_1 = \begin{bmatrix} 1 \\ u-c_s \end{bmatrix}, \quad \vec{r}_2 = \begin{bmatrix} 1 \\ u+c_s \end{bmatrix}$$

**分量形式**：
$$\Phi_\rho = \frac{1}{2}\left[|u-c_s|\alpha_1 + |u+c_s|\alpha_2\right]$$

$$\Phi_{\rho u} = \frac{1}{2}\left[|u-c_s|\alpha_1(u-c_s) + |u+c_s|\alpha_2(u+c_s)\right]$$

---

### 方法对比

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| 直接法 | $\alpha = P^{-1}\Delta U$ | 直接使用守恒变量跳跃 | $P^{-1}$ 求解复杂 |
| **原始变量法** | $\alpha = P_{prim}^{-1}\Delta V$ | $P_{prim}^{-1}$ 简单，物理意义清晰 | 需要先计算 $\Delta V$ |

**推荐**：使用原始变量法，计算流程：
1. 计算原始变量跳跃 $\Delta V = [\Delta\rho, \Delta u, \Delta p]^T$
2. 用 $P_{prim}^{-1}$ 计算波强 $\alpha$
3. 用守恒变量空间的 $\vec{r}_k$ 构造耗散项

---

## 代码实现建议

### 完整欧拉方程（伪代码）

```python
def roe_flux_euler(U_L, U_R, gamma=1.4):
    # 步骤1：解析原始变量
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # 步骤2：Roe平均
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    rho_tilde = np.sqrt(rho_L * rho_R)  # Roe平均密度
    u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_L = (U_L[2] + p_L) / rho_L
    H_R = (U_R[2] + p_R) / rho_R
    H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_tilde = np.sqrt((gamma - 1) * (H_tilde - 0.5 * u_tilde**2))

    # 步骤3：计算原始变量跳跃量
    Delta_rho = rho_R - rho_L
    Delta_u = u_R - u_L
    Delta_p = p_R - p_L

    # 步骤4：波强计算（使用 P_prim^{-1} * Delta_V）
    # P_prim^{-1} = [[0, -rho/(2c), 1/(2c^2)],
    #                [1, 0, -1/c^2],
    #                [0, rho/(2c), 1/(2c^2)]]
    alpha1 = (Delta_p - rho_tilde * c_tilde * Delta_u) / (2 * c_tilde**2)
    alpha2 = Delta_rho - Delta_p / c_tilde**2
    alpha3 = (Delta_p + rho_tilde * c_tilde * Delta_u) / (2 * c_tilde**2)

    # 步骤5：特征值
    lambda1 = u_tilde - c_tilde
    lambda2 = u_tilde
    lambda3 = u_tilde + c_tilde

    # 步骤6：右特征向量（守恒变量空间）
    r1 = np.array([1, u_tilde - c_tilde, H_tilde - u_tilde * c_tilde])
    r2 = np.array([1, u_tilde, 0.5 * u_tilde**2])
    r3 = np.array([1, u_tilde + c_tilde, H_tilde + u_tilde * c_tilde])

    # 步骤7：耗散项 Phi = 0.5 * sum(|lambda_k| * alpha_k * r_k)
    Phi = 0.5 * (abs(lambda1) * alpha1 * r1 +
                 abs(lambda2) * alpha2 * r2 +
                 abs(lambda3) * alpha3 * r3)

    # 步骤8：物理通量
    F_L = flux_from_primitive(rho_L, u_L, p_L, gamma)
    F_R = flux_from_primitive(rho_R, u_R, p_R, gamma)

    return 0.5 * (F_L + F_R) - Phi
```

### 弱可压缩模型（伪代码）

```python
def roe_flux_weak_compressible(U_L, U_R, c_s):
    # 步骤1：解析原始变量
    rho_L, u_L = U_L[0], U_L[1] / U_L[0]
    rho_R, u_R = U_R[0], U_R[1] / U_R[0]

    # 步骤2：Roe平均
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    rho_tilde = 0.5 * (rho_L + rho_R)
    u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)

    # 步骤3：计算原始变量跳跃量
    Delta_rho = rho_R - rho_L
    Delta_u = u_R - u_L

    # 步骤4：波强计算（使用 P_prim^{-1} * Delta_V）
    # P_prim^{-1} = [[1/2, -rho/(2c_s)],
    #                [1/2, rho/(2c_s)]]
    alpha1 = 0.5 * (Delta_rho - rho_tilde / c_s * Delta_u)
    alpha2 = 0.5 * (Delta_rho + rho_tilde / c_s * Delta_u)

    # 步骤5：特征值
    lambda1 = u_tilde - c_s
    lambda2 = u_tilde + c_s

    # 步骤6：右特征向量（守恒变量空间）
    r1 = np.array([1, u_tilde - c_s])
    r2 = np.array([1, u_tilde + c_s])

    # 步骤7：耗散项
    Phi = 0.5 * (abs(lambda1) * alpha1 * r1 + abs(lambda2) * alpha2 * r2)

    # 步骤8：物理通量
    F_L = np.array([rho_L * u_L, rho_L * u_L**2 + c_s**2 * rho_L])
    F_R = np.array([rho_R * u_R, rho_R * u_R**2 + c_s**2 * rho_R])

    return 0.5 * (F_L + F_R) - Phi
```

---

## 总结

本文档通过原始变量变换法，系统地推导了两种模型的Roe格式：

1. **完整欧拉方程**：3个特征波（左行声波、接触波、右行声波）
2. **弱可压缩NS方程**：2个特征波（左行波、右行波）

### 关键公式汇总

| 模型 | 特征值 | $P_{prim}^{-1}$ | 波强公式 |
|------|--------|-----------------|----------|
| 欧拉 | $u-c, u, u+c$ | $\begin{bmatrix} 0 & -\frac{\rho}{2c} & \frac{1}{2c^2} \\ 1 & 0 & -\frac{1}{c^2} \\ 0 & \frac{\rho}{2c} & \frac{1}{2c^2} \end{bmatrix}$ | $\alpha_1 = \frac{\Delta p - \rho c \Delta u}{2c^2}$<br>$\alpha_2 = \Delta\rho - \frac{\Delta p}{c^2}$<br>$\alpha_3 = \frac{\Delta p + \rho c \Delta u}{2c^2}$ |
| 弱可压缩 | $u-c_s, u+c_s$ | $\begin{bmatrix} \frac{1}{2} & -\frac{\rho}{2c_s} \\ \frac{1}{2} & \frac{\rho}{2c_s} \end{bmatrix}$ | $\alpha_1 = \frac{1}{2}\left(\Delta\rho - \frac{\rho}{c_s}\Delta u\right)$<br>$\alpha_2 = \frac{1}{2}\left(\Delta\rho + \frac{\rho}{c_s}\Delta u\right)$ |

### 核心思想

**波强计算**：
$$\vec{\alpha} = P_{prim}^{-1}\Delta V$$

**数值通量**：
$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \frac{1}{2}\sum_{k}|\lambda_k|\alpha_k\vec{r}_k$$

**原始变量变换法的优势**：
1. $P_{prim}$ 结构简单，容易求逆
2. $\Delta V$ 直接使用原始变量跳跃，物理意义清晰
3. 避免了对复杂守恒变量特征向量矩阵 $P$ 的直接求逆