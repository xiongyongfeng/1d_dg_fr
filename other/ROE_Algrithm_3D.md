# 三维（3D）基于原始变量变换法的特征值与Roe格式推导指南

## 目录
1. [引言：从1D到3D的扩展](#引言从1d到3d的扩展)
2. [理论基础：3D原始变量变换与法向投影](#理论基础3d原始变量变换与法向投影)
3. [3D完整欧拉方程的推导](#3d完整欧拉方程的推导)
4. [3D弱可压缩NS方程的推导](#3d弱可压缩ns方程的推导)
5. [Roe数值通量格式实现](#roe数值通量格式实现)
6. [代码实现](#代码实现)
7. [总结与对比](#总结与对比)

---

## 引言：从1D到3D的扩展

在三维计算流体力学（CFD）中，控制方程组的形式为：
$$\frac{\partial U}{\partial t} + \frac{\partial F}{\partial x} + \frac{\partial G}{\partial y} + \frac{\partial H}{\partial z} = 0$$

构造Roe格式的核心挑战在于处理多维界面上的通量。对于任意网格面，我们定义其**单位法向量** $\vec{n} = (n_x, n_y, n_z)$。

通过坐标旋转，三维问题在局部界面法向方向上可以退化为一个**准一维问题**。

**核心策略**：
1. 将守恒变量和通量投影到法向方向
2. 定义法向速度 $u_n = \vec{u} \cdot \vec{n}$
3. 利用1D推导的特征结构，但需考虑**横向速度分量**（切向速度）带来的简并特征值和剪切波

**3D相比1D的关键区别**：
- 1D：3个波（左行声波、接触波、右行声波）
- 3D：5个波（左行声波、熵波、2个剪切波、右行声波）

本文档详细演示两种模型在3D下的推导：
- **3D完整欧拉方程**（5个方程：质量、3动量、能量）
- **3D弱可压缩NS方程**（4个方程：质量、3动量，无能量，线性状态方程）

---

## 理论基础：3D原始变量变换与法向投影

### 1. 变量定义

**守恒变量 $U$**：
- **欧拉**: $U = [\rho, \rho u, \rho v, \rho w, \rho E]^T$
- **弱可压缩**: $U = [\rho, \rho u, \rho v, \rho w]^T$

**原始变量 $V$**：
- **欧拉**: $V = [\rho, u, v, w, p]^T$
- **弱可压缩**: $V = [\rho, u, v, w]^T$（压力由密度决定）

### 2. 法向通量与雅可比矩阵

对于法向量为 $\vec{n}$ 的界面，我们需要计算的通量是法向通量 $F_n$：
$$F_n = F n_x + G n_y + H n_z$$

对应的雅可比矩阵 $\tilde{A}_n$ 定义为：
$$\tilde{A}_n = \frac{\partial F_n}{\partial U} = \frac{\partial F}{\partial U}n_x + \frac{\partial G}{\partial U}n_y + \frac{\partial H}{\partial U}n_z$$

**关键性质**：
$\tilde{A}_n$ 的特征值 $\lambda$ 和右特征向量 $\vec{r}$ 决定了该界面上的波传播特性。

### 3. 法向速度与切向速度

**法向速度**：
$$u_n = u n_x + v n_y + w n_z$$

**切向速度向量**：
选择两个与 $\vec{n}$ 正交的单位向量 $\vec{t}_1, \vec{t}_2$，满足：
$$\vec{t}_1 \cdot \vec{n} = 0, \quad \vec{t}_2 \cdot \vec{n} = 0, \quad \vec{t}_1 \cdot \vec{t}_2 = 0$$

切向速度分量：
$$u_{t_1} = \vec{u} \cdot \vec{t}_1, \quad u_{t_2} = \vec{u} \cdot \vec{t}_2$$

**速度分解**：
$$\vec{u} = u_n \vec{n} + u_{t_1} \vec{t}_1 + u_{t_2} \vec{t}_2$$

**切向速度差**（用于剪切波）：
$$\Delta \vec{u}_t = \Delta \vec{u} - (\Delta u_n) \vec{n}$$

其中 $\Delta u_n = \Delta \vec{u} \cdot \vec{n}$。

---

## 3D完整欧拉方程的推导

### 1. 变量与变换矩阵 $M$

**变量**：$U = [\rho, \rho u, \rho v, \rho w, \rho E]^T$，$V = [\rho, u, v, w, p]^T$

**变换矩阵 $M = \frac{\partial U}{\partial V}$** (5×5):

$$M = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
u & \rho & 0 & 0 & 0 \\
v & 0 & \rho & 0 & 0 \\
w & 0 & 0 & \rho & 0 \\
\frac{1}{2}q^2 & \rho u & \rho v & \rho w & \frac{1}{\gamma-1}
\end{bmatrix}$$

其中 $q^2 = u^2 + v^2 + w^2$。

**逆矩阵 $M^{-1}$**:
$$M^{-1} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
-\frac{u}{\rho} & \frac{1}{\rho} & 0 & 0 & 0 \\
-\frac{v}{\rho} & 0 & \frac{1}{\rho} & 0 & 0 \\
-\frac{w}{\rho} & 0 & 0 & \frac{1}{\rho} & 0 \\
\frac{\gamma-1}{2}q^2 & -(\gamma-1)u & -(\gamma-1)v & -(\gamma-1)w & \gamma-1
\end{bmatrix}$$

### 2. 原始变量空间的法向雅可比矩阵 $\tilde{B}_{n, prim}$

在原始变量空间，3D欧拉方程的法向形式可以写为：
$$\frac{\partial V}{\partial t} + \tilde{B}_{n, prim} \frac{\partial V}{\partial n} = 0$$

上面的公式其实基于一个假设，认为$\tilde B_{n,prim} = \tilde B_x n_x + \tilde B_y n_y + \tilde B_z n_z$
原始变量的ns方程组为
$$ \frac{\partial V} {\partial t } + \tilde B_x \frac{\partial V} {\partial x } +  \tilde B_y \frac{\partial V} {\partial y } + \tilde B_z \frac{\partial V} {\partial z }=0$$

此方程数学意义上与$\frac{\partial V}{\partial t} + \tilde{B}_{n, prim} \frac{\partial V}{\partial n} = 0$并不等价，这里基于波传播理论，认为波主要沿法向传播，$\tilde B_x$代表x方向的波传播特性，$\tilde B_y$代表y方向的波传播特性，$\tilde B_z$代表z方向的波传播特性，那可以认为$\tilde B_{n,prim} = \tilde B_x n_x + \tilde B_y n_y + \tilde B_z n_z$代表波沿法向的传播特性

其中
$$\tilde{B}_{x} = \begin{bmatrix}
u & \rho  & 0 & 0 & 0 \\
0 & u & 0 & 0 & \frac{1}{\rho} \\
0 & 0 & u & 0 & 0 \\
0 & 0 & 0 & u & 0 \\
0 & \gamma p  & 0 & 0 & u
\end{bmatrix}$$
$$\tilde{B}_{y} = \begin{bmatrix}
v & 0  & \rho & 0 & 0 \\
0 & v & 0 & 0 & 0 \\
0 & 0 & v & 0 & \frac{1}{\rho} \\
0 & 0 & 0 & v & 0 \\
0 & 0  & \gamma p & 0 & v
\end{bmatrix}$$
$$\tilde{B}_{z} = \begin{bmatrix}
w & \rho  & 0 & 0 & 0 \\
0 & w & 0 & 0 & 0 \\
0 & 0 & w & 0 & 0 \\
0 & 0 & 0 & w & \frac{1}{\rho} \\
0 & 0  & 0 & \gamma p  & w
\end{bmatrix}$$

经过推导，$\tilde{B}_{n, prim}$ 为：
$$\tilde{B}_{n, prim} = \begin{bmatrix}
u_n & \rho n_x & \rho n_y & \rho n_z & 0 \\
0 & u_n & 0 & 0 & \frac{n_x}{\rho} \\
0 & 0 & u_n & 0 & \frac{n_y}{\rho} \\
0 & 0 & 0 & u_n & \frac{n_z}{\rho} \\
0 & \gamma p n_x & \gamma p n_y & \gamma p n_z & u_n
\end{bmatrix}$$ 

**简化形式**（选择局部坐标系使 $x'$ 轴沿 $\vec{n}$ 方向）：
$$\tilde{B}_{n, prim}' = \begin{bmatrix}
u_n & \rho & 0 & 0 & 0 \\
0 & u_n & 0 & 0 & \frac{1}{\rho} \\
0 & 0 & u_n & 0 & 0 \\
0 & 0 & 0 & u_n & 0 \\
0 & \gamma p & 0 & 0 & u_n
\end{bmatrix}$$

### 3. 特征值求解

观察简化矩阵 $\tilde{B}_{n, prim}'$，其特征方程为：
$$\det(\tilde{B}_{n, prim}' - \lambda I) = (u_n - \lambda)^3 \left[(u_n-\lambda)^2 - c^2\right] = 0$$

其中 $c = \sqrt{\gamma p / \rho}$。

**5个特征值**：
| 编号 | 特征值 | 波类型 | 物理意义 |
|------|--------|--------|----------|
| $\lambda_1$ | $u_n - c$ | 左行声波 | 压力波，沿法向负方向传播 |
| $\lambda_2$ | $u_n$ | 熵波 | 密度扰动，速度和压力不变 |
| $\lambda_3$ | $u_n$ | 剪切波1 | 切向速度扰动（$t_1$方向） |
| $\lambda_4$ | $u_n$ | 剪切波2 | 切向速度扰动（$t_2$方向） |
| $\lambda_5$ | $u_n + c$ | 右行声波 | 压力波，沿法向正方向传播 |

**注意**：$\lambda_2, \lambda_3, \lambda_4$ 是三重简并特征值。

其实即使不简化$\tilde B_{n,prim}$，可以证明其特征值也是$u_n-c, u_n,u_n,u_n,u_n+c$


### 4. 原始变量空间的特征向量

在局部法向坐标系下（$n_x'=1, n_y'=0, n_z'=0$），求解 $(\tilde{B}_{n, prim}' - \lambda I)\vec{k} = 0$：

#### (1) $\lambda_1 = u_n - c$（左行声波）

解得：
$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c}{\rho} \\ 0 \\ 0 \\ c^2 \end{bmatrix}$$

#### (2) $\lambda_5 = u_n + c$（右行声波）

解得：
$$\vec{k}_5 = \begin{bmatrix} 1 \\ \frac{c}{\rho} \\ 0 \\ 0 \\ c^2 \end{bmatrix}$$

#### (3) $\lambda_2 = u_n$（熵波）

对应密度扰动，压力和速度不变：
$$\vec{k}_2 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

#### (4) $\lambda_3 = u_n$（剪切波1）

对应 $t_1$ 方向切向速度扰动：
$$\vec{k}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}$$

#### (5) $\lambda_4 = u_n$（剪切波2）

对应 $t_2$ 方向切向速度扰动：
$$\vec{k}_4 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$$

### 5. 原始变量空间的特征向量矩阵 $P_{prim}$

$$P_{prim} = \begin{bmatrix}
1 & 1 & 0 & 0 & 1 \\
-\frac{c}{\rho} & 0 & 0 & 0 & \frac{c}{\rho} \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
c^2 & 0 & 0 & 0 & c^2
\end{bmatrix}$$

**注意**：这是在局部法向坐标系下的形式。在全局坐标系下，特征向量需要通过坐标变换。

对于全局坐标，可以验证$\tilde B_{n,prim}\vec k_m = \lambda_m \vec k_m$，其中

$$\vec{k}_1 = \begin{bmatrix} 1 \\ -\frac{c n_x}{\rho} \\ -\frac{c n_y}{\rho} \\ -\frac{c n_z}{\rho} \\ c^2 \end{bmatrix}$$
$$\vec{k}_2 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$
$$\vec{k}_3 = \begin{bmatrix} 0 \\ t_{1x} \\ t_{1y} \\ t_{1z} \\ 0 \end{bmatrix}$$
$$\vec{k}_4 = \begin{bmatrix} 0 \\ t_{2x} \\ t_{2y} \\ t_{2z} \\ 0 \end{bmatrix}$$
$$\vec{k}_5 = \begin{bmatrix} 1 \\ \frac{c n_x}{\rho} \\ \frac{c n_y}{\rho} \\ \frac{c n_z}{\rho} \\ c^2 \end{bmatrix}$$

### 6. 守恒变量空间的右特征向量 $\vec{r}_k$

利用 $\vec{r}_k = M\vec{k}_k$，并转换到全局坐标系，
或者不需要转到全局坐标直接用全局坐标的$\vec k_k$, 利用$\vec{r}_k = M\vec{k}_k$即可获得守恒变量的右行特征矢量如下，这里重新写一下转化矩阵$M$
$$M = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
u & \rho & 0 & 0 & 0 \\
v & 0 & \rho & 0 & 0 \\
w & 0 & 0 & \rho & 0 \\
\frac{1}{2}q^2 & \rho u & \rho v & \rho w & \frac{1}{\gamma-1}
\end{bmatrix}$$

#### (1) 左行声波 ($\lambda_1 = u_n - c$)

$$\vec{r}_1 = \begin{bmatrix} 1 \\ u - c n_x \\ v - c n_y \\ w - c n_z \\ H - u_n c \end{bmatrix}$$

#### (2) 右行声波 ($\lambda_5 = u_n + c$)

$$\vec{r}_5 = \begin{bmatrix} 1 \\ u + c n_x \\ v + c n_y \\ w + c n_z \\ H + u_n c \end{bmatrix}$$

#### (3) 熵波 ($\lambda_2 = u_n$)

$$\vec{r}_2 = \begin{bmatrix} 1 \\ u \\ v \\ w \\ \frac{1}{2}q^2 \end{bmatrix}$$

#### (4) 剪切波 ($\lambda_3 = \lambda_4 = u_n$)

对于剪切波，对应的特征向量与切向速度方向相关：

**剪切波1**（对应切向方向 $\vec{t}_1$）：
$$\vec{r}_3 = \begin{bmatrix} 0 \\ t_{1x} \\ t_{1y} \\ t_{1z} \\ \vec{u} \cdot \vec{t}_1 \end{bmatrix}$$

**剪切波2**（对应切向方向 $\vec{t}_2$）：
$$\vec{r}_4 = \begin{bmatrix} 0 \\ t_{2x} \\ t_{2y} \\ t_{2z} \\ \vec{u} \cdot \vec{t}_2 \end{bmatrix}$$

### 7. 波强计算

**关键技巧**：对于3D Roe格式，我们不需要显式构造剪切波的特征向量，而是利用**子空间投影法**。

#### 波强公式

**声波波强**（使用原始变量法，参考1D推导）：
$$\alpha_1 = \frac{\Delta p - \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$$
$$\alpha_5 = \frac{\Delta p + \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$$

**熵波波强**：
$$\alpha_2 = \Delta\rho - \frac{\Delta p}{\tilde{c}^2}$$

**剪切波波强**：
剪切波对应切向速度差 $\Delta \vec{u}_t$，其"波强"可以理解为：
$$\vec{\alpha}_{shear} = \tilde{\rho} \Delta \vec{u}_t$$

#### 简化的耗散项计算

**终极技巧**：利用子空间投影，避免显式分解剪切波。

$$\Phi = \frac{1}{2}\sum_{k=1}^{5}|\lambda_k|\alpha_k\vec{r}_k$$

分解为：
1. **声波贡献**：$|\lambda_1|\alpha_1\vec{r}_1 + |\lambda_5|\alpha_5\vec{r}_5$
2. **熵波贡献**：$|u_n|\alpha_2\vec{r}_2$
3. **剪切波贡献**：利用切向速度差直接计算

这里补充一下完整的波强计算公式，
$$\alpha _k  = P^{-1} \vec r_k$$
其中$P = \left[\vec r_1, \vec r_2, \vec r_3, \vec r_4, \vec r_5\right] = \left[ M\vec k_1, M\vec k_2, \vec k_3, \vec k_4, \vec k_5 \right] = M\left[\vec k_1, \vec k_2, \vec k_3, \vec k_4, \vec k_5\right] = M P_{prim}$ 为守恒变量对应的右行特征向量构成的矩阵, $P_{prim}$为原始变量对应的右行特征向量矩阵。$\vec r_k = M \vec k_k$,由此可得
$$\alpha _k = P_{prim}^{-1}\vec k_k$$
其中$P_{prim}^{-1}$为左行特征向量矩阵

$$P_{prim}^{-1} = \begin{bmatrix} l_1^T \\ l_2^T \\ l_3^T \\ l_4^T \\ l_5^T \end{bmatrix}$$
$$l_1 = \begin{bmatrix} 0 \\ -\rho n_x / (2c) \\  -\rho n_y / (2c)  \\  -\rho n_z / (2c)  \\ 1/(2c^2) \end{bmatrix}$$
$$l_2 = \begin{bmatrix} 1 \\ 0\\0\\0\\-1/c^2 \end{bmatrix}$$
$$l_3 = \begin{bmatrix} 0 \\ t_{1x}\\ t_{1y}\\ t_{1z}\\ 0 \end{bmatrix}$$
$$l_4 = \begin{bmatrix} 0 \\ t_{2x}\\ t_{2y}\\ t_{2z}\\ 0 \end{bmatrix}$$
$$l_1 = \begin{bmatrix} 0 \\ \rho n_x / (2c) \\  \rho n_y / (2c)  \\  \rho n_z / (2c)  \\ 1/(2c^2) \end{bmatrix}$$

这里隐含了左行特征向量矩阵是右行特征向量矩阵的逆。

然后可求$\alpha_k = l_k^T \Delta V$, $\Delta V =[\Delta rho , \Delta u, \Delta v, \Delta w, \Delta p]^T $ 
$$\alpha_1 = l_1^T \Delta V = \Delta p / (2c^2) - \rho \Delta u_n / (2c)$$
$$\alpha 2 = l_2^T \Delta V = \Delta \
rho - \Delta p / c^2$$
$$\alpha 3 = l_2^T \Delta V = \Delta u_{t_1} = \Delta \vec u \cdot \vec t_1$$
$$\alpha 4 = l_2^T \Delta V = \Delta u_{t_2} = \Delta \vec u \cdot \vec t_2$$
$$\alpha_5 = l_1^T \Delta V = \Delta p / (2c^2) + \rho \Delta u_n / (2c)$$


---

## Roe数值通量格式实现

### 通用公式

$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \Phi$$

其中耗散项：
$$\Phi = \frac{1}{2}\sum_{k=1}^{5}|\lambda_k|\alpha_k\vec{r}_k$$

### 分步计算

#### 步骤1：计算Roe平均状态

$$\tilde{u} = \frac{\sqrt{\rho_L}u_L + \sqrt{\rho_R}u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{v} = \frac{\sqrt{\rho_L}v_L + \sqrt{\rho_R}v_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{w} = \frac{\sqrt{\rho_L}w_L + \sqrt{\rho_R}w_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{H} = \frac{\sqrt{\rho_L}H_L + \sqrt{\rho_R}H_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$
$$\tilde{c} = \sqrt{(\gamma-1)(\tilde{H} - \frac{1}{2}\tilde{q}^2)}$$
$$\tilde{\rho} = \sqrt{\rho_L \rho_R}$$

#### 步骤2：计算法向速度和跳跃量

**法向速度**：
$$\tilde{u}_n = \tilde{u}n_x + \tilde{v}n_y + \tilde{w}n_z$$

**原始变量跳跃**：
$$\Delta\rho = \rho_R - \rho_L$$
$$\Delta p = p_R - p_L$$
$$\Delta u_n = (\vec{u}_R - \vec{u}_L) \cdot \vec{n}$$

**切向速度差**：
$$\Delta \vec{u}_t = \Delta \vec{u} - (\Delta u_n)\vec{n}$$

#### 步骤3：计算波强

$$\alpha_1 = \frac{\Delta p - \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$$
$$\alpha_2 = \Delta\rho - \frac{\Delta p}{\tilde{c}^2}$$
$$\alpha_5 = \frac{\Delta p + \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$$

#### 步骤4：构造耗散项

**方法：按波类型逐项累加**

```cpp
// 初始化耗散项
for (int i = 0; i < 5; ++i) diss[i] = 0.0;

// 波1: 左行声波 (u_n - c)
DataType w1 = alpha1 * |lambda1|;
diss[0] += w1 * 1.0;
diss[1] += w1 * (u_tilde - c_tilde * n_x);
diss[2] += w1 * (v_tilde - c_tilde * n_y);
diss[3] += w1 * (w_tilde - c_tilde * n_z);
diss[4] += w1 * (H_tilde - u_n_tilde * c_tilde);

// 波5: 右行声波 (u_n + c)
DataType w5 = alpha5 * |lambda5|;
diss[0] += w5 * 1.0;
diss[1] += w5 * (u_tilde + c_tilde * n_x);
diss[2] += w5 * (v_tilde + c_tilde * n_y);
diss[3] += w5 * (w_tilde + c_tilde * n_z);
diss[4] += w5 * (H_tilde + u_n_tilde * c_tilde);

// 波2: 熵波 (u_n)
DataType w2 = alpha2 * |u_n|;
diss[0] += w2 * 1.0;
diss[1] += w2 * u_tilde;
diss[2] += w2 * v_tilde;
diss[3] += w2 * w_tilde;
diss[4] += w2 * (0.5 * q_tilde^2);

// 剪切波 (u_n)
DataType shear_scale = rho_bar * |u_n|;
for (int i = 0; i < 3; ++i) {
    diss[i+1] += shear_scale * du_t[i];
    diss[4] += shear_scale * (u_tilde[i] * du_t[i]);
}
```

#### 步骤5：计算最终通量

$$F = \frac{1}{2}(F_L + F_R) - \frac{1}{2}\vec{\Phi}$$

---

## 代码实现

### 3D欧拉方程 Roe通量（C++）

```cpp
void computeRiemannFluxROE(const DataType consrv_l[5],
                           const DataType consrv_r[5],
                           DataType flux[5],
                           const DataType normal[3]) const
{
    const DataType gamma = GAMMA;
    const DataType gm1 = gamma - 1.0;

    // --- 1. 提取左右状态的原始变量 ---
    DataType rho_l = consrv_l[0];
    DataType rho_r = consrv_r[0];

    DataType inv_rho_l = 1.0 / rho_l;
    DataType inv_rho_r = 1.0 / rho_r;

    // 速度
    DataType u_l[3], u_r[3];
    for (int i = 0; i < 3; ++i) {
        u_l[i] = consrv_l[i + 1] * inv_rho_l;
        u_r[i] = consrv_r[i + 1] * inv_rho_r;
    }

    // 动能
    DataType u2_l = 0.0, u2_r = 0.0;
    for (int i = 0; i < 3; ++i) {
        u2_l += u_l[i] * u_l[i];
        u2_r += u_r[i] * u_r[i];
    }
    DataType ke_l = 0.5 * rho_l * u2_l;
    DataType ke_r = 0.5 * rho_r * u2_r;

    // 压力 p = (gamma - 1) * (E - ke)
    DataType p_l = gm1 * (consrv_l[4] - ke_l);
    DataType p_r = gm1 * (consrv_r[4] - ke_r);

    // 总焓 H = (E + p) / rho
    DataType h_l = (consrv_l[4] + p_l) * inv_rho_l;
    DataType h_r = (consrv_r[4] + p_r) * inv_rho_r;

    // 法向速度 Un = u . n
    DataType un_l = 0.0, un_r = 0.0;
    for (int i = 0; i < 3; ++i) {
        un_l += u_l[i] * normal[i];
        un_r += u_r[i] * normal[i];
    }

    // --- 2. 计算 Roe 平均量 ---
    DataType sq_rho_l = std::sqrt(rho_l);
    DataType sq_rho_r = std::sqrt(rho_r);
    DataType inv_sum_sq_rho = 1.0 / (sq_rho_l + sq_rho_r);

    // Roe 平均速度
    DataType u_tilde[3];
    for (int i = 0; i < 3; ++i) {
        u_tilde[i] = (sq_rho_l * u_l[i] + sq_rho_r * u_r[i]) * inv_sum_sq_rho;
    }

    // Roe 平均总焓
    DataType h_tilde = (sq_rho_l * h_l + sq_rho_r * h_r) * inv_sum_sq_rho;

    // Roe 平均法向速度
    DataType un_tilde = 0.0;
    for (int i = 0; i < 3; ++i) {
        un_tilde += u_tilde[i] * normal[i];
    }

    // Roe 平均声速 c = sqrt((gamma-1)*(H - 0.5*|u|^2))
    DataType u2_tilde = 0.0;
    for (int i = 0; i < 3; ++i) {
        u2_tilde += u_tilde[i] * u_tilde[i];
    }
    DataType c_tilde = std::sqrt(gm1 * (h_tilde - 0.5 * u2_tilde));

    // Roe 平均密度 (几何平均)
    DataType rho_bar = sq_rho_l * sq_rho_r;

    // --- 3. 计算物理通量 (F_L 和 F_R) ---
    DataType flux_l[5], flux_r[5];

    flux_l[0] = rho_l * un_l;
    flux_r[0] = rho_r * un_r;
    for (int i = 0; i < 3; ++i) {
        flux_l[i + 1] = rho_l * u_l[i] * un_l + p_l * normal[i];
        flux_r[i + 1] = rho_r * u_r[i] * un_r + p_r * normal[i];
    }
    flux_l[4] = (consrv_l[4] + p_l) * un_l;
    flux_r[4] = (consrv_r[4] + p_r) * un_r;

    // --- 4. 特征波分解与耗散项 ---
    // 跳跃量
    DataType drho = rho_r - rho_l;
    DataType dp = p_r - p_l;

    // 速度差
    DataType du_vec[3];
    for (int i = 0; i < 3; ++i) {
        du_vec[i] = u_r[i] - u_l[i];
    }

    // 法向速度差
    DataType dun = 0.0;
    for (int i = 0; i < 3; ++i) {
        dun += du_vec[i] * normal[i];
    }

    // 切向速度差
    DataType du_t[3];
    for (int i = 0; i < 3; ++i) {
        du_t[i] = du_vec[i] - dun * normal[i];
    }

    // 特征值 (波速)
    DataType lambda1 = un_tilde - c_tilde;
    DataType lambda2 = un_tilde;
    DataType lambda5 = un_tilde + c_tilde;

    DataType abs_lambda1 = std::fabs(lambda1);
    DataType abs_lambda2 = std::fabs(lambda2);
    DataType abs_lambda5 = std::fabs(lambda5);

    // 波强
    DataType c2 = c_tilde * c_tilde;
    DataType alpha1 = 0.5 * (dp - rho_bar * c_tilde * dun) / c2;
    DataType alpha5 = 0.5 * (dp + rho_bar * c_tilde * dun) / c2;
    DataType alpha2 = drho - dp / c2;

    // 耗散项
    DataType diss[5] = {0};

    // Wave 1 (u - c)
    DataType w1 = alpha1 * abs_lambda1;
    diss[0] += w1;
    for (int i = 0; i < 3; ++i) {
        diss[i + 1] += w1 * (u_tilde[i] - c_tilde * normal[i]);
    }
    diss[4] += w1 * (h_tilde - un_tilde * c_tilde);

    // Wave 5 (u + c)
    DataType w5 = alpha5 * abs_lambda5;
    diss[0] += w5;
    for (int i = 0; i < 3; ++i) {
        diss[i + 1] += w5 * (u_tilde[i] + c_tilde * normal[i]);
    }
    diss[4] += w5 * (h_tilde + un_tilde * c_tilde);

    // Wave 2 (熵波, u)
    DataType w2 = alpha2 * abs_lambda2;
    diss[0] += w2;
    for (int i = 0; i < 3; ++i) {
        diss[i + 1] += w2 * u_tilde[i];
    }
    diss[4] += w2 * (0.5 * u2_tilde);

    // 剪切波 (Shear waves)
    DataType shear_scale = rho_bar * abs_lambda2;
    for (int i = 0; i < 3; ++i) {
        diss[i + 1] += shear_scale * du_t[i];
        diss[4] += shear_scale * (u_tilde[i] * du_t[i]);
    }

    // --- 5. 最终通量 ---
    for (int i = 0; i < 5; ++i) {
        flux[i] = 0.5 * (flux_l[i] + flux_r[i]) - 0.5 * diss[i];
    }
}
```

### 3D欧拉方程 Roe通量（Python）

```python
import numpy as np

def roe_flux_3d_euler(UL, UR, n, gamma=1.4):
    """
    3D Roe Flux for Euler Equations

    Parameters:
    -----------
    UL, UR : array (5,)
        左右状态守恒变量 [rho, rhou, rhov, rhow, rhoE]
    n : array (3,)
        单位法向量 (nx, ny, nz)
    gamma : float
        比热比

    Returns:
    --------
    flux : array (5,)
        法向数值通量
    """
    gm1 = gamma - 1.0

    # 1. 提取原始变量
    rhoL, rhoR = UL[0], UR[0]
    uL = UL[1:4] / rhoL
    uR = UR[1:4] / rhoR

    u2L, u2R = np.dot(uL, uL), np.dot(uR, uR)
    pL = gm1 * (UL[4] - 0.5 * rhoL * u2L)
    pR = gm1 * (UR[4] - 0.5 * rhoR * u2R)
    hL = (UL[4] + pL) / rhoL
    hR = (UR[4] + pR) / rhoR

    unL, unR = np.dot(uL, n), np.dot(uR, n)

    # 2. Roe 平均
    sqrt_rhoL, sqrt_rhoR = np.sqrt(rhoL), np.sqrt(rhoR)
    denom = sqrt_rhoL + sqrt_rhoR

    u_tilde = (sqrt_rhoL * uL + sqrt_rhoR * uR) / denom
    h_tilde = (sqrt_rhoL * hL + sqrt_rhoR * hR) / denom
    un_tilde = np.dot(u_tilde, n)

    u2_tilde = np.dot(u_tilde, u_tilde)
    c_tilde = np.sqrt(gm1 * (h_tilde - 0.5 * u2_tilde))
    rho_bar = sqrt_rhoL * sqrt_rhoR

    # 3. 物理通量
    FL = np.zeros(5)
    FR = np.zeros(5)

    FL[0] = rhoL * unL
    FR[0] = rhoR * unR
    FL[1:4] = rhoL * np.outer(uL, n).sum(axis=1) * unL + pL * n
    FR[1:4] = rhoR * np.outer(uR, n).sum(axis=1) * unR + pR * n
    FL[4] = (UL[4] + pL) * unL
    FR[4] = (UR[4] + pR) * unR

    # 4. 跳跃量
    drho = rhoR - rhoL
    dp = pR - pL
    du = uR - uL
    dun = np.dot(du, n)
    du_t = du - dun * n  # 切向速度差

    # 5. 波强
    c2 = c_tilde**2
    alpha1 = 0.5 * (dp - rho_bar * c_tilde * dun) / c2
    alpha5 = 0.5 * (dp + rho_bar * c_tilde * dun) / c2
    alpha2 = drho - dp / c2

    # 6. 特征值
    lambda1 = un_tilde - c_tilde
    lambda2 = un_tilde
    lambda5 = un_tilde + c_tilde

    # 7. 耗散项
    diss = np.zeros(5)

    # 左行声波
    w1 = alpha1 * abs(lambda1)
    diss[0] += w1
    diss[1:4] += w1 * (u_tilde - c_tilde * n)
    diss[4] += w1 * (h_tilde - un_tilde * c_tilde)

    # 右行声波
    w5 = alpha5 * abs(lambda5)
    diss[0] += w5
    diss[1:4] += w5 * (u_tilde + c_tilde * n)
    diss[4] += w5 * (h_tilde + un_tilde * c_tilde)

    # 熵波
    w2 = alpha2 * abs(lambda2)
    diss[0] += w2
    diss[1:4] += w2 * u_tilde
    diss[4] += w2 * 0.5 * u2_tilde

    # 剪切波
    shear_scale = rho_bar * abs(lambda2)
    diss[1:4] += shear_scale * du_t
    diss[4] += shear_scale * np.dot(u_tilde, du_t)

    # 8. 最终通量
    return 0.5 * (FL + FR) - 0.5 * diss
```

---

## 3D弱可压缩NS方程的推导

### 1. 变量与模型

- **守恒变量**: $U = [\rho, \rho u, \rho v, \rho w]^T$（4个）
- **原始变量**: $V = [\rho, u, v, w]^T$
- **状态方程**: $p = p_0 + c_s^2(\rho - \rho_0)$

### 2. 特征值

| 编号 | 特征值 | 波类型 |
|------|--------|--------|
| $\lambda_1$ | $u_n - c_s$ | 左行声波 |
| $\lambda_2$ | $u_n + c_s$ | 右行声波 |
| $\lambda_3$ | $u_n$ | 剪切波1 |
| $\lambda_4$ | $u_n$ | 剪切波2 |

**注意**：弱可压缩模型没有熵波，因为压力仅由密度决定。

### 3. 波强与耗散项

**波强**：
$$\alpha_1 = \frac{1}{2}\left(\Delta\rho - \frac{\tilde{\rho}}{c_s}\Delta u_n\right)$$
$$\alpha_2 = \frac{1}{2}\left(\Delta\rho + \frac{\tilde{\rho}}{c_s}\Delta u_n\right)$$

**右特征向量**：
$$\vec{r}_1 = \begin{bmatrix} 1 \\ u - c_s n_x \\ v - c_s n_y \\ w - c_s n_z \end{bmatrix}, \quad \vec{r}_2 = \begin{bmatrix} 1 \\ u + c_s n_x \\ v + c_s n_y \\ w + c_s n_z \end{bmatrix}$$

**剪切波处理**：
剪切波波强为 $\tilde{\rho}\Delta\vec{u}_t$，直接累加到耗散项。

---

## 总结与对比

### 1D vs 3D Roe格式对比

| 特性 | 1D欧拉 | 3D欧拉 | 3D弱可压缩 |
|------|--------|--------|------------|
| 守恒变量数 | 3 | 5 | 4 |
| 特征波数 | 3 | 5 | 4 |
| 声波 | $u \pm c$ | $u_n \pm c$ | $u_n \pm c_s$ |
| 熵波 | 有 ($u$) | 有 ($u_n$) | 无 |
| 剪切波 | 无 | 2个 ($u_n$) | 2个 ($u_n$) |

### 关键公式汇总

**波强计算**（3D欧拉）：

| 波 | 波强公式 | 特征向量 |
|----|----------|----------|
| 左行声波 | $\alpha_1 = \frac{\Delta p - \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$ | $\vec{r}_1 = [1, \vec{u}-\tilde{c}\vec{n}, H-u_n c]^T$ |
| 熵波 | $\alpha_2 = \Delta\rho - \frac{\Delta p}{\tilde{c}^2}$ | $\vec{r}_2 = [1, \vec{u}, \frac{1}{2}q^2]^T$ |
| 剪切波 | $\tilde{\rho}\Delta\vec{u}_t$ | 动量分量: $\Delta\vec{u}_t$，能量分量: $\vec{u}\cdot\Delta\vec{u}_t$ |
| 右行声波 | $\alpha_5 = \frac{\Delta p + \tilde{\rho}\tilde{c}\Delta u_n}{2\tilde{c}^2}$ | $\vec{r}_5 = [1, \vec{u}+\tilde{c}\vec{n}, H+u_n c]^T$ |

### 核心思想

1. **法向投影**：将3D问题投影到法向，退化为准1D问题
2. **剪切波处理**：利用切向速度差直接计算剪切波贡献
3. **原始变量法**：使用原始变量跳跃量计算波强，简洁明了

**数值通量统一形式**：
$$F_{i+1/2} = \frac{1}{2}(F_L + F_R) - \frac{1}{2}\sum_{k}|\lambda_k|\alpha_k\vec{r}_k$$