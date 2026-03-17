# DG 矩限制器 (Moment Limiter) 文档

## 1. 概述

矩限制器 (Moment Limiter) 是间断有限元 (DG) 方法中用于抑制数值振荡、保持解的单调性的关键技术。本文档描述了基于 Cockburn & Shu 方法的矩限制器实现，支持 P1 (K1) 和 P2 (K2) 多项式阶数。

## 2. 理论基础

### 2.1 Legendre 多项式基

DG 解在参考单元 $[-1, 1]$ 上用 Legendre 多项式展开：

$$u(\xi) = u_0 \cdot L_0(\xi) + u_1 \cdot L_1(\xi) + u_2 \cdot L_2(\xi) + \cdots$$

其中 Legendre 多项式定义为：

| 阶数 | $L_k(\xi)$ | 表达式 |
|------|------------|--------|
| 0 | $L_0$ | $1$ |
| 1 | $L_1$ | $\xi$ |
| 2 | $L_2$ | $\frac{3\xi^2 - 1}{2}$ |

### 2.2 矩系数计算

矩系数通过 $L^2$ 投影计算：

$$u_k = \frac{2k+1}{2} \int_{-1}^{1} u(\xi) L_k(\xi) \, d\xi$$

归一化因子：
- $u_0$: 直接等于单元平均值
- $u_1$: 归一化因子 $\frac{3}{2}$ （因为 $\int L_1^2 = \frac{2}{3}$）
- $u_2$: 归一化因子 $\frac{5}{2}$ （因为 $\int L_2^2 = \frac{2}{5}$）

## 3. 算法实现

### 3.1 整体流程

```
对每个单元 iele:
    对每个变量 ivar:
        1. 计算矩系数 (u_0, u_1, u_2)
        2. 限制二阶矩 (仅 P2)
        3. 限制一阶矩
        4. 重构解
```

### 3.2 P1 (K1) 实现

#### 矩系数计算

对于 P1，解在 LGL 点 $\xi = \{-1, 1\}$ 上的值为 $u[0], u[1]$：

$$u[0] = u_0 - u_1, \quad u[1] = u_0 + u_1$$

因此可以直接计算：

$$u_1 = \frac{u[1] - u[0]}{2}$$

#### 一阶矩限制

使用 minmod 函数限制斜率：

$$u_1^{lim} = \text{minmod}(u_1, \bar{u}_R - \bar{u}, \bar{u} - \bar{u}_L)$$

其中 $\bar{u}, \bar{u}_L, \bar{u}_R$ 分别为当前单元和左右邻居的单元平均值。

#### 解的重构

$$u(\xi) = u_0 + u_1^{lim} \cdot \xi$$

### 3.3 P2 (K2) 实现

#### 矩系数计算

对于 P2，解在 LGL 点 $\xi = \{-1, 0, 1\}$ 上：

$$u(\xi_i) = u_0 \cdot 1 + u_1 \cdot \xi_i + u_2 \cdot \frac{3\xi_i^2 - 1}{2}$$

通过数值积分计算矩系数：

```cpp
u_1 = 1.5 * Σ w_i * u[i] * ξ_i
u_2 = 2.5 * Σ w_i * u[i] * (3ξ_i² - 1) / 2
```

#### 二阶矩限制

使用相邻单元的二阶矩作为参考：

$$u_2^{lim} = \text{minmod}(u_2, u_2^L, u_2^R)$$

#### 一阶矩限制

检查重构后的边界值是否产生新极值：

$$u_{left} = u_0 - u_1 + u_2, \quad u_{right} = u_0 + u_1 + u_2$$

如果 $u_{left}$ 或 $u_{right}$ 超出 $[\min(\bar{u}_L, \bar{u}, \bar{u}_R), \max(\bar{u}_L, \bar{u}, \bar{u}_R)]$ 范围，则限制一阶矩并置二阶矩为零。

#### 解的重构

$$u(\xi) = u_0 + u_1 \cdot \xi + u_2 \cdot \frac{3\xi^2 - 1}{2}$$

## 4. Minmod 函数

### 4.1 定义

```cpp
std::pair<DataType, int> Minmod(DataType a, DataType b, DataType c)
{
    // a 接近零，不限制
    if (|a| < ε) return {a, 0};

    // 符号不一致，限制为零
    if (!(同号(a, b, c))) return {0, 1};

    // 返回绝对值最小者，保持符号
    return {sign(a) * min(|a|, |b|, |c|), 激活标志};
}
```

### 4.2 TVD 性质

minmod 函数保证：
1. 当解单调时，保持单调性
2. 当解有局部极值时，斜率限制为零
3. 避免产生新的局部极值

## 5. 代码实现

### 5.1 核心代码结构

```cpp
template <typename ConfigType>
void Solver<ConfigType>::TvdLimiter()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &elem = elem_pool_old[TORDER][iele];
        Element &elem_L = /* 左邻居 */;
        Element &elem_R = /* 右邻居 */;

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            // 1. 计算矩系数
            DataType u_0 = elem.u_avg[ivar];
            DataType u_1, u_2;

            if constexpr (ORDER == 1) {
                u_1 = 0.5 * (u[1] - u[0]);
            } else {
                // L2 投影计算 u_1, u_2
            }

            // 2. 限制二阶矩 (P2)
            if constexpr (ORDER >= 2) {
                auto [u_2_lim, lim2] = Minmod(u_2, u_2_L, u_2_R);
                // ...
            }

            // 3. 限制一阶矩
            // 检查边界值是否产生新极值
            // 使用 minmod 限制

            // 4. 重构解
            if (limited) {
                for (int isp = 0; isp < NSP; isp++) {
                    u_new = u_0 + u_1 * ξ + u_2 * L_2(ξ);
                }
            }
        }
    }
}
```

## 6. LGL 点和权重

### 6.1 P1 (ORDER=1)

| 点 | $\xi$ | 权重 $w$ |
|----|-------|----------|
| 0  | -1    | 1        |
| 1  | +1    | 1        |

### 6.2 P2 (ORDER=2)

| 点 | $\xi$ | 权重 $w$ |
|----|-------|----------|
| 0  | -1    | 1/3      |
| 1  | 0     | 4/3      |
| 2  | +1    | 1/3      |

## 7. 关键公式汇总

| 公式 | 表达式 |
|------|--------|
| P1 斜率 | $u_1 = \frac{u[1] - u[0]}{2}$ |
| P2 一阶矩 | $u_1 = \frac{3}{2} \sum_i w_i u_i \xi_i$ |
| P2 二阶矩 | $u_2 = \frac{5}{2} \sum_i w_i u_i \frac{3\xi_i^2 - 1}{2}$ |
| minmod | $\text{minmod}(a,b,c) = \begin{cases} 0 & \text{符号不同} \\ \text{sign}(a) \cdot \min(|a|,|b|,|c|) & \text{符号相同} \end{cases}$ |
| P1 重构 | $u(\xi) = u_0 + u_1 \xi$ |
| P2 重构 | $u(\xi) = u_0 + u_1 \xi + u_2 \frac{3\xi^2-1}{2}$ |

## 8. 参考文献

1. Cockburn, B., & Shu, C. W. (1998). The Runge-Kutta discontinuous Galerkin method for conservation laws V: multidimensional systems. *Journal of Computational Physics*, 141(2), 199-224.

2. Cockburn, B., Lin, S. Y., & Shu, C. W. (1989). TVB Runge-Kutta local projection discontinuous Galerkin finite element method for conservation laws III: one-dimensional systems. *Journal of Computational Physics*, 84(1), 90-113.

3. Zhang, X., & Shu, C. W. (2010). On maximum-principle-satisfying high order schemes for scalar conservation laws. *Journal of Computational Physics*, 229(9), 3091-3120.