import argparse

import numpy as np
import matplotlib.pyplot as plt

# 默认使用双精度；命令行可通过 --precision float 切换为单精度。
DTYPE = np.float64
_GAUSS_CACHE = {}


def configure_precision(precision):
    """设置计算精度并重新创建所有依赖精度的全局参数。"""
    global DTYPE, lambda_val, y0, t0, T, h_list, log_h

    DTYPE = np.float32 if precision == "float" else np.float64
    lambda_val = DTYPE(-1.0)
    y0 = DTYPE(1.0)
    t0 = DTYPE(0.0)
    T = DTYPE(2.0)
    h_list = np.array(
        [
            0.4,
            0.2,
            0.1,
            0.05,
            0.025,
            0.0125,
            0.00625,
            0.003125,
            0.0015625,
            0.00078125,
            0.000390625,
        ],
        dtype=DTYPE,
    )
    log_h = np.log10(h_list)


configure_precision("double")


# 定义微分方程 dy/dt = λy
def lambda_y(t, y):
    return lambda_val * y


# 理论解 y(t) = exp(λt)，使用当前选择的精度
def exact_solution(t):
    t = np.asarray(t, dtype=DTYPE)
    return np.exp(lambda_val * t, dtype=DTYPE)


# 欧拉向前方法
def euler_forward(f, y0, t0, T, h):
    t = DTYPE(t0)
    y = DTYPE(y0)
    t_seq = [t]
    y_seq = [y]
    n = int(round((T - t0) / h))
    for i in range(n):
        y = y + h * f(t, y)
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
    return y, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


# RK4 方法
def rk4(f, y0, t0, T, h):
    t = DTYPE(t0)
    y = DTYPE(y0)
    t_seq = [t]
    y_seq = [y]
    n = int(round((T - t0) / h))
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + h / DTYPE(2.0), y + k1 * h / DTYPE(2.0))
        k3 = f(t + h / DTYPE(2.0), y + k2 * h / DTYPE(2.0))
        k4 = f(t + h, y + h * k3)
        y = y + h / DTYPE(6.0) * (
            k1 + DTYPE(2.0) * k2 + DTYPE(2.0) * k3 + k4
        )
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
    return y, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


def _integrate_rhs_on_cerk(f, t, un, h, coefficients):
    """计算 h * integral_0^1 RHS(t_n+theta*h, U_CERK(theta)) dtheta。

    coefficients=(c0, c1, ...) 表示
    U_CERK(theta) = un + c0*(theta*h) + c1*(theta*h)**2 + ...。
    积分采用 Gauss-Legendre 求积；节点数比连续多项式次数多一阶，
    使求积误差不限制 CERK 守恒格式的时间精度。
    """
    quadrature_points = len(coefficients) + 1
    cache_key = (DTYPE, quadrature_points)
    if cache_key not in _GAUSS_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        _GAUSS_CACHE[cache_key] = (
            np.asarray(nodes, dtype=DTYPE),
            np.asarray(weights, dtype=DTYPE),
        )
    nodes, weights = _GAUSS_CACHE[cache_key]

    rhs_integral = DTYPE(0.0)
    for node, weight in zip(nodes, weights):
        theta = (node + DTYPE(1.0)) / DTYPE(2.0)
        tau = theta * h
        u_cerk = un
        tau_power = tau
        for coefficient in coefficients:
            u_cerk = u_cerk + coefficient * tau_power
            tau_power *= tau
        rhs_integral = rhs_integral + weight * f(t + tau, u_cerk)

    return un + h * rhs_integral / DTYPE(2.0)


# CERK2
def cerk2(f, y0, t0, T, h, conservative=False):
    b01 = DTYPE(1.0)
    b02 = DTYPE(0.0)
    b11 = DTYPE(-1.0) / DTYPE(2.0)
    b12 = DTYPE(1.0) / DTYPE(2.0)
    a21 = DTYPE(1.0)

    t = DTYPE(t0)
    un = DTYPE(y0)
    t_seq = [t]
    y_seq = [un]
    n = int(round((T - t0) / h))
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        c0 = b01 * k1 + b02 * k2
        c1 = (b11 * k1 + b12 * k2) / h
        if conservative:
            un = _integrate_rhs_on_cerk(f, t, un, h, (c0, c1))
        else:
            un = un + c0 * h + c1 * h * h
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


# CERK4（paper version）
def cerk4(f, y0, t0, T, h, conservative=False):
    b01 = DTYPE(1.0)
    b02 = DTYPE(0.0)
    b03 = DTYPE(0.0)
    b04 = DTYPE(0.0)
    b11 = DTYPE(-65.0) / DTYPE(48.0)
    b12 = DTYPE(529.0) / DTYPE(384.0)
    b13 = DTYPE(125.0) / DTYPE(128.0)
    b14 = DTYPE(-1.0)
    b21 = DTYPE(41.0) / DTYPE(72.0)
    b22 = DTYPE(-529.0) / DTYPE(576.0)
    b23 = DTYPE(-125.0) / DTYPE(192.0)
    b24 = DTYPE(1.0)
    a21 = DTYPE(12.0) / DTYPE(23.0)
    a31 = DTYPE(-68.0) / DTYPE(375.0)
    a32 = DTYPE(368.0) / DTYPE(375.0)
    a41 = DTYPE(31.0) / DTYPE(144.0)
    a42 = DTYPE(529.0) / DTYPE(1152.0)
    a43 = DTYPE(125.0) / DTYPE(384.0)

    t = DTYPE(t0)
    un = DTYPE(y0)
    t_seq = [t]
    y_seq = [un]
    n = int(round((T - t0) / h))
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        v3 = un + (a31 * k1 + a32 * k2) * h
        k3 = f(t, v3)
        v4 = un + (a41 * k1 + a42 * k2 + a43 * k3) * h
        k4 = f(t, v4)
        c0 = b01 * k1 + b02 * k2 + b03 * k3 + b04 * k4
        c1 = (b11 * k1 + b12 * k2 + b13 * k3 + b14 * k4) / h
        c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4) / (h * h)
        if conservative:
            un = _integrate_rhs_on_cerk(f, t, un, h, (c0, c1, c2))
        else:
            un = un + c0 * h + c1 * h * h + c2 * h * h * h
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


# CERK6（paper version）
def cerk6(f, y0, t0, T, h, conservative=False):
    b01 = DTYPE(1.0)
    b02 = DTYPE(0.0)
    b03 = DTYPE(0.0)
    b04 = DTYPE(0.0)
    b05 = DTYPE(0.0)
    b06 = DTYPE(0.0)
    b11 = DTYPE(-104217.0) / DTYPE(37466.0)
    b12 = DTYPE(0.0)
    b13 = DTYPE(861101.0) / DTYPE(230560.0)
    b14 = DTYPE(-63869.0) / DTYPE(293440.0)
    b15 = DTYPE(-1522125.0) / DTYPE(762944.0)
    b16 = DTYPE(165.0) / DTYPE(131.0)
    b21 = DTYPE(1806901.0) / DTYPE(618189.0)
    b22 = DTYPE(0.0)
    b23 = DTYPE(-2178079.0) / DTYPE(380424.0)
    b24 = DTYPE(6244423.0) / DTYPE(5325936.0)
    b25 = DTYPE(982125.0) / DTYPE(190736.0)
    b26 = DTYPE(-461.0) / DTYPE(131.0)
    b31 = DTYPE(-866577.0) / DTYPE(824252.0)
    b32 = DTYPE(0.0)
    b33 = DTYPE(12308679.0) / DTYPE(5072320.0)
    b34 = DTYPE(-7816583.0) / DTYPE(10144640.0)
    b35 = DTYPE(-624375.0) / DTYPE(217984.0)
    b36 = DTYPE(296.0) / DTYPE(131.0)
    a21 = DTYPE(1.0) / DTYPE(6.0)
    a31 = DTYPE(44.0) / DTYPE(1369.0)
    a32 = DTYPE(363.0) / DTYPE(1369.0)
    a41 = DTYPE(3388.0) / DTYPE(4913.0)
    a42 = DTYPE(-8349.0) / DTYPE(4913.0)
    a43 = DTYPE(8140.0) / DTYPE(4913.0)
    a51 = DTYPE(-36764.0) / DTYPE(408375.0)
    a52 = DTYPE(767.0) / DTYPE(1125.0)
    a53 = DTYPE(-32708.0) / DTYPE(136125.0)
    a54 = DTYPE(210392.0) / DTYPE(408375.0)
    a61 = DTYPE(1697.0) / DTYPE(18876.0)
    a62 = DTYPE(0.0)
    a63 = DTYPE(50653.0) / DTYPE(116160.0)
    a64 = DTYPE(299693.0) / DTYPE(1626240.0)
    a65 = DTYPE(3375.0) / DTYPE(11648.0)

    t = DTYPE(t0)
    un = DTYPE(y0)
    t_seq = [t]
    y_seq = [un]
    n = int(round((T - t0) / h))
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        v3 = un + (a31 * k1 + a32 * k2) * h
        k3 = f(t, v3)
        v4 = un + (a41 * k1 + a42 * k2 + a43 * k3) * h
        k4 = f(t, v4)
        v5 = un + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h
        k5 = f(t, v5)
        v6 = un + (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5) * h
        k6 = f(t, v6)
        c0 = b01 * k1 + b02 * k2 + b03 * k3 + b04 * k4 + b05 * k5 + b06 * k6
        c1 = (b11 * k1 + b12 * k2 + b13 * k3 + b14 * k4 + b15 * k5 + b16 * k6) / h
        c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4 + b25 * k5 + b26 * k6) / (h * h)
        c3 = (b31 * k1 + b32 * k2 + b33 * k3 + b34 * k4 + b35 * k5 + b36 * k6) / (
            h * h * h
        )
        if conservative:
            un = _integrate_rhs_on_cerk(f, t, un, h, (c0, c1, c2, c3))
        else:
            un = un + c0 * h + c1 * h * h + c2 * h * h * h + c3 * h * h * h * h
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


# CERK8（paper version）
def cerk8(f, y0, t0, T, h, conservative=False):
    # 系数统一转换为当前选择的精度
    b01 = DTYPE(1.0)
    b02 = DTYPE(0.0)
    b03 = DTYPE(0.0)
    b04 = DTYPE(0.0)
    b05 = DTYPE(0.0)
    b06 = DTYPE(0.0)
    b07 = DTYPE(0.0)
    b08 = DTYPE(0.0)

    b11 = DTYPE(-3292.0) / DTYPE(819.0)
    b12 = DTYPE(0.0)
    b13 = DTYPE(5112.0) / DTYPE(715.0)
    b14 = DTYPE(-123.0) / DTYPE(52.0)
    b15 = DTYPE(-63.0) / DTYPE(52.0)
    b16 = DTYPE(-40817.0) / DTYPE(33462.0)
    b17 = DTYPE(18048.0) / DTYPE(5915.0)
    b18 = DTYPE(-18.0) / DTYPE(13.0)

    b21 = DTYPE(17893.0) / DTYPE(2457.0)
    b22 = DTYPE(0.0)
    b23 = DTYPE(-43568.0) / DTYPE(2145.0)
    b24 = DTYPE(3161.0) / DTYPE(234.0)
    b25 = DTYPE(1061.0) / DTYPE(234.0)
    b26 = DTYPE(60025.0) / DTYPE(50193.0)
    b27 = DTYPE(-637696.0) / DTYPE(53235.0)
    b28 = DTYPE(75.0) / DTYPE(13.0)

    b31 = DTYPE(-4969.0) / DTYPE(819.0)
    b32 = DTYPE(0.0)
    b33 = DTYPE(1344.0) / DTYPE(65.0)
    b34 = DTYPE(-1465.0) / DTYPE(78.0)
    b35 = DTYPE(-413.0) / DTYPE(78.0)
    b36 = DTYPE(2401.0) / DTYPE(1521.0)
    b37 = DTYPE(96256.0) / DTYPE(5915.0)
    b38 = DTYPE(-109.0) / DTYPE(13.0)

    b41 = DTYPE(596.0) / DTYPE(315.0)
    b42 = DTYPE(0.0)
    b43 = DTYPE(-1984.0) / DTYPE(275.0)
    b44 = DTYPE(118.0) / DTYPE(15.0)
    b45 = DTYPE(2.0)
    b46 = DTYPE(-9604.0) / DTYPE(6435.0)
    b47 = DTYPE(-48128.0) / DTYPE(6825.0)
    b48 = DTYPE(4.0)

    a21 = DTYPE(1.0) / DTYPE(6.0)
    a31 = DTYPE(1.0) / DTYPE(16.0)
    a32 = DTYPE(3.0) / DTYPE(16.0)
    a41 = DTYPE(1.0) / DTYPE(4.0)
    a42 = DTYPE(-3.0) / DTYPE(4.0)
    a43 = DTYPE(1.0)
    a51 = DTYPE(-3.0) / DTYPE(4.0)
    a52 = DTYPE(15.0) / DTYPE(4.0)
    a53 = DTYPE(-3.0)
    a54 = DTYPE(1.0) / DTYPE(2.0)
    a61 = DTYPE(369.0) / DTYPE(1372.0)
    a62 = DTYPE(-243.0) / DTYPE(343.0)
    a63 = DTYPE(297.0) / DTYPE(343.0)
    a64 = DTYPE(1485.0) / DTYPE(9604.0)
    a65 = DTYPE(297.0) / DTYPE(4802.0)
    a71 = DTYPE(-133.0) / DTYPE(4512.0)
    a72 = DTYPE(1113.0) / DTYPE(6016.0)
    a73 = DTYPE(7945.0) / DTYPE(16544.0)
    a74 = DTYPE(-12845.0) / DTYPE(24064.0)
    a75 = DTYPE(-315.0) / DTYPE(24064.0)
    a76 = DTYPE(156065.0) / DTYPE(198528.0)
    a81 = DTYPE(83.0) / DTYPE(945.0)
    a82 = DTYPE(0.0)
    a83 = DTYPE(248.0) / DTYPE(825.0)
    a84 = DTYPE(41.0) / DTYPE(180.0)
    a85 = DTYPE(1.0) / DTYPE(36.0)
    a86 = DTYPE(2401.0) / DTYPE(38610.0)
    a87 = DTYPE(6016.0) / DTYPE(20475.0)

    t = DTYPE(t0)
    un = DTYPE(y0)
    t_seq = [t]
    y_seq = [un]
    n = int(round((T - t0) / h))
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        v3 = un + (a31 * k1 + a32 * k2) * h
        k3 = f(t, v3)
        v4 = un + (a41 * k1 + a42 * k2 + a43 * k3) * h
        k4 = f(t, v4)
        v5 = un + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h
        k5 = f(t, v5)
        v6 = un + (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5) * h
        k6 = f(t, v6)
        v7 = un + (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6) * h
        k7 = f(t, v7)
        v8 = (
            un
            + (
                a81 * k1
                + a82 * k2
                + a83 * k3
                + a84 * k4
                + a85 * k5
                + a86 * k6
                + a87 * k7
            )
            * h
        )
        k8 = f(t, v8)

        c0 = (
            b01 * k1
            + b02 * k2
            + b03 * k3
            + b04 * k4
            + b05 * k5
            + b06 * k6
            + b07 * k7
            + b08 * k8
        )
        c1 = (
            b11 * k1
            + b12 * k2
            + b13 * k3
            + b14 * k4
            + b15 * k5
            + b16 * k6
            + b17 * k7
            + b18 * k8
        ) / h
        c2 = (
            b21 * k1
            + b22 * k2
            + b23 * k3
            + b24 * k4
            + b25 * k5
            + b26 * k6
            + b27 * k7
            + b28 * k8
        ) / (h * h)
        c3 = (
            b31 * k1
            + b32 * k2
            + b33 * k3
            + b34 * k4
            + b35 * k5
            + b36 * k6
            + b37 * k7
            + b38 * k8
        ) / (h * h * h)
        c4 = (
            b41 * k1
            + b42 * k2
            + b43 * k3
            + b44 * k4
            + b45 * k5
            + b46 * k6
            + b47 * k7
            + b48 * k8
        ) / (h * h * h * h)

        if conservative:
            un = _integrate_rhs_on_cerk(f, t, un, h, (c0, c1, c2, c3, c4))
        else:
            un = (
                un
                + c0 * h
                + c1 * h * h
                + c2 * h * h * h
                + c3 * h * h * h * h
                + c4 * h * h * h * h * h
            )
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=DTYPE), np.array(y_seq, dtype=DTYPE)


def cerk2_cons(f, y0, t0, T, h):
    return cerk2(f, y0, t0, T, h, conservative=True)


def cerk4_cons(f, y0, t0, T, h):
    return cerk4(f, y0, t0, T, h, conservative=True)


def cerk6_cons(f, y0, t0, T, h):
    return cerk6(f, y0, t0, T, h, conservative=True)


def cerk8_cons(f, y0, t0, T, h):
    return cerk8(f, y0, t0, T, h, conservative=True)


# 时间阶分析
def time_order_analysis(time_scheme):
    errors = []
    u_seq_all = []
    t_seq_all = []
    for h in h_list:
        y_num, t_seq, u_seq = time_scheme(lambda_y, y0, t0, T, h)
        t_seq_all.append(t_seq)
        u_seq_all.append(u_seq)
        y_exact = exact_solution(T)
        error = np.abs(y_num - y_exact)
        errors.append(error)

    error_array = np.array(errors, dtype=DTYPE)
    log_errors = np.full(error_array.shape, np.nan, dtype=DTYPE)
    positive = error_array > 0
    log_errors[positive] = np.log10(error_array[positive])
    finite = np.isfinite(log_errors)
    coefficients = np.polyfit(log_h[finite], log_errors[finite], 1)

    # 打印收敛阶
    print(f"h = {h_list[0]:.6f}, error = {errors[0]:.6e}")
    for i in range(1, len(h_list)):
        if finite[i] and finite[i - 1]:
            order = (log_errors[i] - log_errors[i - 1]) / (log_h[i] - log_h[i - 1])
            order_text = f"{order:.4f}"
        else:
            order_text = "precision limit"
        print(f"h = {h_list[i]:.6f}, error = {errors[i]:.6e}, order ≈ {order_text}")

    return errors, log_errors, coefficients, t_seq_all, u_seq_all


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CERK 时间阶与精度分析")
    parser.add_argument(
        "--precision",
        choices=("float", "double"),
        default="double",
        help="计算精度：float=float32，double=float64（默认）",
    )
    args = parser.parse_args()
    configure_precision(args.precision)

    print(f"Precision: {args.precision} ({DTYPE.__name__})")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    schemes = [
        (euler_forward, "Euler forward", "ko-", "k-"),
        (rk4, "RK4", "ro-", "r-"),
        (cerk2, "CERK2", "bo-", "b-"),
        (cerk2_cons, "CERK2-cons", "b^--", "b--"),
        (cerk4, "CERK4", "go--", "g--"),
        (cerk4_cons, "CERK4-cons", "g^:", "g:"),
        (cerk6, "CERK6", "mo-.", "m-."),
        (cerk6_cons, "CERK6-cons", "m^--", "m--"),
        (cerk8, "CERK8", "co:", "c:"),
        (cerk8_cons, "CERK8-cons", "c^--", "c--"),
    ]

    for method, name, marker_style, line_style in schemes:
        print(f"\n{name}:")
        errors, log_errors, coeff, t_seqs, u_seqs = time_order_analysis(method)
        slope = coeff[0]
        ax1.plot(
            log_h,
            log_errors,
            marker_style,
            label=f"{name} (slope={slope:.2f})",
            markersize=6,
        )
        # 绘制最粗网格的解（第一个 h）
        ax2.plot(t_seqs[0], u_seqs[0], line_style, label=f"{name}")

    # 精确解（用于对比）
    t_fine = np.linspace(t0, T, 200, dtype=DTYPE)
    y_exact_fine = exact_solution(t_fine)
    ax2.plot(t_fine, y_exact_fine, "k:", linewidth=1.5, label="Exact")

    ax1.set_xlabel("log₁₀(h)")
    ax1.set_ylabel("log₁₀(error)")
    ax1.set_title("Convergence (log-log)")
    ax1.grid(True, ls="--", alpha=0.7)

    ax2.set_xlabel("t")
    ax2.set_ylabel("y(t)")
    ax2.set_title(f"Numerical Solutions (h={h_list[0]:g})")
    ax2.grid(True, ls="--", alpha=0.7)

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    output_file = f"time_order_cerk_{args.precision}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
