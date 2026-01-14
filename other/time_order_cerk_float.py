import numpy as np
import matplotlib.pyplot as plt

# 定义参数（全部转为 float32）
lambda_val = np.float32(-1.0)  # λ值
y0 = np.float32(1.0)  # 初始条件 y(0)=1
t0 = np.float32(0.0)  # 初始时间
T = np.float32(2.0)  # 评估时间点

h_list = [
    np.float32(0.4),
    np.float32(0.2),
    np.float32(0.1),
    np.float32(0.05),
    np.float32(0.025),
    np.float32(0.0125),
    np.float32(0.00625),
    np.float32(0.003125),
    np.float32(0.0015625),
    np.float32(0.00078125),
    np.float32(0.000390625),
]  # 步长列表（float32）

log_h = np.log10(np.array(h_list, dtype=np.float32))


# 定义微分方程 dy/dt = λy
def lambda_y(t, y):
    return lambda_val * y


# 理论解 y(t) = exp(λt)，使用 float32
def exact_solution(t):
    t = np.asarray(t, dtype=np.float32)
    return np.exp(lambda_val * t, dtype=np.float32)


# 欧拉向前方法
def euler_forward(f, y0, t0, T, h):
    t = np.float32(t0)
    y = np.float32(y0)
    t_seq = [t]
    y_seq = [y]
    n = int(round((T - t0) / h))
    for i in range(n):
        y = y + h * f(t, y)
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
    return y, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# RK4 方法
def rk4(f, y0, t0, T, h):
    t = np.float32(t0)
    y = np.float32(y0)
    t_seq = [t]
    y_seq = [y]
    n = int(round((T - t0) / h))
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + h / np.float32(2.0), y + k1 * h / np.float32(2.0))
        k3 = f(t + h / np.float32(2.0), y + k2 * h / np.float32(2.0))
        k4 = f(t + h, y + h * k3)
        y = y + h / np.float32(6.0) * (
            k1 + np.float32(2.0) * k2 + np.float32(2.0) * k3 + k4
        )
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
    return y, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# CERK2
def cerk2(f, y0, t0, T, h):
    b01 = np.float32(1.0)
    b02 = np.float32(0.0)
    b11 = np.float32(-1.0) / np.float32(2.0)
    b12 = np.float32(1.0) / np.float32(2.0)
    a21 = np.float32(1.0)

    t = np.float32(t0)
    un = np.float32(y0)
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
        un = un + c0 * h + c1 * h * h
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# CERK4（paper version）
def cerk4(f, y0, t0, T, h):
    b01 = np.float32(1.0)
    b02 = np.float32(0.0)
    b03 = np.float32(0.0)
    b04 = np.float32(0.0)
    b11 = np.float32(-65.0) / np.float32(48.0)
    b12 = np.float32(529.0) / np.float32(384.0)
    b13 = np.float32(125.0) / np.float32(128.0)
    b14 = np.float32(-1.0)
    b21 = np.float32(41.0) / np.float32(72.0)
    b22 = np.float32(-529.0) / np.float32(576.0)
    b23 = np.float32(-125.0) / np.float32(192.0)
    b24 = np.float32(1.0)
    a21 = np.float32(12.0) / np.float32(23.0)
    a31 = np.float32(-68.0) / np.float32(375.0)
    a32 = np.float32(368.0) / np.float32(375.0)
    a41 = np.float32(31.0) / np.float32(144.0)
    a42 = np.float32(529.0) / np.float32(1152.0)
    a43 = np.float32(125.0) / np.float32(384.0)

    t = np.float32(t0)
    un = np.float32(y0)
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
        un = un + h * (
            (b01 + b11 + b21) * k1
            + (b02 + b12 + b22) * k2
            + (b03 + b13 + b23) * k3
            + (b04 + b14 + b24) * k4
        )
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# CERK6（paper version）
def cerk6(f, y0, t0, T, h):
    b01 = np.float32(1.0)
    b02 = np.float32(0.0)
    b03 = np.float32(0.0)
    b04 = np.float32(0.0)
    b05 = np.float32(0.0)
    b06 = np.float32(0.0)
    b11 = np.float32(-104217.0) / np.float32(37466.0)
    b12 = np.float32(0.0)
    b13 = np.float32(861101.0) / np.float32(230560.0)
    b14 = np.float32(-63869.0) / np.float32(293440.0)
    b15 = np.float32(-1522125.0) / np.float32(762944.0)
    b16 = np.float32(165.0) / np.float32(131.0)
    b21 = np.float32(1806901.0) / np.float32(618189.0)
    b22 = np.float32(0.0)
    b23 = np.float32(-2178079.0) / np.float32(380424.0)
    b24 = np.float32(6244423.0) / np.float32(5325936.0)
    b25 = np.float32(982125.0) / np.float32(190736.0)
    b26 = np.float32(-461.0) / np.float32(131.0)
    b31 = np.float32(-866577.0) / np.float32(824252.0)
    b32 = np.float32(0.0)
    b33 = np.float32(12308679.0) / np.float32(5072320.0)
    b34 = np.float32(-7816583.0) / np.float32(10144640.0)
    b35 = np.float32(-624375.0) / np.float32(217984.0)
    b36 = np.float32(296.0) / np.float32(131.0)
    a21 = np.float32(1.0) / np.float32(6.0)
    a31 = np.float32(44.0) / np.float32(1369.0)
    a32 = np.float32(363.0) / np.float32(1369.0)
    a41 = np.float32(3388.0) / np.float32(4913.0)
    a42 = np.float32(-8349.0) / np.float32(4913.0)
    a43 = np.float32(8140.0) / np.float32(4913.0)
    a51 = np.float32(-36764.0) / np.float32(408375.0)
    a52 = np.float32(767.0) / np.float32(1125.0)
    a53 = np.float32(-32708.0) / np.float32(136125.0)
    a54 = np.float32(210392.0) / np.float32(408375.0)
    a61 = np.float32(1697.0) / np.float32(18876.0)
    a62 = np.float32(0.0)
    a63 = np.float32(50653.0) / np.float32(116160.0)
    a64 = np.float32(299693.0) / np.float32(1626240.0)
    a65 = np.float32(3375.0) / np.float32(11648.0)

    t = np.float32(t0)
    un = np.float32(y0)
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
        un = un + c0 * h + c1 * h * h + c2 * h * h * h + c3 * h * h * h * h
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# CERK8（paper version）
def cerk8(f, y0, t0, T, h):
    # Coefficients as float32
    b01 = np.float32(1.0)
    b02 = np.float32(0.0)
    b03 = np.float32(0.0)
    b04 = np.float32(0.0)
    b05 = np.float32(0.0)
    b06 = np.float32(0.0)
    b07 = np.float32(0.0)
    b08 = np.float32(0.0)

    b11 = np.float32(-3292.0) / np.float32(819.0)
    b12 = np.float32(0.0)
    b13 = np.float32(5112.0) / np.float32(715.0)
    b14 = np.float32(-123.0) / np.float32(52.0)
    b15 = np.float32(-63.0) / np.float32(52.0)
    b16 = np.float32(-40817.0) / np.float32(33462.0)
    b17 = np.float32(18048.0) / np.float32(5915.0)
    b18 = np.float32(-18.0) / np.float32(13.0)

    b21 = np.float32(17893.0) / np.float32(2457.0)
    b22 = np.float32(0.0)
    b23 = np.float32(-43568.0) / np.float32(2145.0)
    b24 = np.float32(3161.0) / np.float32(234.0)
    b25 = np.float32(1061.0) / np.float32(234.0)
    b26 = np.float32(60025.0) / np.float32(50193.0)
    b27 = np.float32(-637696.0) / np.float32(53235.0)
    b28 = np.float32(75.0) / np.float32(13.0)

    b31 = np.float32(-4969.0) / np.float32(819.0)
    b32 = np.float32(0.0)
    b33 = np.float32(1344.0) / np.float32(65.0)
    b34 = np.float32(-1465.0) / np.float32(78.0)
    b35 = np.float32(-413.0) / np.float32(78.0)
    b36 = np.float32(2401.0) / np.float32(1521.0)
    b37 = np.float32(96256.0) / np.float32(5915.0)
    b38 = np.float32(-109.0) / np.float32(13.0)

    b41 = np.float32(596.0) / np.float32(315.0)
    b42 = np.float32(0.0)
    b43 = np.float32(-1984.0) / np.float32(275.0)
    b44 = np.float32(118.0) / np.float32(15.0)
    b45 = np.float32(2.0)
    b46 = np.float32(-9604.0) / np.float32(6435.0)
    b47 = np.float32(-48128.0) / np.float32(6825.0)
    b48 = np.float32(4.0)

    a21 = np.float32(1.0) / np.float32(6.0)
    a31 = np.float32(1.0) / np.float32(16.0)
    a32 = np.float32(3.0) / np.float32(16.0)
    a41 = np.float32(1.0) / np.float32(4.0)
    a42 = np.float32(-3.0) / np.float32(4.0)
    a43 = np.float32(1.0)
    a51 = np.float32(-3.0) / np.float32(4.0)
    a52 = np.float32(15.0) / np.float32(4.0)
    a53 = np.float32(-3.0)
    a54 = np.float32(1.0) / np.float32(2.0)
    a61 = np.float32(369.0) / np.float32(1372.0)
    a62 = np.float32(-243.0) / np.float32(343.0)
    a63 = np.float32(297.0) / np.float32(343.0)
    a64 = np.float32(1485.0) / np.float32(9604.0)
    a65 = np.float32(297.0) / np.float32(4802.0)
    a71 = np.float32(-133.0) / np.float32(4512.0)
    a72 = np.float32(1113.0) / np.float32(6016.0)
    a73 = np.float32(7945.0) / np.float32(16544.0)
    a74 = np.float32(-12845.0) / np.float32(24064.0)
    a75 = np.float32(-315.0) / np.float32(24064.0)
    a76 = np.float32(156065.0) / np.float32(198528.0)
    a81 = np.float32(83.0) / np.float32(945.0)
    a82 = np.float32(0.0)
    a83 = np.float32(248.0) / np.float32(825.0)
    a84 = np.float32(41.0) / np.float32(180.0)
    a85 = np.float32(1.0) / np.float32(36.0)
    a86 = np.float32(2401.0) / np.float32(38610.0)
    a87 = np.float32(6016.0) / np.float32(20475.0)

    t = np.float32(t0)
    un = np.float32(y0)
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
    return un, np.array(t_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


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

    log_errors = np.log10(np.array(errors, dtype=np.float32))
    coefficients = np.polyfit(log_h, log_errors, 1)

    # 打印收敛阶
    print(f"h = {h_list[0]:.6f}, error = {errors[0]:.6e}")
    for i in range(1, len(h_list)):
        order = (log_errors[i] - log_errors[i - 1]) / (log_h[i] - log_h[i - 1])
        print(f"h = {h_list[i]:.6f}, error = {errors[i]:.6e}, order ≈ {order:.4f}")

    return errors, log_errors, coefficients, t_seq_all, u_seq_all


# 主程序
if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    schemes = [
        (euler_forward, "Euler forward", "ko-", "k-"),
        (rk4, "RK4", "ro-", "r-"),
        (cerk2, "CERK2", "bo-", "b-"),
        (cerk4, "CERK4", "go--", "g--"),
        (cerk6, "CERK6", "mo-.", "m-."),
        (cerk8, "CERK8", "co:", "c:"),
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
    t_fine = np.linspace(t0, T, 200, dtype=np.float32)
    y_exact_fine = exact_solution(t_fine)
    ax2.plot(t_fine, y_exact_fine, "k:", linewidth=1.5, label="Exact")

    ax1.set_xlabel("log₁₀(h)")
    ax1.set_ylabel("log₁₀(error)")
    ax1.set_title("Convergence (log-log)")
    ax1.grid(True, ls="--", alpha=0.7)

    ax2.set_xlabel("t")
    ax2.set_ylabel("y(t)")
    ax2.set_title("Numerical Solutions (h=0.1)")
    ax2.grid(True, ls="--", alpha=0.7)

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig("time_order_cerk_float32.png", dpi=300, bbox_inches="tight")
