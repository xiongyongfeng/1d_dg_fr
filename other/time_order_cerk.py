import numpy as np
import matplotlib.pyplot as plt

# 定义参数
lambda_val = -2.01  # λ值
y0 = 1.0  # 初始条件 y(0)=1
t0 = 0.0  # 初始时间
T = 1.0  # 评估时间点
h_list = [0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # 步长列表


# 定义微分方程 dy/dt = λy
def lambda_y(t, y):
    return lambda_val * y


# 理论解 y(t) = exp(λt)
def exact_solution(t):
    return np.exp(lambda_val * t)


# 欧拉向前方法计算在时间T的数值解
def euler_forward(f, y0, t0, T, h):
    """
    使用欧拉向前方法计算在时间T的数值解
    :param f: 函数f(t, y)
    :param y0: 初始值
    :param t0: 初始时间
    :param T: 目标时间
    :param h: 步长
    :return: 在时间T的数值解
    """
    t = t0
    y = y0
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(y)
    n = int(round((T - t0) / h))  # 计算步数
    for i in range(n):
        y = y + h * f(t, y)
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
        # if h == 0.025:
        #     print(f"t={t},y(t)={y}")
    return y, t_seq, y_seq


def rk4(f, y0, t0, T, h):

    t = t0
    y = y0
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(y)
    n = int(round((T - t0) / h))  # 计算步数
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + k1 * h / 2)
        k3 = f(t + h / 2, y + k2 * h / 2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
        t_seq.append(t)
        y_seq.append(y)
    return y, t_seq, y_seq


def cerk2(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b11 = -1 / 2
    b12 = 1 / 2
    a21 = 1

    t = t0
    un = y0
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
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
    return un, t_seq, y_seq


def cerk4(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b11 = -65 / 48
    b12 = 529 / 384
    b13 = 125 / 128
    b14 = -1
    b21 = 41 / 72
    b22 = -529 / 576
    b23 = -125 / 192
    b24 = 1
    a21 = 12 / 23
    a31 = -68 / 375
    a32 = 368 / 375
    a41 = 31 / 144
    a42 = 529 / 1152  # dimaxer 529/1154  #paper 529/1152
    a43 = 125 / 384

    t = t0
    un = y0
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        v3 = un + (a31 * k1 + a32 * k2) * h
        k3 = f(t, v3)
        v4 = un + (a41 * k1 + a42 * k2 + a43 * k3) * h
        k4 = f(t, v4)
        # c0 = b01 * k1 + b02 * k2 + b03 * k3 + b04 * k4
        # c1 = (b11 * k1 + b12 * k2 + b13 * k3 + b14 * k4) / h
        # c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4) / h / h

        # un = un + c0 * h + c1 * h * h + c2 * h * h * h
        # 等价于
        un = un + h * (
            (b01 + b11 + b21) * k1
            + (b02 + b12 + b22) * k2
            + (b03 + b13 + b23) * k3
            + (b04 + b14 + b24) * k4
        )  # paper
        # un = un + 1 / 24 * k1 + 23 / 24 * k2  # dimaxer
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
        # if h == 0.025:
        #     print(f"t={t},y(t)={un}")
    return un, t_seq, y_seq


def cerk4_dimaxer(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b11 = -65 / 48
    b12 = 529 / 384
    b13 = 125 / 128
    b14 = -1
    b21 = 41 / 72
    b22 = -529 / 576
    b23 = -125 / 192
    b24 = 1
    a21 = 12 / 23
    a31 = -68 / 375
    a32 = 368 / 375
    a41 = 31 / 144
    a42 = 529 / 1154  # dimaxer 529/1154  #paper 529/1152
    a43 = 125 / 384

    t = t0
    un = y0
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
    for i in range(n):
        v1 = un
        k1 = f(t, v1)
        v2 = un + a21 * k1 * h
        k2 = f(t, v2)
        v3 = un + (a31 * k1 + a32 * k2) * h
        k3 = f(t, v3)
        v4 = un + (a41 * k1 + a42 * k2 + a43 * k3) * h
        k4 = f(t, v4)
        # c0 = b01 * k1 + b02 * k2 + b03 * k3 + b04 * k4
        # c1 = (b11 * k1 + b12 * k2 + b13 * k3 + b14 * k4) / h
        # c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4) / h / h

        # un = un + c0 * h + c1 * h * h + c2 * h * h * h
        # 等价于
        # un = un + h * (
        #     (b01 + b11 + b21) * k1
        #     + (b02 + b12 + b22) * k2
        #     + (b03 + b13 + b23) * k3
        #     + (b04 + b14 + b24) * k4
        # )  # paper
        un = un + 1 / 24 * k1 + 23 / 24 * k2  # dimaxer
        t = t + h
        t_seq.append(t)
        y_seq.append(un)
        # if h == 0.025:
        #     print(f"t={t},y(t)={un}")
    return un, t_seq, y_seq


def cerk6(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b05 = 0
    b06 = 0
    b11 = -104217 / 37466
    b12 = 0
    b13 = 861101 / 230560
    b14 = -63869 / 293440
    b15 = -1522125 / 762944
    b16 = 165 / 131
    b21 = 1806901 / 618189
    b22 = 0
    b23 = -2178079 / 380424
    b24 = 6244423 / 5325936
    b25 = 982125 / 190736
    b26 = -461 / 131
    b31 = -866577 / 824252
    b32 = 0
    b33 = 12308679 / 5072320
    b34 = -7816583 / 10144640
    b35 = -624375 / 217984
    b36 = 296 / 131
    a21 = 1 / 6
    a31 = 44 / 1369
    a32 = 363 / 1369
    a41 = 3388 / 4913
    a42 = -8349 / 4913
    a43 = 8140 / 4913
    a51 = -36764 / 408375
    a52 = 767 / 1125
    a53 = -32708 / 136125
    a54 = 210392 / 408375
    a61 = 1697 / 18876  # dimaxer 1697 / 18876   #paper -1697 / 18876
    a62 = 0
    a63 = 50653 / 116160
    a64 = 299693 / 1626240
    a65 = 3375 / 11648

    print(f"(b01 + b11 + b21 + b31)={b01 + b11 + b21 + b31}")
    print(f"(b02 + b12 + b22 + b32)={b02 + b12 + b22 + b32}")
    print(f"(b03 + b13 + b23 + b33)={b03 + b13 + b23 + b33}")
    print(f"(b04 + b14 + b24 + b34)={b04 + b14 + b24 + b34}")
    print(f"(b05 + b15 + b25 + b35)={b05 + b15 + b25 + b35}")
    print(f"(b06 + b16 + b26 + b36)={b06 + b16 + b26 + b36}")

    t_seq = []
    y_seq = []

    t = t0
    un = y0
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
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
        c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4 + b25 * k5 + b26 * k6) / h / h
        c3 = (
            (b31 * k1 + b32 * k2 + b33 * k3 + b34 * k4 + b35 * k5 + b36 * k6)
            / h
            / h
            / h
        )

        un = un + c0 * h + c1 * h * h + c2 * h * h * h + c3 * h * h * h * h
        # 等价于
        # un = un + h * (
        #     (b01 + b11 + b21 + b31) * k1
        #     + (b02 + b12 + b22 + b32) * k2
        #     + (b03 + b13 + b23 + b33) * k3
        #     + (b04 + b14 + b24 + b34) * k4
        #     + (b05 + b15 + b25 + b35) * k5
        #     + (b06 + b16 + b26 + b36) * k6
        # )  # paper

        # un = un + h * (101 / 363 * k1 - 1369 / 14520 * k3 + 11849 / 14520 * k4) #dimaxer

        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, t_seq, y_seq


def cerk6_dimaxer(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b05 = 0
    b06 = 0
    b11 = -104217 / 37466
    b12 = 0
    b13 = 861101 / 230560
    b14 = -63869 / 293440
    b15 = -1522125 / 762944
    b16 = 165 / 131
    b21 = 1806901 / 618189
    b22 = 0
    b23 = -2178079 / 380424
    b24 = 6244423 / 5325936
    b25 = 982125 / 190736
    b26 = -461 / 131
    b31 = -866577 / 824252
    b32 = 0
    b33 = 12308679 / 5072320
    b34 = -7816583 / 10144640
    b35 = -624375 / 217984
    b36 = 296 / 131
    a21 = 1 / 6
    a31 = 44 / 1369
    a32 = 363 / 1369
    a41 = 3388 / 4913
    a42 = -8349 / 4913
    a43 = 8140 / 4913
    a51 = -36764 / 408375
    a52 = 767 / 1125
    a53 = -32708 / 136125
    a54 = 210392 / 408375
    a61 = 1697 / 18876  # dimaxer 1697 / 18876   #paper -1697 / 18876
    a62 = 0
    a63 = 50653 / 116160
    a64 = 299693 / 1626240
    a65 = 3375 / 11648

    print(f"(b01 + b11 + b21 + b31)={b01 + b11 + b21 + b31}")
    print(f"(b02 + b12 + b22 + b32)={b02 + b12 + b22 + b32}")
    print(f"(b03 + b13 + b23 + b33)={b03 + b13 + b23 + b33}")
    print(f"(b04 + b14 + b24 + b34)={b04 + b14 + b24 + b34}")
    print(f"(b05 + b15 + b25 + b35)={b05 + b15 + b25 + b35}")
    print(f"(b06 + b16 + b26 + b36)={b06 + b16 + b26 + b36}")

    t_seq = []
    y_seq = []

    t = t0
    un = y0
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
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
        c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4 + b25 * k5 + b26 * k6) / h / h
        c3 = (
            (b31 * k1 + b32 * k2 + b33 * k3 + b34 * k4 + b35 * k5 + b36 * k6)
            / h
            / h
            / h
        )

        # un = un + c0 * h + c1 * h * h + c2 * h * h * h + c3 * h * h * h * h
        # 等价于
        # un = un + h * (
        #     (b01 + b11 + b21 + b31) * k1
        #     + (b02 + b12 + b22 + b32) * k2
        #     + (b03 + b13 + b23 + b33) * k3
        #     + (b04 + b14 + b24 + b34) * k4
        #     + (b05 + b15 + b25 + b35) * k5
        #     + (b06 + b16 + b26 + b36) * k6
        # )  # paper

        un = un + h * (
            101 / 363 * k1 - 1369 / 14520 * k3 + 11849 / 14520 * k4
        )  # dimaxer

        t = t + h
        t_seq.append(t)
        y_seq.append(un)
    return un, t_seq, y_seq


def time_order_analysis(time_scheme):
    errors = []
    u_seq = []
    t_seq = []
    for h in h_list:
        y_numerical, t_seq_hi, u_seq_hi = time_scheme(lambda_y, y0, t0, T, h)
        t_seq.append(t_seq_hi)
        u_seq.append(u_seq_hi)
        y_exact = exact_solution(T)
        # y_exact = exact_solution(NSTEP * h)
        error = abs(y_numerical - y_exact)
        errors.append(error)

    log_errors = np.log10(errors)

    index = 0
    order = 0
    for h, error in zip(h_list, errors):
        if index > 0:
            order = (log_errors[index] - log_errors[index - 1]) / (
                log_h[index] - log_h[index - 1]
            )
        print(
            f"h = {h:.6f}, error = {error:.6e}, order = {order:.6e}, exact={exact_solution(T)}"
        )
        index = index + 1

    coefficients = np.polyfit(log_h, log_errors, 1)

    return errors, log_errors, coefficients, t_seq, u_seq


def cerk4_continuous(f, y0, t0, T, h):
    b01 = 1
    b02 = 0
    b03 = 0
    b04 = 0
    b11 = -65 / 48
    b12 = 529 / 384
    b13 = 125 / 128
    b14 = -1
    b21 = 41 / 72
    b22 = -529 / 576
    b23 = -125 / 192
    b24 = 1
    a21 = 12 / 23
    a31 = -68 / 375
    a32 = 368 / 375
    a41 = 31 / 144
    a42 = 529 / 1152  # dimaxer 529/1154  #paper 529/1152
    a43 = 125 / 384

    t = t0
    un = y0

    t_last_time_step_seq = []
    y_last_time_step_seq = []
    y_seq = []
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(un)
    n = int(round((T - t0) / h))  # 计算步数
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
        c2 = (b21 * k1 + b22 * k2 + b23 * k3 + b24 * k4) / h / h

        if i == n - 1:
            ratio = np.linspace(0, 1, 101)
            for r in ratio:
                t_last_time_step_seq.append(t + h * r)
                u_r = un + c0 * r * h + c1 * r * r * h * h + c2 * r * r * r * h * h * h
                y_last_time_step_seq.append(u_r)

        # un = un + c0 * h + c1 * h * h + c2 * h * h * h
        # 等价于
        un = un + h * (
            (b01 + b11 + b21) * k1
            + (b02 + b12 + b22) * k2
            + (b03 + b13 + b23) * k3
            + (b04 + b14 + b24) * k4
        )  # paper
        # un = un + 1 / 24 * k1 + 23 / 24 * k2  # dimaxer
        t = t + h
        t_seq.append(t)
        y_seq.append(un)

    return t_last_time_step_seq, y_last_time_step_seq


def cerk_continous_soln_analysis():
    h = 0.025
    t_seq_cerk4, y_seq_cerk4 = cerk4_continuous(lambda_y, y0, t0, T, h)

    t_seq = np.linspace(T - h, T, 101)
    y_seq_analysis = exact_solution(t_seq)

    plt.figure(figsize=(10, 6))
    plt.plot(t_seq_cerk4, y_seq_cerk4, label="CERK4")
    plt.plot(t_seq, y_seq_analysis, label="exact soln")
    plt.title("continuous soln in one time step")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig("continus_soln_one_time_step.png", dpi=300)


if __name__ == "__main__":

    # cerk_continous_soln_analysis()

    log_h = np.log10(h_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    print("\nEuler forward:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(euler_forward)
    ax1.plot(
        log_h,
        log_errors,
        "ko-",
        label=f"euler forward error (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(
        t_seq[4],
        u_seq[4],
        "k-",
        label=f"euler forward error (slope = {coefficients[0]:.4f})",
    )

    print("\n RK4:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(rk4)
    ax1.plot(
        log_h,
        log_errors,
        "ro-",
        label=f"RK4 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], "r-", label=f"RK4 (slope = {coefficients[0]:.4f})")

    print("\n CERK2:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk2)
    ax1.plot(
        log_h,
        log_errors,
        "bo-",
        label=f"CERK2 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], "b-", label=f"CERK2 (slope = {coefficients[0]:.4f})")

    print("\n CERK4:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk4)
    ax1.plot(
        log_h,
        log_errors,
        "ob--",
        label=f"CERK4 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[1], u_seq[1], "b--", label=f"CERK4 (slope = {coefficients[0]:.4f})")

    # CHECK
    # print("\n CERK4-Dimaxer:")
    # errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk4_dimaxer)
    # ax1.plot(
    #     log_h,
    #     log_errors,
    #     "og--",
    #     label=f"CERK4-Dimaxer (slope = {coefficients[0]:.4f})",
    #     markersize=8,
    # )
    # ax2.plot(
    #     t_seq[1],
    #     u_seq[1],
    #     "g--",
    #     label=f"CERK4-Dimaxer (slope = {coefficients[0]:.4f})",
    # )

    print("\n CERK6:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk6)
    ax1.plot(
        log_h,
        log_errors,
        "ob-.",
        label=f"CERK6 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], "b-.", label=f"CERK6 (slope = {coefficients[0]:.4f})")

    print("\n CERK6-Dimaxer:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk6_dimaxer)
    ax1.plot(
        log_h,
        log_errors,
        "go-.",
        label=f"CERK6-Dimaxer (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(
        t_seq[0],
        u_seq[0],
        "g-.",
        label=f"CERK6-Dimaxer (slope = {coefficients[0]:.4f})",
    )

    ax1.set_xlabel("log(h)")
    ax1.set_ylabel("log(error)")
    ax1.set_title("(log-log)")
    # ax1.legend()
    # ax1.set_aspect(1)
    ax1.grid(True, which="both", ls="--")

    ax2.set_xlabel("t")
    ax2.set_ylabel("y(t)")
    ax2.set_title("y-t")
    # ax2.legend()
    ax2.grid(True, which="both", ls="--")
    ax1.set_aspect(0.1, adjustable="box")

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0),  # 在图形正下方
        ncol=4,  # 4列布局
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        title="legend",
        title_fontsize=10,
    )
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    plt.savefig("time_order_cerk.png", dpi=300)
