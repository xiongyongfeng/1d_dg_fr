import numpy as np
import matplotlib.pyplot as plt

# 定义参数
lambda_val = -3  # λ值
y0 = 1.0  # 初始条件 y(0)=1
t0 = 0.0  # 初始时间
T = 1.0  # 评估时间点
h_list = [0.2, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # 步长列表
# h_list = [0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # 步长列表


# 定义微分方程 dy/dt = λy
def lambda_y(t, y):
    return lambda_val * y


# 理论解 y(t) = exp(λt)
def exact_solution(t):
    return np.exp(lambda_val * np.array(t))


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


def twosteptwostage(f, y0, t0, T, h):
    omega = 1 / 1.0
    theta = 1.0

    # a_20 = 0
    # d_20 = ((2.0 - np.sqrt(2.0)) / (2.0 * omega)) ** 2
    # v2 = 2.0

    # a_20 = 0
    # d_20 = (3.0 / 2.0 / omega) ** 2
    # v2 = 0.2

    # a_20 = -0.4994
    # d_20 = 0.0101
    # v2 = 0.4149

    a_20 = -1
    d_20 = 1
    v2 = 0.167

    a_21 = omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - a_20)
    d_21 = 1.0 - d_20
    b0 = (
        2 * theta**3
        + 3 * theta**2 * omega
        - 6 * v2 * omega * omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - 2 * a_20)
    ) / omega**3
    b1 = 1 - b0

    v0 = (
        theta**3
        + theta**2 * omega
        - v2 * omega * omega * (3 * (d_20 - 2 * a_20) + 2 * np.sqrt(d_20 - 2 * a_20))
    ) / omega**3
    v1 = theta * (theta + omega) ** 2 / omega**2 - v2 * (
        1 + 4 * np.sqrt(d_20 - 2 * a_20) + 3 * (d_20 - 2 * a_20)
    )

    t = t0
    n = int(round((T - t0) / h))  # 计算步数
    u_n = y0
    k1_0 = lambda_y(t0, y0)
    k2_0 = lambda_y(t0, y0 - k1_0 * h)
    u_n_1 = y0 - 0.5 * h * (k1_0 + k2_0)
    # u_n_1 = y0  # 第一步会严重影响精度
    # u_n_1 = y0 - lambda_y(t0, y0) * h
    # u_n_1 = exact_solution(-h)

    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(u_n)

    for i in range(n):
        k0 = lambda_y(t, u_n_1)
        # if i == 0:
        #     k0 = 0
        k1 = lambda_y(t, u_n)
        y2 = d_20 * u_n_1 + d_21 * u_n + h * (omega * a_20 * k0 + a_21 * k1)
        k2 = lambda_y(t, y2)
        y = b0 * u_n_1 + b1 * u_n + h * (omega * v0 * k0 + v1 * k1 + v2 * k2)
        t = t + h
        u_n_1 = u_n
        u_n = y
        t_seq.append(t)
        y_seq.append(y)
    return y, t_seq, y_seq


def twosteptwostagev2(f, y0, t0, T, h):
    omega = 1 / 1
    alpha = 1 / omega
    theta = 1.0

    a1 = -alpha
    a2 = np.sqrt(2) + alpha

    b11 = 0
    b12 = -alpha * a1 / ((a1 - a2))
    b13 = -alpha * (a1 + a2) * alpha / (3 * (a1 - a2))
    b21 = 1
    b22 = (3 + 6 * a1 * alpha) / (6 * (a1 - a2))
    b23 = 2 * alpha * (1 + (a1 + a2) * alpha) / (6 * (a1 - a2))
    b31 = 0
    b32 = -1 / (2 * (a1 - a2))
    b33 = -alpha / (3 * (a1 - a2))

    b1 = -((alpha * theta**2 * (3 * a1 + (a1 + a2) * alpha * theta)) / (3 * (a1 - a2)))
    b2 = (
        theta
        * (
            6 * a1
            - 6 * a2
            + 3 * theta
            + 6 * a1 * alpha * theta
            + 2 * alpha * (1 + (a1 + a2) * alpha) * theta**2
        )
    ) / (6 * (a1 - a2))
    b3 = -((theta**2 * (3 + 2 * alpha * theta)) / (6 * (a1 - a2)))

    t = t0
    n = int(round((T - t0) / h))  # 计算步数
    u_n = y0
    k1_0 = lambda_y(t0, y0)
    k2_0 = lambda_y(t0, y0 - k1_0 * h)
    # u_n_1 = y0 - 0.5 * h * (k1_0 + k2_0)
    # u_n_1 = y0  # 第一步会严重影响精度
    # u_n_1 = y0 - lambda_y(t0, y0) * h
    u_n_1 = exact_solution(-h)

    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(u_n)

    for i in range(n):
        k0 = lambda_y(t, u_n_1)
        # if i == 0:
        #     k0 = 0
        k1 = lambda_y(t, u_n)
        y2 = u_n + a1 * h * k0 + a2 * h * k1
        k2 = lambda_y(t, y2)
        # y = u_n + b1 * h * k0 + b2 * h * k1 + b3 * h * k2
        y = (
            u_n
            + (b11 * theta + b12 * theta * theta + b13 * theta * theta * theta) * h * k0
            + (b21 * theta + b22 * theta * theta + b23 * theta * theta * theta) * h * k1
            + (b31 * theta + b32 * theta * theta + b33 * theta * theta * theta) * h * k2
        )
        t = t + h
        u_n_1 = u_n
        u_n = y
        t_seq.append(t)
        y_seq.append(y)
    return y, t_seq, y_seq


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

    log_errors = np.log(errors)

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

        if i >= n - 2:
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

    return t_last_time_step_seq, y_last_time_step_seq, t_seq, y_seq


def twosteptwostage_continuous(y0, t0, T, h):
    omega = 1 / 1.0
    theta = 1.0

    a_20 = 0
    d_20 = ((2.0 - np.sqrt(2.0)) / (2.0 * omega)) ** 2
    v2 = 2.0

    # a_20 = 0
    # d_20 = (3.0 / 2.0 / omega) ** 2
    # v2 = 0.2

    # a_20 = -1
    # d_20 = 1
    # v2 = 0.167

    a_21 = omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - a_20)
    d_21 = 1.0 - d_20
    b0 = (
        2 * theta**3
        + 3 * theta**2 * omega
        - 6 * v2 * omega * omega * (np.sqrt(d_20 - 2 * a_20) + d_20 - 2 * a_20)
    ) / omega**3
    b1 = 1 - b0

    v0 = (
        theta**3
        + theta**2 * omega
        - v2 * omega * omega * (3 * (d_20 - 2 * a_20) + 2 * np.sqrt(d_20 - 2 * a_20))
    ) / omega**3
    v1 = theta * (theta + omega) ** 2 / omega**2 - v2 * (
        1 + 4 * np.sqrt(d_20 - 2 * a_20) + 3 * (d_20 - 2 * a_20)
    )

    t = t0
    n = int(round((T - t0) / h))  # 计算步数
    u_n = y0
    # u_n_1 = y0  # 第一步会严重影响精度
    u_n_1 = exact_solution(-h)
    # u_n_1 = y0 - lambda_y(t0, y0) * h

    t_last_time_step_seq = []
    y_last_time_step_seq = []
    y_last_time_step_seq_modify = []
    t_seq = []
    y_seq = []
    t_seq.append(t)
    y_seq.append(u_n)

    for i in range(n):
        y2 = (
            d_20 * u_n_1
            + d_21 * u_n
            + h * (omega * a_20 * lambda_y(t, u_n_1) + a_21 * lambda_y(t, u_n))
        )
        y = (
            b0 * u_n_1
            + b1 * u_n
            + h
            * (
                omega * v0 * lambda_y(t, u_n_1)
                + v1 * lambda_y(t, u_n)
                + v2 * lambda_y(t, y2)
            )
        )

        if i >= n - 2:
            print(f"istep={i}")
            print(f"u_n = {u_n}")
            print(f"u_n+1 = {y}")
            ratio = np.linspace(0, 1, 101)
            r_idx = 0
            u_r0 = 0
            for r in ratio:
                b0 = (
                    2 * r**3
                    + 3 * r**2 * omega
                    - 6
                    * v2
                    * omega
                    * omega
                    * (np.sqrt(d_20 - 2 * a_20) + d_20 - 2 * a_20)
                ) / omega**3
                b1 = 1 - b0

                v0 = (
                    r**3
                    + r**2 * omega
                    - v2
                    * omega
                    * omega
                    * (3 * (d_20 - 2 * a_20) + 2 * np.sqrt(d_20 - 2 * a_20))
                ) / omega**3
                v1 = r * (r + omega) ** 2 / omega**2 - v2 * (
                    1 + 4 * np.sqrt(d_20 - 2 * a_20) + 3 * (d_20 - 2 * a_20)
                )
                t_last_time_step_seq.append(t + h * r)
                u_r = (
                    b0 * u_n_1
                    + b1 * u_n
                    + h
                    * (
                        omega * v0 * lambda_y(t, u_n_1)
                        + v1 * lambda_y(t, u_n)
                        + v2 * lambda_y(t, y2)
                    )
                )
                y_last_time_step_seq.append(u_r)

                if r_idx == 0:
                    u_r0 = u_r

                # y_last_time_step_seq_modify.append(u_r + (1 - r) * (u_n - u_r0))

                # 四阶
                c0 = u_n
                c1 = lambda_y(t, u_n)
                delta_m = u_n_1 - u_n + lambda_y(t, u_n) * omega
                gamma = lambda_y(t, u_n_1) - lambda_y(t, u_n)
                delta_p = y - u_n - lambda_y(t, u_n)
                c2 = (
                    (3 + 4 * omega) / (omega * (1 + omega)) ** 2 * delta_m
                    + gamma / (omega * (1 + omega))
                    + omega * omega / (1 + omega) ** 2 * delta_p
                )
                c3 = (
                    2
                    * (1 - 2 * omega * omega)
                    / (omega**3 * (1 + omega) ** 2)
                    * delta_m
                    + (1 - omega) / (omega * omega * (1 + omega)) * gamma
                    + (2 * omega) / (1 + omega) ** 2 * delta_p
                )
                c4 = (
                    -(2 + 3 * omega) / (omega**3 * (1 + omega) ** 2) * delta_m
                    - gamma / (omega**2 * (1 + omega))
                    + delta_p / (1 + omega) ** 2
                )

                # y_last_time_step_seq_modify.append(
                #     c0 + c1 * r + c2 * r**2 + c3 * r**3 + c4 * r**4
                # )  # u(r) = c0+c1*r+c2*r^2+c3*r^3+c4*r^4

                # 三阶
                c0 = u_n
                c1 = lambda_y(t, u_n)
                delta_m = u_n_1 - u_n + lambda_y(t, u_n) * omega
                delta_p = y - u_n - lambda_y(t, u_n)
                c2 = (delta_m + omega**3 * delta_p) / (omega**2 * (1 + omega))
                c3 = (-delta_m + omega**2 * delta_p) / (omega**2 * (1 + omega))
                # y_last_time_step_seq_modify.append(
                #     c0 + c1 * r + c2 * r**2 + c3 * r**3
                # )  # u(r) = c0+c1*r+c2*r^2+c3*r^3+c4*r^4

                # 二阶
                c0 = u_n
                c1 = (omega * omega * y + (1 - omega * omega) * u_n - u_n_1) / (
                    omega * (1 + omega)
                )
                c2 = (u_n_1 - (1 + omega) * u_n + omega * y) / (omega * (1 + omega))
                y_last_time_step_seq_modify.append(
                    c0 + c1 * r + c2 * r**2
                )  # u(r) = c0+c1*r+c2*r^2+c3*r^3

                r_idx += 1

            print(f"yn(r=0)={y_last_time_step_seq[0]}")
            print(f"yn(r=1)={y_last_time_step_seq[100]}")
            print(f"(yn(r=0)-un)/un={(y_last_time_step_seq[0]-u_n)/u_n}")
        t = t + h
        u_n_1 = u_n
        u_n = y
        t_seq.append(t)
        y_seq.append(y)
    return (
        t_last_time_step_seq,
        y_last_time_step_seq,
        y_last_time_step_seq_modify,
        t_seq,
        y_seq,
    )


def twosteptwostagev2_continuous(y0, t0, T, h):
    omega = 1 / 1
    alpha = 1 / omega
    theta = 1.0

    a1 = -alpha
    a2 = np.sqrt(2) + alpha

    b11 = 0
    b12 = -alpha * a1 / ((a1 - a2))
    b13 = -alpha * (a1 + a2) * alpha / (3 * (a1 - a2))
    b21 = 1
    b22 = (3 + 6 * a1 * alpha) / (6 * (a1 - a2))
    b23 = 2 * alpha * (1 + (a1 + a2) * alpha) / (6 * (a1 - a2))
    b31 = 0
    b32 = -3 / (6 * (a1 - a2))
    b33 = -2 * alpha / (6 * (a1 - a2))

    b1 = -((alpha * theta**2 * (3 * a1 + (a1 + a2) * alpha * theta)) / (3 * (a1 - a2)))
    b2 = (
        theta
        * (
            6 * a1
            - 6 * a2
            + 3 * theta
            + 6 * a1 * alpha * theta
            + 2 * alpha * (1 + (a1 + a2) * alpha) * theta**2
        )
    ) / (6 * (a1 - a2))
    b3 = -((theta**2 * (3 + 2 * alpha * theta)) / (6 * (a1 - a2)))

    t = t0
    n = int(round((T - t0) / h))  # 计算步数
    u_n = y0
    k1_0 = lambda_y(t0, y0)
    k2_0 = lambda_y(t0, y0 - k1_0 * h)
    # u_n_1 = y0 - 0.5 * h * (k1_0 + k2_0)
    # u_n_1 = y0  # 第一步会严重影响精度
    # u_n_1 = y0 - lambda_y(t0, y0) * h
    u_n_1 = exact_solution(-h)

    t_seq = []
    y_seq = []
    t_last_time_step_seq = []
    y_last_time_step_seq = []
    t_seq.append(t)
    y_seq.append(u_n)

    for i in range(n):
        k0 = lambda_y(t, u_n_1)
        # if i == 0:
        #     k0 = 0
        k1 = lambda_y(t, u_n)
        y2 = u_n + a1 * h * k0 + a2 * h * k1
        k2 = lambda_y(t, y2)
        y = u_n + b1 * h * k0 + b2 * h * k1 + b3 * h * k2
        if i >= n - 2:
            ratio = np.linspace(0, 1, 101)
            for theta in ratio:
                t_last_time_step_seq.append(t + h * theta)
                u_r = (
                    u_n
                    + (b11 * theta + b12 * theta * theta + b13 * theta * theta * theta)
                    * h
                    * k0
                    + (b21 * theta + b22 * theta * theta + b23 * theta * theta * theta)
                    * h
                    * k1
                    + (b31 * theta + b32 * theta * theta + b33 * theta * theta * theta)
                    * h
                    * k2
                )
                y_last_time_step_seq.append(u_r)
        t = t + h
        u_n_1 = u_n
        u_n = y
        t_seq.append(t)
        y_seq.append(y)
    return t_last_time_step_seq, y_last_time_step_seq, t_seq, y_seq


def cerk_continous_soln_analysis():
    h = 0.2
    nstep = int(round((T - t0) / h))
    (
        t_seq_cerk4_onestep,
        y_seq_cerk4_onestep,
        t_seq_cerk4,
        y_seq_cerk4,
    ) = cerk4_continuous(lambda_y, y0, t0, T, h)
    (
        t_seq_2step2stage_onestep,
        y_seq_2step2stage_onestep,
        y_seq_2step2stage_onestep_modify,
        t_seq_2step2stage,
        y_seq_2step2stage,
    ) = twosteptwostage_continuous(y0, t0, T, h)
    (
        t_seq_2step2stagev2_onestep,
        y_seq_2step2stagev2_onestep,
        t_seq_2step2stagev2,
        y_seq_2step2stagev2,
    ) = twosteptwostagev2_continuous(y0, t0, T, h)

    t_seq_analysis = np.linspace(t_seq_cerk4[nstep - 2], t_seq_cerk4[nstep], 201)
    y_seq_analysis = exact_solution(t_seq_analysis)

    plt.figure(figsize=(10, 6))
    plt.plot(
        t_seq_cerk4_onestep, y_seq_cerk4_onestep, color="red", label="CERK4_continous"
    )
    plt.scatter(
        t_seq_cerk4[nstep - 2 : nstep + 1],
        y_seq_cerk4[nstep - 2 : nstep + 1],
        label="CERK4",
        color="red",
        marker="o",
    )
    plt.plot(
        t_seq_2step2stage_onestep,
        y_seq_2step2stage_onestep,
        label="2step2stage_continuous",
        color="blue",
    )
    plt.plot(
        t_seq_2step2stage_onestep,
        y_seq_2step2stage_onestep_modify,
        label="2step2stage_continuous_modify",
        color="blue",
        linestyle="--",
    )
    plt.scatter(
        t_seq_2step2stage[nstep - 2 : nstep + 1],
        y_seq_2step2stage[nstep - 2 : nstep + 1],
        label="2step2stage",
        color="blue",
        marker="o",
    )
    plt.scatter(
        t_seq_2step2stagev2[nstep - 2 : nstep + 1],
        y_seq_2step2stagev2[nstep - 2 : nstep + 1],
        label="2step2stage",
        color="pink",
        marker="o",
    )
    plt.plot(
        t_seq_2step2stagev2_onestep,
        y_seq_2step2stagev2_onestep,
        label="2step2stagev2_continuous",
        color="pink",
        linestyle="--",
    )
    plt.plot(t_seq_analysis, y_seq_analysis, label="exact soln")
    plt.title("continuous soln in one time step")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig("continus_soln_one_time_step.png", dpi=300)


if __name__ == "__main__":

    cerk_continous_soln_analysis()

    log_h = np.log(h_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    print("\nEuler forward:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(euler_forward)
    ax1.plot(
        log_h,
        log_errors,
        "o-",
        label=f"euler forward error (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(
        t_seq[0], u_seq[0], label=f"euler forward error (slope = {coefficients[0]:.4f})"
    )

    print("\n RK4:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(rk4)
    ax1.plot(
        log_h,
        log_errors,
        "o-",
        label=f"RK4 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], label=f"RK4 (slope = {coefficients[0]:.4f})")

    print("\n CERK4:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(cerk4)
    ax1.plot(
        log_h,
        log_errors,
        "o-",
        label=f"CERK4 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], label=f"CERK4 (slope = {coefficients[0]:.4f})")

    print("\n2step2stage:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(
        twosteptwostage
    )
    ax1.plot(
        log_h,
        log_errors,
        "o-",
        label=f"2step2stage (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(t_seq[0], u_seq[0], label=f"2step2stage (slope = {coefficients[0]:.4f})")

    print("\n2step2stagev2:")
    errors, log_errors, coefficients, t_seq, u_seq = time_order_analysis(
        twosteptwostagev2
    )
    ax1.plot(
        log_h,
        log_errors,
        "o-",
        color="pink",
        label=f"2step2stagev2 (slope = {coefficients[0]:.4f})",
        markersize=8,
    )
    ax2.plot(
        t_seq[0],
        u_seq[0],
        color="pink",
        label=f"2step2stagev2 (slope = {coefficients[0]:.4f})",
    )

    ax1.set_xlabel("log(h)")
    ax1.set_ylabel("log(error)")
    ax1.set_title("(log-log)")
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    ax2.set_xlabel("t")
    ax2.set_ylabel("y(t)")
    ax2.set_title("y-t")
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    plt.savefig("time_order.png", dpi=300)
