import numpy as np
import matplotlib.pyplot as plt

# 定义参数
lambda_val = -3  # λ值
y0 = 1.0  # 初始条件 y(0)=1
t0 = 0.0  # 初始时间
T = 1.0  # 评估时间点
h_list = [0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # 步长列表


# 定义微分方程 dy/dt = λy
def f(t, y):
    return lambda_val * y


# 理论解 y(t) = exp(λt)
def exact_solution(t):
    return np.exp(lambda_val * t)


# 欧拉向前方法计算在时间T的数值解
def euler_forward_at_T(f, y0, t0, T, h):
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
    n = int(round((T - t0) / h))  # 计算步数
    for i in range(n):
        y = y + h * f(t, y)
        t = t + h
    return y


# 欧拉向前方法计算在时间T的数值解
def new_timescheme_at_T(f, y0, t0, T, h, w):

    t = t0
    n = int(round((T - t0) / h))  # 计算步数
    t_0 = t0
    t_1 = t0 + h
    y_tmp_0 = y0 - h * f(t0, y0)
    y_tmp_1 = y0
    ustar_0 = y0
    ustar_1 = ustar_0 + h * f(t0, y0)
    print(f"ustar_0={ustar_0},ustar_1={ustar_1},f(ustar_0, t_0)={f(ustar_0, t_0)}")
    for i in range(n):
        # print(
        #     f"i={i},t_0 = {t_0},t_1 = {t_1},y_tmp_0={y_tmp_0},y_tmp_1={y_tmp_1},ustar_0={ustar_0},ustar_1={ustar_1}"
        # )
        # y_0 = (
        #     (1.0 / w - 1) * y_tmp_0
        #     + (2 - 1.0 / w) * y_tmp_1
        #     + 0.5 * h * (y_tmp_0 + (2.0 / w - 3.0) * y_tmp_1)
        # )
        # y_1 = y_tmp_1 + 0.5 * h * (-y_tmp_0 + 3 * y_tmp_1)

        y_0 = (
            (1.0 / w - 1) * y_tmp_0
            + (2 - 1.0 / w) * y_tmp_1
            + 0.5 * h * ((2.0 / w - 1.0) * f(t_0, ustar_0) - f(t_1, ustar_1))
        )
        y_1 = y_tmp_1 + 0.5 * h * (f(t_0, ustar_0) + f(t_1, ustar_1))
        y_tmp_0 = y_0
        y_tmp_1 = y_1
        ustar_0 = y_tmp_1
        ustar_1 = -y_tmp_0 + 2 * y_tmp_1
        t_0 = t_0 + h
        t_1 = t_1 + h

    y = y_1
    return y


# 计算不同步长下的误差
errors_new = []
for h in h_list:
    y_numerical = new_timescheme_at_T(f, y0, t0, T, h, 0.6)
    y_exact = exact_solution(T)
    error = abs(y_numerical - y_exact)
    errors_new.append(error)

# 计算log(h)和log(error)
log_h = np.log(h_list)
log_error_new = np.log(errors_new)

# 线性拟合求斜率（阶数p）
coefficients_new = np.polyfit(log_h, log_error_new, 1)
slope_new = coefficients_new[0]
intercept_new = coefficients_new[1]

# 输出结果
print("New Scheme 步长h和误差error:")
index = 0
for h, error in zip(h_list, errors_new):
    # order = (np.log(errors_new[index + 1]) - np.log(errors_new[index])) / (
    #     np.log(h_list[index + 1]) - np.log(h_list[index])
    # )
    index = index + 1
    # print(f"h = {h:.6f}, error = {error:.6e},order = {order:.6e}")
    print(f"h = {h:.6f}, error = {error:.6e}")
print(f"\nNew Scheme  拟合的斜率 (阶数 p) = {slope_new:.4f}")

errors = []
for h in h_list:
    y_numerical = euler_forward_at_T(f, y0, t0, T, h)
    y_exact = exact_solution(T)
    error = abs(y_numerical - y_exact)
    errors.append(error)

# 计算log(h)和log(error)
log_h = np.log(h_list)
log_error = np.log(errors)

# 线性拟合求斜率（阶数p）
coefficients = np.polyfit(log_h, log_error, 1)
slope = coefficients[0]
intercept = coefficients[1]

# 输出结果
print("Euler forward 步长h和误差error:")
for h, error in zip(h_list, errors):
    print(f"h = {h:.6f}, error = {error:.6e}")
print(f"\nEuler forward  拟合的斜率 (阶数 p) = {slope:.4f}")


# 绘制log-log图
plt.figure(figsize=(8, 6))
plt.plot(log_h, log_error, "o-", label="euler forward error", markersize=8)
plt.plot(
    log_h, slope * log_h + intercept, "r--", label=f"euler forward slope= {slope:.4f})"
)

plt.plot(log_h, log_error_new, "o-", label="new schme error", markersize=8)
plt.plot(
    log_h,
    slope_new * log_h + intercept_new,
    "r--",
    label=f"new schme  slope= {slope_new:.4f})",
)

plt.xlabel("log(h)")
plt.ylabel("log(error)")
plt.title("(log-log)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("time_order.png", dpi=300)
# plt.show()
