import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 定义复平面范围 (xmin, xmax, ymin, ymax) 和分辨率
x_min, x_max = -3.0, 3.0
y_min, y_max = -3.0, 3.0
resolution = 800  # 图像分辨率，提高分辨率会增加计算时间但图像更精细

# 创建网格点
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# 将网格点转换为复数
Z = X + Y * 1j

# 计算表达式 1 + z + z^2/2 + z^3/6 + z^4/24
expression_rk4 = 1 + Z + (Z**2) / 2 + (Z**3) / 6 + (Z**4) / 24
expression_forward_euler = 1 + Z

expression_twostep_alpha1 = (1 + 1.5 * Z + np.sqrt(2.25 * Z * Z + Z + 1)) / 2.0
expression_twostep_alpha2 = (1 + 1.5 * Z - np.sqrt(2.25 * Z * Z + Z + 1)) / 2.0

# 数值积分
# expression_twostep_discontinuous_alpha1 = (
#     1 + 2.5 * Z + np.sqrt((1 + 2.5 * Z) ** 2 - 4 * Z * (1 + 1.5 * Z))
# ) / 2.0
# expression_twostep_discontinuous_alpha2 = (
#     1 + 2.5 * Z - np.sqrt((1 + 2.5 * Z) ** 2 - 4 * Z * (1 + 1.5 * Z))
# ) / 2.0

# expression_twostep_discontinuous_alpha1 = (
#     1 + Z + np.sqrt((1 + Z) ** 2 - 4 * Z * Z)
# ) / 2.0
# expression_twostep_discontinuous_alpha2 = (
#     1 + Z - np.sqrt((1 + Z) ** 2 - 4 * Z * Z)
# ) / 2.0


# 精确积分
w = 1.0
bb = -(1 + 5 / 3 * Z)
cc = 2 / 3 * Z + 1 / 6 * Z * Z
bb = -(1 + 1.5 * Z + (1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
cc = Z / w * (1 - 1.5 * (1 - w) + 2 * Z / 3 - 0.5 * w * (1 + 1.5 * Z)) + (
    1 + 1.5 * Z
) * ((1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
expression_twostep_discontinuous_alpha1 = (-bb + np.sqrt(bb * bb - 4 * cc)) / 2.0
expression_twostep_discontinuous_alpha2 = (-bb - np.sqrt(bb * bb - 4 * cc)) / 2.0


# 计算表达式的模
modulus_rk4 = np.abs(expression_rk4)
modulus_forward_euler = np.abs(expression_forward_euler)
modulus_twostep_continuous_alpha1 = np.abs(expression_twostep_alpha1)
modulus_twostep_continuous_alpha2 = np.abs(expression_twostep_alpha2)

modulus_twostep_discontinuous_alpha1 = np.abs(expression_twostep_discontinuous_alpha1)
modulus_twostep_discontinuous_alpha2 = np.abs(expression_twostep_discontinuous_alpha2)

# 绘制图形
plt.figure(figsize=(10, 10))


plt.contourf(X, Y, modulus_rk4, levels=[0, 1.0], colors="red", alpha=0.1)
plt.contourf(X, Y, modulus_forward_euler, levels=[0, 1], colors="gray", alpha=0.3)

val_twostep_continuous = 1.0
mask_intersection = (modulus_twostep_continuous_alpha1 < val_twostep_continuous) & (
    modulus_twostep_continuous_alpha2 < val_twostep_continuous
)
plt.contourf(X, Y, mask_intersection, levels=[0.5, 1.5], colors="green", alpha=0.2)

val_twostep_discontinuous = 1.0
mask_intersection_testfunc = (
    modulus_twostep_discontinuous_alpha1 < val_twostep_discontinuous
) & (modulus_twostep_discontinuous_alpha2 < val_twostep_discontinuous)
plt.contourf(
    X, Y, mask_intersection_testfunc, levels=[0.5, 1.5], colors="blue", alpha=0.2
)

# 使用 contour 绘制 |expression| = 1 的等高线[7](@ref)
# levels=[1] 表示只绘制模等于1的等高线[7](@ref)
contour_rk4 = plt.contour(X, Y, modulus_rk4, levels=[1], colors="red", linewidths=2)
contour_forward_euler = plt.contour(
    X, Y, modulus_forward_euler, levels=[1], colors="black", linewidths=2
)
contour_twostep_continuous_alpha1 = plt.contour(
    X,
    Y,
    modulus_twostep_continuous_alpha1,
    levels=[val_twostep_continuous],
    colors="green",
    linewidths=2,
)
contour_twostep_continuous_alpha2 = plt.contour(
    X,
    Y,
    modulus_twostep_continuous_alpha2,
    levels=[val_twostep_continuous],
    colors="green",
    linewidths=2,
)

contour_twostep_continuous_alpha1 = plt.contour(
    X,
    Y,
    modulus_twostep_discontinuous_alpha1,
    levels=[val_twostep_discontinuous],
    colors="blue",
    linewidths=2,
)
contour_twostep_continuous_alpha2 = plt.contour(
    X,
    Y,
    modulus_twostep_discontinuous_alpha2,
    levels=[val_twostep_discontinuous],
    colors="blue",
    linewidths=2,
)

# 添加标题和标签
# plt.title("|1+z+z^2/2+z^3/6+z^4/24| = 1", fontsize=14)
plt.xlabel("(Re(z))")
plt.ylabel("(Im(z))")

# 添加网格线以便更好地读取坐标
plt.grid(True, linestyle="--", alpha=0.5)

# 设置坐标轴范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# # 创建代理艺术家用于图例
red_line = mlines.Line2D([], [], color="red", linewidth=2, label="RK4")
black_line = mlines.Line2D([], [], color="black", linewidth=2, label="Forward Euler")
green_line = mlines.Line2D([], [], color="green", linewidth=2, label="Two-step")
blue_solid_line = mlines.Line2D(
    [], [], color="blue", linewidth=2, label="new time scheme (solid)"
)

# 添加图例
plt.legend(
    handles=[red_line, black_line, green_line, blue_solid_line],
    loc="best",
    fontsize=10,
)

plt.savefig("stable_region.png")

# 显示图形
# plt.tight_layout()
# plt.show()
