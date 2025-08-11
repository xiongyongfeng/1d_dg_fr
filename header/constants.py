from sympy import symbols, Eq, solve, diff, integrate, Matrix, init_printing, expand, factor, simplify,sin
init_printing()  # 优化输出格式（如 LaTeX 风格）
x, y = symbols('x y')  # 定义符号变量[1,5,10](@ref)
expr = (x + y)**2
print(expr)
expanded_expr = expand(expr)      # 展开 → x² + 2xy + y²[5,10](@ref)
print(expanded_expr)
factored_expr = factor(x**2 - 4)  # 因式分解 → (x-2)(x+2)[10](@ref)
print(factored_expr)
simplified_expr = simplify((x**2 - 1)/(x + 1))  # 化简 → x - 1[5](@ref)
print(simplified_expr)
derivative = diff(x**3 + 2*x, x)  # 导数 → 3x² + 2[1,7,10](@ref)
print(derivative)
integral = integrate(sin(x), x)          # 不定积分 → -cos(x)[5,7](@ref)
print(integral)
definite_int = integrate(x**2, (x, 0, 1)) # 定积分（0→1）→ 1/3[1,10](@ref)
print(definite_int)

A = Matrix([[x, y], [1, x]])
B = Matrix([[2, 0], [y, 3]])
result = A * B            # 矩阵乘法[1,2](@ref)
inverse_A = A.inv()       # 矩阵求逆[1,6](@ref)
eigenvalues = A.eigenvals()  # 特征值[2](@ref)

import numpy as np
from scipy.special import roots_legendre, legendre

import numpy as np
from scipy.special import legendre
from scipy.optimize import newton

def legendre_deriv_roots(n, num_guesses=50):
    # 获取n阶Legendre多项式及其导数
    poly = legendre(n)
    deriv = poly.deriv()  # P_n'(x)
    
    # 在[-1, 1]生成初始猜测点（避免边界）
    guesses = np.linspace(-0.99, 0.99, num_guesses)
    roots = []
    
    # 对每个猜测点应用牛顿法求根
    for x0 in guesses:
        try:
            root = newton(deriv, x0=x0, tol=1e-10, maxiter=100)
            root = np.round(root, 8)  # 避免浮点误差导致的重复
            if -1 <= root <= 1 and root not in roots:
                roots.append(root)
        except RuntimeError:  # 处理迭代不收敛
            continue
    
    return np.sort(np.array(roots))

def lgl_nodes_weights(n):
    if n == 0:
        return np.array([0.0]), np.array([2.0])
    elif n == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    
    # 计算n-1阶Gauss-Legendre节点（内部点）
    
    x_lgl = np.concatenate([[-1.0], legendre_deriv_roots(n), [1.0]])
    
    # 计算权重：w_i = 2 / [n(n+1) * P_n(x_i)^2
    P_n = legendre(n)
    denom = n * (n + 1) * P_n(x_lgl)**2
    w_lgl = 2.0 / denom
    w_lgl[0] = w_lgl[-1] = 2.0 / (n * (n + 1))  # 端点修正
    return x_lgl, w_lgl
def lagrange_basis(x_nodes, x_eval):
    n = len(x_nodes)
    basis = np.zeros((n, len(x_eval)))
    for i in range(n):
        numerator = np.ones_like(x_eval)
        denominator = 1.0
        for j in range(n):
            if i != j:
                numerator *= (x_eval - x_nodes[j])
                denominator *= (x_nodes[i] - x_nodes[j])
        basis[i] = numerator / denominator
    return basis
def generate_lagrange_basis(points):
    """
    自动生成拉格朗日基函数
    :param points: 插值点数组，例如 [-1, 0, 1]
    :return: 基函数列表，每个元素为符号表达式
    """
    x = symbols('x')  # 定义符号变量
    n = len(points)    # 插值点数量
    basis_functions = []
    
    for i in range(n):
        # 构造第 i 个基函数 φ_i(x)
        numerator = 1   # 分子初始化
        denominator = 1 # 分母初始化
        
        # 计算 ∏_{j≠i} (x - x_j) 和 ∏_{j≠i} (x_i - x_j)
        for j in range(n):
            if j != i:
                numerator *= (x - points[j])
                denominator *= (points[i] - points[j])
        
        # 组合成分式并简化
        phi_i = numerator / denominator
        basis_functions.append(simplify(phi_i))
    
    return basis_functions

n = 1
nsp = n+1
x_lgl, w_lgl = lgl_nodes_weights(n)
print(x_lgl)
print(w_lgl)
basis =generate_lagrange_basis(x_lgl)
print(basis)
deri_basis = [diff(phi, x) for phi in basis]
print(deri_basis)

mmatrix = np.zeros((nsp, nsp))
for i in range(0, nsp):
  for j in range(0, nsp):
    mmatrix[i,j] =  integrate(basis[i]*basis[j], (x, -1, 1))
print("Mass Matrix = \n",mmatrix)

inverse_mmatrix = np.linalg.inv(mmatrix)
print("Mass Matrix invert = \n",inverse_mmatrix)

dmatrix = np.zeros((nsp, nsp))
for i in range(0, nsp):
  for j in range(0, nsp):
    dmatrix[i,j] =  deri_basis[j].subs(x,x_lgl[i])
print("Deritive Matrix invert = \n",dmatrix)

smatrix = np.dot(mmatrix,dmatrix)
print("S Matrix = \n",smatrix)