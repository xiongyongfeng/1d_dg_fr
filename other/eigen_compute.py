from sympy import symbols, Matrix, expand, simplify, sqrt

# 1. 定义符号变量
w, z, Z = symbols("w z Z")

bb = -(1 + 1.5 * Z + (1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
cc = Z / w * (1 - 1.5 * (1 - w) + 2 * Z / 3 - 0.5 * w * (1 + 1.5 * Z)) + (
    1 + 1.5 * Z
) * ((1 - w) / w + (0.5 * w - 1 / 3) * Z / w)
eigen1 = (-bb + sqrt(bb * bb - 4 * cc)) / 2.0
print(eigen1)
print(simplify(eigen1))
# 2. 构建符号矩阵 A, B, D, E
A = Matrix([[1 / w - 1, 2 - 1 / w], [0, 1]])
B = Matrix([[2 / w - 1, -1], [1, 1]])
D = Matrix([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]])
E = Matrix([[0, 1], [-1, 2]])

# 3. 计算 C = A + B * D * E
print("计算 BD = B * D:")
BD = B * D
print(BD)

print("\n计算 BDE = BD * E:")
BDE = BD * E
print(BDE)

print("\n计算 C = A + BDE:")
C = A + BDE * z / 2
print(C)

print("\n展开矩阵 C 的表达式:")
C_expanded = expand(C)
print(C_expanded)

# 4. 计算矩阵 C 的特征值
print("\n计算矩阵 C 的特征值:")
eigenvalues = C_expanded.eigenvals()

# 特征值可能非常复杂
for eigenval, multiplicity in eigenvalues.items():
    print(f"特征值: {eigenval}, 重数: {multiplicity}")

# 5. (可选) 尝试简化特征值表达式
print("\n尝试简化特征值表达式:")
simplified_eigenvals = {}
for eigenval, mult in eigenvalues.items():
    simplified_eigenval = simplify(eigenval)
    simplified_eigenvals[simplified_eigenval] = mult

for eigenval, multiplicity in simplified_eigenvals.items():
    print(f"特征值: {eigenval}, 重数: {multiplicity}")
