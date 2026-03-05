import os
import csv
import matplotlib.pyplot as plt

# 测试用例配置
cases = [
    ("lim0_fr0", "./soln_lim0_fr0", "Limiter=0, FR=DG"),
    ("lim0_fr1", "./soln_lim0_fr1", "Limiter=0, FR=FR"),
    ("lim1_fr0", "./soln_lim1_fr0", "Limiter=1, FR=DG"),
    ("lim1_fr1", "./soln_lim1_fr1", "Limiter=1, FR=FR"),
]

# 读取最终时刻的结果
final_time = "1.999998"

plt.figure(figsize=(12, 8))

colors = ['red', 'blue', 'green', 'orange']
markers = ['o', 's', '^', 'd']

for i, (name, path, label) in enumerate(cases):
    csv_file = os.path.join(path, f"result_after{final_time}.csv")

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        continue

    x_values = []
    y_values = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            # 提取x坐标（第一个点）
            x = float(row[0])
            # 提取u值（原始变量）
            if len(row) == 5:  # ORDER=1, NSP=2
                y = float(row[3])  # 第一个原始变量
            elif len(row) == 7:  # ORDER=2, NSP=3
                y = float(row[3])
            else:
                continue
            x_values.append(x)
            y_values.append(y)

    plt.plot(x_values, y_values, color=colors[i], marker=markers[i],
             markersize=3, linewidth=1.5, label=label)

plt.xlabel("X", fontsize=12)
plt.ylabel("U", fontsize=12)
plt.title("Burgers Equation: Limiter vs DG/FR Comparison (t=2.0)", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 6.28)
plt.ylim(-1, 2.5)

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.close()

print("Comparison plot saved to comparison.png")
