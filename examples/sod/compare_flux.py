import os
import json
import subprocess
import matplotlib.pyplot as plt
import csv

# 基础配置
base_config = {
    "x0": -5.0,
    "x1": 5.0,
    "n_ele": 400,
    "total_time": 1.2999999523162842,
    "output_time_step": 0.10000000149011612,
    "cfl": 0.4,
    "dt": 1e-08,
    "limiter_type": 1,
    "dg_fr_type": 0,
    "enable_entropy_modify": False,
    "weight": 0.5,
    "time_scheme_type": 0,
    "bc_type": 2,
    "bc_left": 0.0,
    "bc_right": 0.0,
}

# 六个 case 的配置
flux_names = {
    0: "LF",
    1: "HLL",
    2: "HLLC",
    3: "ROE",
    4: "AUSM",
    5: "AUSM+UP",
    6: "AUSM+",
}

cases = [
    {"name": "common_flux_0", "common_flux_type": 0, "output_dir": "./soln_flux0"},
    {"name": "common_flux_1", "common_flux_type": 1, "output_dir": "./soln_flux1"},
    {"name": "common_flux_2", "common_flux_type": 2, "output_dir": "./soln_flux2"},
    {"name": "common_flux_3", "common_flux_type": 3, "output_dir": "./soln_flux3"},
    {"name": "common_flux_4", "common_flux_type": 4, "output_dir": "./soln_flux4"},
    {"name": "common_flux_5", "common_flux_type": 5, "output_dir": "./soln_flux5"},
    {"name": "common_flux_6", "common_flux_type": 6, "output_dir": "./soln_flux6"},
]

results = {}

for case in cases:
    config = base_config.copy()
    config["common_flux_type"] = case["common_flux_type"]
    config["output_dir"] = case["output_dir"]

    # 写入配置文件
    config_file = f"sod_flux{case['common_flux_type']}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    # 清理旧输出目录
    output_dir = case["output_dir"]
    avg_dir = output_dir + "_avg"
    if os.path.exists(output_dir):
        subprocess.run(["rm", "-r", output_dir])
    if os.path.exists(avg_dir):
        subprocess.run(["rm", "-r", avg_dir])

    # 运行计算
    print(f"Running case: {case['name']}")
    subprocess.run(["../main_ns_k2", config_file])

    # 找到最后一个输出文件
    csv_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("result_after") and f.endswith(".csv")
    ]
    if csv_files:
        csv_files.sort()
        last_file = csv_files[-1]
        results[case["name"]] = {
            "file": os.path.join(output_dir, last_file),
            "label": flux_names[case['common_flux_type']],
        }
        print(f"  Last output: {last_file}")

# 绘制对比图
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

colors = {
    "common_flux_0": "r",
    "common_flux_1": "b",
    "common_flux_2": "g",
    "common_flux_3": "m",
    "common_flux_4": "c",
    "common_flux_5": "orange",
    "common_flux_6": "purple",
}

for case_name, info in results.items():
    color = colors[case_name]
    label_added = False

    # 读取CSV文件并绘制线段
    with open(info["file"], "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # 过滤空行
            if len(row) < 8:
                continue
            # k1: 每个单元2个点，共11列 (x0,x1,rho0,rho1,u0,u1,p0,p1,t0,t1,)
            if len(row) == 11:
                x0, x1, rho0, rho1, u0, u1, p0, p1, t0, t1 = map(float, row[:10])
                label = info["label"] if not label_added else ""
                axes[0].plot(
                    [x0, x1], [rho0, rho1], "-", linewidth=1, color=color, label=label
                )
                axes[1].plot(
                    [x0, x1], [u0, u1], "-", linewidth=1, color=color, label=label
                )
                axes[2].plot(
                    [x0, x1], [p0, p1], "-", linewidth=1, color=color, label=label
                )
                label_added = True
            # k2: 每个单元3个点，共16列
            if len(row) == 16:
                x0, x1, x2, rho0, rho1, rho2, u0, u1, u2, p0, p1, p2, t0, t1, t2 = map(
                    float, row[:15]
                )
                label = info["label"] if not label_added else ""
                axes[0].plot(
                    [x0, x1, x2],
                    [rho0, rho1, rho2],
                    "-",
                    linewidth=1,
                    color=color,
                    label=label,
                )
                axes[1].plot(
                    [x0, x1, x2],
                    [u0, u1, u2],
                    "-",
                    linewidth=1,
                    color=color,
                    label=label,
                )
                axes[2].plot(
                    [x0, x1, x2],
                    [p0, p1, p2],
                    "-",
                    linewidth=1,
                    color=color,
                    label=label,
                )
                label_added = True

axes[0].set_ylabel("rho", fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.7)
axes[0].set_xlim(-5, 5)

axes[1].set_ylabel("u", fontsize=12)
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.7)
axes[1].set_xlim(-5, 5)

axes[2].set_ylabel("p", fontsize=12)
axes[2].set_xlabel("x", fontsize=12)
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.7)
axes[2].set_xlim(-5, 5)

fig.suptitle(
    "Comparison: common_flux_type=0 vs 1 vs 2 vs 3 vs 4 vs 5 vs 6 (Last Output)", fontsize=14
)
plt.tight_layout()
plt.savefig("comparison_flux.png", dpi=300)
plt.close()

print("Comparison plot saved to comparison_flux.png")
