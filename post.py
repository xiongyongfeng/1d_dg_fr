import matplotlib.pyplot as plt
import csv

# 创建图形和坐标轴
plt.figure(figsize=(10, 6))
ax = plt.gca()

# 读取CSV文件并绘制线段
with open('result.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        # 过滤空行并确保有6个数据点
        if len(row) < 6: 
            continue
        
        # 提取前4个坐标值（忽略最后2个）
        x0, x1, y0, y1 = map(float, row[:4])
        
        # 绘制线段
        plt.plot([x0, x1], [y0, y1], 'b-', linewidth=1.5, marker='o')

# 设置图形属性
plt.title('Segment Visualization from CSV Data', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1)  # 根据数据范围设置X轴
plt.ylim(-0.1, 1.1)  # 根据数据范围设置Y轴

# 显示图形
plt.tight_layout()
plt.savefig('segments_plot.png', dpi=300)  # 保存高清图像
plt.show()