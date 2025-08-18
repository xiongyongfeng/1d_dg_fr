import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime  # 用于添加时间戳（可选）
import pandas as pd

# 设置输入和输出目录
input_dir = './soln_fr_tvdRK1_fine'  # 替换为你的CSV文件夹路径
output_dir =  input_dir + '_png'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取当前时间戳（可选，用于文件名）
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# 遍历目录中的所有CSV文件[1,4](@ref)
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # 构造完整文件路径
        csv_path = os.path.join(input_dir, filename)
        print(f"processing file {csv_path}")
        # 创建图形和坐标轴
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        df = pd.read_csv('exactsod.csv')
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label='rho', 
         linestyle='-', linewidth=2)  # 实线
        plt.plot(df.iloc[:, 0], df.iloc[:, 2], label='u', 
         linestyle='--', linewidth=2)  # 虚线
        plt.plot(df.iloc[:, 0], df.iloc[:, 3], label='p', 
         linestyle='-.', linewidth=2)  # 点划线

        # 读取CSV文件并绘制线段
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                # 过滤空行并确保有6个数据点
                if len(row) < 8: 
                    continue
                
                # 提取前4个坐标值（忽略最后2个）
                x0, x1, rho0, rho1,u0,u1,p0,p1 = map(float, row[:8])
                # flag = float(row[4])
                # line_color = 'red' if flag > 0 else 'blue'
                
                # 绘制线段
                plt.plot([x0, x1], [rho0, rho1], '-', linewidth=1.5,color='r')
                plt.plot([x0, x1], [u0, u1], '-', linewidth=1.5,color='b')
                plt.plot([x0, x1], [p0, p1], '-', linewidth=1.5,color='g')
        
        

        # 设置图形属性
        plt.title(f'Segment Visualization: {filename}', fontsize=14)  # 使用文件名作为标题
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-5, 5)
        
        # 生成输出文件名（同CSV文件名但扩展名为.png）
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存高清图像（300 DPI）[7,8](@ref)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # 关闭图形释放内存[8](@ref)
        plt.close()

print(f"处理完成！所有图表已保存至: {output_dir}")