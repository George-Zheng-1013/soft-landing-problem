import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 1. 设置参数 ---
# 数据所在的文件夹
data_dir = "simulation_results" 
# 字体设置，确保能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 查找所有生成的数据文件 ---
# 使用glob找到所有符合命名规则的Excel文件
f_files = sorted(glob.glob(os.path.join(data_dir, "data_*_F_*.xlsx")))
a_files = sorted(glob.glob(os.path.join(data_dir, "data_*_a_*.xlsx")))

# 检查文件是否成对
if len(f_files) != len(a_files):
    print("警告：恒F策略和恒a策略的数据文件数量不匹配！")
    print(f"恒F文件: {len(f_files)}, 恒a文件: {len(a_files)}")
    # 尝试基于文件名进行匹配
    f_basenames = {os.path.basename(f).split('_F_')[0] for f in f_files}
    a_basenames = {os.path.basename(f).split('_a_')[0] for f in a_files}
    common_bases = list(f_basenames.intersection(a_basenames))
    f_files = [f for f in f_files if os.path.basename(f).split('_F_')[0] in common_bases]
    a_files = [f for f in a_files if os.path.basename(f).split('_a_')[0] in common_bases]
    print(f"已找到 {len(f_files)} 对匹配的文件。")

# --- 3. 循环处理每一对数据并绘图 ---
for f_file, a_file in zip(f_files, a_files):
    try:
        # --- a. 读取数据 ---
        data_f = pd.read_excel(f_file, sheet_name='轨迹数据')
        data_a = pd.read_excel(a_file, sheet_name='轨迹数据')
        
        # 提取参数名称用于图表标题
        # e.g., from 'data_1_F_标准着陆器.xlsx' to '标准着陆器'
        param_name = os.path.basename(f_file).split('_F_')[-1].replace('.xlsx', '')

        # --- b. 数据清洗 ---
        # 删除所有值都为NaN的行，这些是填充产生的
        data_f.dropna(how='all', inplace=True)
        data_a.dropna(how='all', inplace=True)
        
        # --- c. 提取关键信息 ---
        # 最终剩余质量
        final_mass_f = data_f['质量_kg'].iloc[-1]
        final_mass_a = data_a['质量_kg'].iloc[-1]
        
        # 初始质量
        initial_mass_f = data_f['质量_kg'].iloc[0]
        initial_mass_a = data_a['质量_kg'].iloc[0]
        
        # 燃料消耗
        fuel_consumed_f = initial_mass_f - final_mass_f
        fuel_consumed_a = initial_mass_a - final_mass_a
        
        # --- d. 绘制对比图 ---
        plt.figure(figsize=(12, 7))
        
        # 绘制恒F策略的质量变化曲线
        plt.plot(data_f['时间_s'], data_f['质量_kg'], label=f'恒F策略 (消耗: {fuel_consumed_f:.2f} kg)', color='red', linewidth=2.5)
        
        # 绘制恒a策略的质量变化曲线
        plt.plot(data_a['时间_s'], data_a['质量_kg'], label=f'恒a策略 (消耗: {fuel_consumed_a:.2f} kg)', color='blue', linestyle='--', linewidth=2.5)
        
        # 在曲线末端标注最终质量
        plt.text(data_f['时间_s'].iloc[-1], final_mass_f, f'{final_mass_f:.2f} kg', fontsize=12, color='red', va='bottom', ha='right')
        plt.text(data_a['时间_s'].iloc[-1], final_mass_a, f'{final_mass_a:.2f} kg', fontsize=12, color='blue', va='top', ha='right')
        
        # --- e. 美化图表 ---
        plt.title(f'质量-时间对比: {param_name}\n恒F策略 vs 恒a策略', fontsize=16, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('着陆器质量 (kg)', fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # --- f. 保存图像 ---
        # 创建一个专门存放图像的文件夹
        plot_dir = "comparison_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # 定义图像文件名
        plot_filename = os.path.join(plot_dir, f'comparison_{param_name}.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"已保存图像: {plot_filename}")
        
        plt.show()

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保已成功运行 `run_simulations.py`。")
    except Exception as e:
        print(f"处理文件 {f_file} 和 {a_file} 时发生错误: {e}")

print("\n--- 所有对比图已生成完毕 ---")
