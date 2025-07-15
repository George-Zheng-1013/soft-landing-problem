import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# --- 1. 设置绘图样式和中文字体 ---
# --- 1. Setup plot style and Chinese fonts ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# --- 2. 定义从用户文件中获取的常量和参数 (已更新) ---
# --- 2. Define constants and parameters from user files (UPDATED) ---
# 从 joint_F.py 获取
# From joint_F.py
F_max_dim = 7500.0     # 最大推力 (N) / Max thrust (N)
v_e = 3038.0           # 发动机等效排气速度 (m/s) / Engine equivalent exhaust velocity (m/s)
mu_moon = 6.67430e-11 * 7.3477e22 # 月球引力常数 (m^3/s^2) / Moon's gravitational constant (m^3/s^2)
R_moon_dim = 1737013.0 # 月球平均半径 (m) / Mean radius of the Moon (m)
m_dry_estimate = 1900.0 # 估计的着陆器干重 (kg) / Estimated lander dry mass (kg)

# --- 更新：使用用户提供的最新优化结果 ---
# --- UPDATE: Use the latest optimization results provided by the user ---
m_turn = 2067.73        # 转折点质量 (kg) / Mass at turning point (kg)
h_turn = 3000.0         # 转折点高度 (m) / Altitude at turning point (m)
vr_turn = -60.0        # 转折点径向速度 (m/s) / Radial velocity at turning point (m/s)


# --- 3. 计算派生常量 ---
# --- 3. Calculate derived constants ---
g = 1.62
beta = F_max_dim / v_e # 最大推力时的质量流率 (kg/s) / Mass flow rate at max thrust (kg/s)
print(f"--- 工程参数计算结果 (已更新) ---")
print(f"转折点质量 m_turn: {m_turn} kg")
print(f"转折点径向速度 vr_turn: {vr_turn} m/s")
print(f"局部重力加速度 g: {g:.4f} m/s^2")
print(f"质量流率 beta: {beta:.4f} kg/s")

# --- 4. 定义切换曲线方程 ---
# --- 4. Define switching curve equations ---
def get_switching_curve(m_start, t_burn_max):
    """
    根据推导的参数方程计算切换曲线
    Calculates the switching curve based on the derived parametric equations.
    """
    t_b = np.linspace(0.01, t_burn_max, 500) # 从0.01开始以避免log(1)问题 / Start from 0.01 to avoid log(1) issues
    
    v_switch = g * t_b + v_e * np.log(1 - (beta / m_start) * t_b)
    h_switch = (v_e * m_start / beta) * np.log(m_start / (m_start - beta * t_b + 1e-9)) \
               - v_e * t_b - 0.5 * g * t_b**2
               
    return v_switch, h_switch

# --- 5. 计算自由落体轨迹 ---
# --- 5. Calculate free-fall trajectory ---
def get_freefall_trajectory(h_start, vr_start, max_time=60):
    """计算从初始点开始的自由落体轨迹"""
    """Calculates the free-fall trajectory from a starting point."""
    t_ff = np.linspace(0, max_time, 500)
    h_ff = h_start + vr_start * t_ff - 0.5 * g * t_ff**2
    vr_ff = vr_start - g * t_ff
    return vr_ff, h_ff


# --- 6. 计算和绘图 ---
# --- 6. Calculation and Plotting ---
# 使用更新后的参数计算
# Calculate using updated parameters
t_b_max = (m_turn - m_dry_estimate) / beta
v_curve, h_curve = get_switching_curve(m_turn, t_b_max)
vr_ff_traj, h_ff_traj = get_freefall_trajectory(h_turn, vr_turn)

# --- 新增：数值计算交点 ---
# --- NEW: Numerically find the intersection point ---
# 创建KDTree以便快速查找最近点
# Create KDTree for efficient nearest point search
switch_curve_points = np.vstack((v_curve, h_curve)).T
tree = KDTree(switch_curve_points)

# 查找自由落体轨迹上每个点到切换曲线的最近距离
# Find the nearest distance from each point on the free-fall trajectory to the switching curve
free_fall_points = np.vstack((vr_ff_traj, h_ff_traj)).T
distances, indices = tree.query(free_fall_points)

# 找到距离最小的点，即为交点
# The point with the minimum distance is the intersection
intersection_idx = np.argmin(distances)
intersection_vr = vr_ff_traj[intersection_idx]
intersection_h = h_ff_traj[intersection_idx]

# --- 新增：实际切换点数据 ---
# --- NEW: Actual switching point data ---
# 根据实际优化结果设置的切换点
actual_switch_vr = -86.185  # 实际切换点径向速度 (m/s)
actual_switch_h = 1818.925  # 实际切换点高度 (m)


# 开始绘图
# Start plotting
fig, ax = plt.subplots()

# 绘制理论切换曲线
ax.plot(v_curve, h_curve, 'r-', linewidth=3, label='理论切换曲线')

# 绘制自由落体轨迹
ax.plot(vr_ff_traj, h_ff_traj, 'c--', linewidth=2, label=f'从新转折点({vr_turn:.1f}m/s,{h_turn:.0f}m)的自由落体')

# 标记起点（转折点）
ax.plot(vr_turn, h_turn, 'o', color='cyan', markersize=12, markeredgecolor='k', label='阶段二起点(新转折点)')

# 标记计算出的交点（理论最优切换点）
ax.plot(intersection_vr, intersection_h, 'm*', markersize=10, markeredgecolor='k', 
        label=f'理论最优切换点')

# --- 新增：标记实际切换点 ---
# --- NEW: Mark actual switching point ---
ax.plot(actual_switch_vr, actual_switch_h, 's', color='orange', markersize=8, markeredgecolor='k', 
        label=f'实际切换点')

# 标记软着陆目标点
ax.plot(0, 4, 'go', markersize=12, markeredgecolor='k', label='软着陆目标(0m/s,4m)')

# --- 新增：添加坐标值标注 ---
# --- NEW: Add coordinate value annotations ---
# 理论切换点坐标标注
ax.annotate(f'({intersection_vr:.1f}, {intersection_h:.0f})', 
            xy=(intersection_vr, intersection_h), 
            xytext=(intersection_vr-40, intersection_h+400),
            fontsize=11, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='magenta', alpha=0.6),
            arrowprops=dict(arrowstyle='->', color='magenta', lw=1))

# 实际切换点坐标标注
ax.annotate(f'({actual_switch_vr:.1f}, {actual_switch_h:.0f})', 
            xy=(actual_switch_vr, actual_switch_h), 
            xytext=(actual_switch_vr+30, actual_switch_h-400),
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.6),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1))

# 设置图表属性
ax.set_title('月球着陆第二阶段：理论切换曲线与计算结果对比')
ax.set_xlabel('垂直速度 (m/s) [向下为负]')
ax.set_ylabel('高度 (m)')
ax.legend()
ax.grid(True)

# 调整坐标轴
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlim(-140, 2)
ax.set_ylim(-50, 3500)
ax.invert_xaxis() 
plt.tight_layout()
plt.show()

