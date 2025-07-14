import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义物理常量和初始条件 ---
G_const = 6.67430e-11  # 万有引力常数 (m^3/kg/s^2)
M_moon = 7.3477e22     # 月球质量 (kg)
mu_moon = G_const * M_moon # 月球引力常数 (m^3/s^2)
v_e = 3038           # 发动机等效排气速度, 对应比冲 (m/s)
R_moon_dim = 1737013.0 # 月球平均半径 (m)
m0_dim = 3762        # 初始质量 (kg)
F_max_dim = 7500.0     # 最大推力 (N)
F_min_dim = 1500.0     # 最小推力 (N)

# 初始状态
H0_dim = 11700.0       # 初始高度 (m)
r0_dim = R_moon_dim + H0_dim # 初始轨道半径 (m)
vr0_dim = 0.0          # 初始径向速度 (m/s)
vt0_dim = 1762.83      # 初始切向速度 (m/s)

# --- 3. 优化问题构建 ---
opti = ca.Opti()

# 定义两个阶段的离散点数
N1 = 500  # 阶段一：主减速段
N2 = 300  # 阶段二：缓速下降段

# --- 阶段一：主减速段变量 ---
# 状态变量 [r, v_r, v_t, m]
X1 = opti.variable(4, N1 + 1)
r1, vr1, vt1, m1 = X1[0, :], X1[1, :], X1[2, :], X1[3, :]
# 控制变量：推力大小 F1 和推力角 psi (相对于切向)
F1 = opti.variable(1, N1)
psi1 = opti.variable(1, N1)
# 阶段时长
T1 = opti.variable()

# --- 阶段二：缓速下降段变量 ---
# 状态变量 [r, v_r, v_t, m]
X2 = opti.variable(4, N2 + 1)
r2, vr2, vt2, m2 = X2[0, :], X2[1, :], X2[2, :], X2[3, :]
# 控制变量：推力大小 F
F2 = opti.variable(1, N2)
# 阶段时长
T2 = opti.variable()

# --- 目标函数：最大化最终质量 (即最小化总燃料消耗) ---
opti.minimize(-m2[N2])

# --- 4. 动力学模型定义 ---
def dynamics(x_state, F_k, psi_k):
    """
    统一的动力学模型
    x_state: [r, v_r, v_t, m]
    F_k: 推力大小
    psi_k: 推力角 (相对于切向)
    """
    r_k, vr_k, vt_k, m_k = x_state[0], x_state[1], x_state[2], x_state[3]
    r_dot   = vr_k
    vr_dot  = (vt_k**2 / r_k) - (mu_moon / r_k**2) + (F_k / m_k) * ca.sin(psi_k)
    vt_dot  = -(vr_k * vt_k / r_k) - (F_k / m_k) * ca.cos(psi_k)
    m_dot   = -F_k / v_e
    return ca.vertcat(r_dot, vr_dot, vt_dot, m_dot)

# --- 5. 施加约束 ---
# 阶段一：动力学约束
dt1 = T1 / N1
for k in range(N1):
    opti.subject_to(opti.bounded(1.95,F1[k]/m1[k],2))

    x_dot = dynamics(X1[:, k], F1[k], psi1[k])
    opti.subject_to(X1[:, k+1] == X1[:, k] + dt1 * x_dot)

# 阶段二：动力学约束 (推力可变，推力角固定为竖直向下)
dt2 = T2 / N2
for k in range(N2):
    x_dot = dynamics(X2[:, k], F2[k], np.pi/2)
    opti.subject_to(X2[:, k+1] == X2[:, k] + dt2 * x_dot)

# 边界条件：初始和最终
opti.subject_to(r1[0] == r0_dim)
opti.subject_to(vr1[0] == vr0_dim)
opti.subject_to(vt1[0] == vt0_dim)
opti.subject_to(m1[0] == m0_dim)
opti.subject_to(r1[N1] == R_moon_dim + 3000)  # 阶段一结束时高度

opti.subject_to(r2[N2] == R_moon_dim+4) # 最终高度为月面
opti.subject_to(vr2[N2] == 0)         # 最终垂直速度为0 (软着陆)
opti.subject_to(vt2[N2] == 0)         # 最终水平速度为0 (软着陆)

# 【核心】阶段缝合约束 (Stitching Constraints)
# 强制阶段一的终点等于阶段二的起点
opti.subject_to(X1[:, N1] == X2[:, 0])

# 路径和控制变量的约束
opti.subject_to(opti.bounded(0, psi1, np.pi)) # 阶段一推力角
opti.subject_to(opti.bounded(F_min_dim, F1, F_max_dim)) # 阶段一推力大小约束
opti.subject_to(opti.bounded(0, F2, F_max_dim)) # 阶段二推力大小，允许为0
opti.subject_to(r1 >= R_moon_dim) # 路径高度约束
opti.subject_to(r2 >= R_moon_dim) # 路径高度约束
opti.subject_to(m2[N2] >= 500) # 保证最小干重

# 时间约束
#opti.subject_to(opti.bounded(300, T1, 1000))
opti.subject_to(opti.bounded(50, T2, 300))

# --- 6. 设置初始猜测值 ---
# 更合理的初始猜测值设置

# 阶段一：主减速段 (800秒左右)
T1_guess = 800
opti.set_initial(T1, T1_guess)

# 阶段一推力角：从接近切向(小角度)逐渐过渡到接近径向(大角度)
# 初期主要是切向减速，后期主要是径向减速
psi1_guess = np.linspace(0.2, np.pi/2 - 0.1, N1)  # 从11°到79°
opti.set_initial(psi1, psi1_guess)

# 阶段一推力：考虑推力/质量约束(1.95-2.0)，初期推力较大
F1_guess = np.linspace(F_max_dim * 0.95, F_max_dim * 0.8, N1)
opti.set_initial(F1, F1_guess)

# 阶段一轨道半径：从初始高度平滑下降到3000m
r1_guess = np.linspace(r0_dim, R_moon_dim + 3000, N1 + 1)
opti.set_initial(r1, r1_guess)

# 阶段一径向速度：从0逐渐增加到向下的速度
vr1_guess = np.linspace(vr0_dim, -50, N1 + 1)  # 更温和的径向速度变化
opti.set_initial(vr1, vr1_guess)

# 阶段一切向速度：从初始值逐渐减少，但保持一定的水平分量
vt1_guess = np.linspace(vt0_dim, 200, N1 + 1)  # 保持合理的水平速度
opti.set_initial(vt1, vt1_guess)

# 阶段一质量：根据推力和比冲估算燃料消耗
# 估算燃料消耗率：平均推力/比冲
avg_thrust_1 = np.mean(F1_guess)
fuel_rate_1 = avg_thrust_1 / v_e
total_fuel_1 = fuel_rate_1 * T1_guess
m1_guess = np.linspace(m0_dim, m0_dim - total_fuel_1, N1 + 1)
opti.set_initial(m1, m1_guess)

# 阶段二：缓速下降段 (150秒左右)
T2_guess = 150
opti.set_initial(T2, T2_guess)

# 阶段二推力：初期较小，后期增大以实现软着陆
# 采用更平滑的推力曲线
t2_normalized = np.linspace(0, 1, N2)
F2_guess = F_max_dim * 0.3 * (1 + 2 * t2_normalized**2)  # 二次增长
opti.set_initial(F2, F2_guess)

# 阶段二状态变量：确保与阶段一终点连续
r2_start = R_moon_dim + 3000
vr2_start = -50
vt2_start = 200
m2_start = m0_dim - total_fuel_1

# 阶段二轨道半径：从3000m平滑下降到月面
r2_guess = np.linspace(r2_start, R_moon_dim + 4, N2 + 1)
opti.set_initial(r2, r2_guess)

# 阶段二径向速度：从-50 m/s逐渐减速到0
vr2_guess = np.linspace(vr2_start, 0, N2 + 1)
opti.set_initial(vr2, vr2_guess)

# 阶段二切向速度：从200 m/s逐渐减速到0
vt2_guess = np.linspace(vt2_start, 0, N2 + 1)
opti.set_initial(vt2, vt2_guess)

# 阶段二质量：根据推力估算进一步的燃料消耗
avg_thrust_2 = np.mean(F2_guess)
fuel_rate_2 = avg_thrust_2 / v_e
total_fuel_2 = fuel_rate_2 * T2_guess
m2_guess = np.linspace(m2_start, m2_start - total_fuel_2, N2 + 1)
opti.set_initial(m2, m2_guess)

# 为求解器添加更宽松的初始约束
opti.subject_to(opti.bounded(600, T1, 1200))  # 添加T1的约束

# --- 7. 求解 ---
s_opts = {"max_iter": 5000, "print_level": 5, "tol": 1e-6, "acceptable_tol": 1e-4}
opti.solver('ipopt', {}, s_opts)

try:
    sol = opti.solve()
    print("\n--- 联合优化求解成功！ ---\n")
    
    # 提取转折点信息
    r_turn = sol.value(r1[N1])
    vr_turn = sol.value(vr1[N1])
    vt_turn = sol.value(vt1[N1])
    m_turn = sol.value(m1[N1])
    h_turn = r_turn - R_moon_dim
    
    print("--- 最优转折点信息 ---")
    print(f"转折点高度: {h_turn:.2f} m")
    print(f"转折点径向速度 (v_y): {vr_turn:.2f} m/s")
    print(f"转折点切向速度: {vt_turn:.2f} m/s")
    print(f"转折点质量: {m_turn:.2f} kg")
    
    # 提取总消耗
    m_final = sol.value(m2[N2])
    total_fuel_consumed = m0_dim - m_final
    print("\n--- 全局最优结果 ---")
    print(f"总燃料消耗: {total_fuel_consumed:.2f} kg")
    print(f"总飞行时间: {sol.value(T1) + sol.value(T2):.2f} s")
    
    # --- 准备绘图数据 ---
    t1_opt = np.linspace(0, sol.value(T1), N1 + 1)
    t2_opt = np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2 + 1)
    t_axis = np.concatenate((t1_opt, t2_opt[1:]))
    
    r_opt = np.concatenate((sol.value(r1), sol.value(r2)[1:]))
    vr_opt = np.concatenate((sol.value(vr1), sol.value(vr2)[1:]))
    vt_opt = np.concatenate((sol.value(vt1), sol.value(vt2)[1:]))
    m_opt = np.concatenate((sol.value(m1), sol.value(m2)[1:]))
    
    # 计算推力数据
    thrust_time = np.concatenate((
        np.linspace(0, sol.value(T1), N1),
        np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2)
    ))
    thrust_magnitude = np.concatenate((
        sol.value(F1),
        sol.value(F2)
    ))
    
    # 计算推力角数据
    angle_time = np.concatenate((
        np.linspace(0, sol.value(T1), N1),
        np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2)
    ))
    angle_degrees = np.concatenate((
        np.degrees(sol.value(psi1)),
        np.full(N2, 90)
    ))
    
    # 计算轨迹投影数据
    dt_traj = np.diff(t_axis)
    d_theta_rad = (vt_opt[:-1] / r_opt[:-1]) * dt_traj
    theta_traj_rad = np.cumsum(np.concatenate(([0], d_theta_rad)))
    x_traj = r_opt * np.cos(theta_traj_rad)
    y_traj = r_opt * np.sin(theta_traj_rad)
    
    # --- 导出数据到Excel ---
    print("\n--- 导出数据到Excel文件 ---")
    
    # 创建包含所有数据的字典
    max_len = max(len(t_axis), len(thrust_time), len(angle_time))
    
    # 状态数据 (时间序列)
    state_data = {
        '时间_s': np.pad(t_axis, (0, max_len - len(t_axis)), constant_values=np.nan),
        '高度_m': np.pad((r_opt - R_moon_dim), (0, max_len - len(r_opt)), constant_values=np.nan),
        '径向速度_ms': np.pad(vr_opt, (0, max_len - len(vr_opt)), constant_values=np.nan),
        '切向速度_ms': np.pad(vt_opt, (0, max_len - len(vt_opt)), constant_values=np.nan),
        '质量_kg': np.pad(m_opt, (0, max_len - len(m_opt)), constant_values=np.nan),
        '轨迹X坐标_m': np.pad(x_traj, (0, max_len - len(x_traj)), constant_values=np.nan),
        '轨迹Y坐标_m': np.pad(y_traj, (0, max_len - len(y_traj)), constant_values=np.nan)
    }
    
    # 控制数据 (推力和角度)
    control_data = {
        '控制时间_s': np.pad(thrust_time, (0, max_len - len(thrust_time)), constant_values=np.nan),
        '推力大小_N': np.pad(thrust_magnitude, (0, max_len - len(thrust_magnitude)), constant_values=np.nan),
        '推力角度_deg': np.pad(angle_degrees, (0, max_len - len(angle_degrees)), constant_values=np.nan)
    }
    
    # 合并数据
    all_data = {**state_data, **control_data}
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 导出到Excel文件
    output_filename = "lunar_landing_joint_a_data.xlsx"
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # 主数据表
        df.to_excel(writer, sheet_name='轨迹数据', index=False)
        
        # 关键参数表
        summary_data = {
            '参数': ['总燃料消耗_kg', '总飞行时间_s', '转折点高度_m', '转折点径向速度_ms', 
                    '转折点切向速度_ms', '转折点质量_kg', '阶段一时长_s', '阶段二时长_s'],
            '数值': [total_fuel_consumed, sol.value(T1) + sol.value(T2), h_turn, vr_turn, 
                    vt_turn, m_turn, sol.value(T1), sol.value(T2)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='关键参数', index=False)
        
        # 转折点信息表
        transition_data = {
            '阶段': ['阶段一终点', '阶段二起点'],
            '高度_m': [h_turn, h_turn],
            '径向速度_ms': [vr_turn, vr_turn],
            '切向速度_ms': [vt_turn, vt_turn],
            '质量_kg': [m_turn, m_turn],
            '时间_s': [sol.value(T1), sol.value(T1)]
        }
        transition_df = pd.DataFrame(transition_data)
        transition_df.to_excel(writer, sheet_name='转折点信息', index=False)
    
    print(f"数据已成功导出到: {output_filename}")
    print(f"包含工作表: 轨迹数据, 关键参数, 转折点信息")
    
    # --- 绘图 ---
    t1_opt = np.linspace(0, sol.value(T1), N1 + 1)
    t2_opt = np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2 + 1)
    t_axis = np.concatenate((t1_opt, t2_opt[1:]))
    
    r_opt = np.concatenate((sol.value(r1), sol.value(r2)[1:]))
    vr_opt = np.concatenate((sol.value(vr1), sol.value(vr2)[1:]))
    vt_opt = np.concatenate((sol.value(vt1), sol.value(vt2)[1:]))
    m_opt = np.concatenate((sol.value(m1), sol.value(m2)[1:]))
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'两阶段联合优化轨迹 (最优转折高度: {h_turn:.1f} m)', fontsize=16)
    
    plt.subplot(2, 3, 1)
    plt.plot(t_axis, (r_opt - R_moon_dim) / 1000)
    plt.axvline(x=sol.value(T1), color='r', linestyle='--', label=f'转折点 T={sol.value(T1):.1f}s')
    plt.title('高度变化'), plt.xlabel('时间 (s)'), plt.ylabel('高度 (km)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t_axis, vr_opt, label='径向速度')
    plt.plot(t_axis, vt_opt, label='切向速度')
    plt.axvline(x=sol.value(T1), color='r', linestyle='--')
    plt.title('速度分量'), plt.xlabel('时间 (s)'), plt.ylabel('速度 (m/s)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t_axis, m_opt)
    plt.axvline(x=sol.value(T1), color='r', linestyle='--')
    plt.title('质量变化'), plt.xlabel('时间 (s)'), plt.ylabel('质量 (kg)'), plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.step(np.linspace(0, sol.value(T1), N1), sol.value(F1), where='post', label='阶段一 (固定加速度)')
    plt.step(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), sol.value(F2), where='post', label='阶段二 (可变推力)')
    plt.title('推力剖面'), plt.xlabel('时间 (s)'), plt.ylabel('推力 (N)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(np.linspace(0, sol.value(T1), N1), np.degrees(sol.value(psi1)), label='阶段一')
    plt.plot(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), np.full(N2, 90), label='阶段二 (固定90°)')
    plt.title('推力角 ψ (相对切向)'), plt.xlabel('时间 (s)'), plt.ylabel('角度 (°)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 6)
    dt_traj = np.diff(t_axis)
    d_theta_rad = (vt_opt[:-1] / r_opt[:-1]) * dt_traj
    theta_traj_rad = np.cumsum(np.concatenate(([0], d_theta_rad)))
    x_traj = r_opt * np.cos(theta_traj_rad)
    y_traj = r_opt * np.sin(theta_traj_rad)
    plt.plot(x_traj/1000, y_traj/1000, label='轨迹')
    plt.plot(x_traj[N1]/1000, y_traj[N1]/1000, 'r*', markersize=10, label='最优转折点')
    plt.title('轨迹投影'), plt.xlabel('X (km)'), plt.ylabel('Y (km)'), plt.axis('equal'), plt.grid(True), plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

except Exception as e:
    print(f"\n--- 求解失败: {e} ---\n")