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
# 控制变量：推力角多项式系数 (5个系数)
psi_coeffs = opti.variable(5, 1)  # 多项式系数 [a0, a1, a2, a3, a4]
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
# 阶段一：动力学约束 (推力固定为最大值，推力角使用多项式)
dt1 = T1 / N1
for k in range(N1):
    # 计算当前时刻的推力角 (使用多项式)
    t_norm = k / N1  # 归一化时间 [0, 1]
    psi_k = psi_coeffs[0] + psi_coeffs[1]*t_norm + psi_coeffs[2]*t_norm**2 + psi_coeffs[3]*t_norm**3 + psi_coeffs[4]*t_norm**4
    
    x_dot = dynamics(X1[:, k], F_max_dim, psi_k)
    x_next = X1[:, k] + dt1 * x_dot
    opti.subject_to(X1[:, k+1] == x_next)

# 阶段二：动力学约束 (推力可变，推力角固定为竖直向下)
dt2 = T2 / N2
for k in range(N2):
    x_dot = dynamics(X2[:, k], F2[k], np.pi/2)
    x_next = X2[:, k] + dt2 * x_dot
    opti.subject_to(X2[:, k+1] == x_next)

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
for k in range(N1):
    t_norm = k / N1
    psi_k = psi_coeffs[0] + psi_coeffs[1]*t_norm + psi_coeffs[2]*t_norm**2 + psi_coeffs[3]*t_norm**3 + psi_coeffs[4]*t_norm**4

opti.subject_to(opti.bounded(0, F2, F_max_dim)) # 阶段二推力大小，允许为0
opti.subject_to(r1 >= R_moon_dim) # 路径高度约束
opti.subject_to(r2 >= R_moon_dim) # 路径高度约束
opti.subject_to(m2[N2] >= 500) # 保证最小干重

# 时间约束
# opti.subject_to(opti.bounded(300, T1, 2000))
# opti.subject_to(opti.bounded(50, T2, 500))

# --- 6. 设置初始猜测值 ---
# 为整个复杂问题提供一个合理的初始猜测是至关重要的
# 阶段一猜测
opti.set_initial(T1, 1000)
# 初始化多项式系数，使推力角从小角度逐渐增大到π/2
opti.set_initial(psi_coeffs, np.array([0.2, 0.3, 0.2, 0.1, 0.2]))  # 基本上是线性增长
opti.set_initial(r1, np.linspace(r0_dim, R_moon_dim + 3000, N1 + 1))
opti.set_initial(vr1, np.linspace(vr0_dim, -80, N1 + 1))
opti.set_initial(vt1, np.linspace(vt0_dim, 100, N1 + 1))
opti.set_initial(m1, np.linspace(m0_dim, m0_dim - 1100, N1 + 1))

# 阶段二猜测 (起点必须与阶段一终点猜测一致)
opti.set_initial(T2, 100)
opti.set_initial(F2, np.concatenate([np.zeros(N2//2), np.full(N2//2, F_max_dim)]))
opti.set_initial(r2, np.linspace(R_moon_dim + 3000, R_moon_dim, N2 + 1))
opti.set_initial(vr2, np.linspace(-80, 0, N2 + 1))
opti.set_initial(vt2, 0)
opti.set_initial(m2, np.linspace(m0_dim - 1100, m0_dim - 1300, N2 + 1))

# --- 7. 求解 ---
s_opts = {"max_iter": 5000, "print_level": 5, "tol": 1e-6, "acceptable_tol": 1e-4}
opti.solver('ipopt', {}, s_opts)

try:
    sol = opti.solve()
    print("\n--- 联合优化求解成功！ ---\n")
    
    # 提取多项式系数
    psi_coeffs_opt = sol.value(psi_coeffs)
    print("--- 推力角多项式系数 ---")
    for i, coeff in enumerate(psi_coeffs_opt):
        print(f"a{i}: {coeff:.6f}")
    
    # 提取转折点信息
    r_turn = sol.value(r1[N1])
    vr_turn = sol.value(vr1[N1])
    vt_turn = sol.value(vt1[N1])
    m_turn = sol.value(m1[N1])
    h_turn = r_turn - R_moon_dim
    
    print("\n--- 最优转折点信息 ---")
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
    
    # --- 计算推力角数据 (使用多项式) ---
    t1_norm = np.linspace(0, 1, N1)
    psi1_values = np.zeros(N1)
    for k in range(N1):
        t_norm = t1_norm[k]
        psi1_values[k] = (psi_coeffs_opt[0] + psi_coeffs_opt[1]*t_norm + 
                         psi_coeffs_opt[2]*t_norm**2 + psi_coeffs_opt[3]*t_norm**3 + 
                         psi_coeffs_opt[4]*t_norm**4)
    
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
        np.full(N1, F_max_dim),
        sol.value(F2)
    ))
    
    # 计算推力角数据
    angle_time = np.concatenate((
        np.linspace(0, sol.value(T1), N1),
        np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2)
    ))
    angle_degrees = np.concatenate((
        np.degrees(psi1_values),
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
    output_filename = "lunar_landing_optimization_data.xlsx"
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
        
        # 多项式系数表
        poly_data = {
            '系数': [f'a{i}' for i in range(5)],
            '数值': psi_coeffs_opt
        }
        poly_df = pd.DataFrame(poly_data)
        poly_df.to_excel(writer, sheet_name='推力角多项式系数', index=False)
        
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
    print(f"包含工作表: 轨迹数据, 关键参数, 推力角多项式系数, 转折点信息")
    
    # --- 绘图 ---
    t1_opt = np.linspace(0, sol.value(T1), N1 + 1)
    t2_opt = np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2 + 1)
    t_axis = np.concatenate((t1_opt, t2_opt[1:]))
    
    r_opt = np.concatenate((sol.value(r1), sol.value(r2)[1:]))
    vr_opt = np.concatenate((sol.value(vr1), sol.value(vr2)[1:]))
    vt_opt = np.concatenate((sol.value(vt1), sol.value(vt2)[1:]))
    m_opt = np.concatenate((sol.value(m1), sol.value(m2)[1:]))
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'两阶段联合优化轨迹 (多项式拟合推力角, 最优转折高度: {h_turn:.1f} m)', fontsize=16)
    
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
    plt.step(np.linspace(0, sol.value(T1), N1), np.full(N1, F_max_dim), where='post', label='阶段一 (最大推力)')
    plt.step(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), sol.value(F2), where='post', label='阶段二 (可变推力)')
    plt.title('推力剖面'), plt.xlabel('时间 (s)'), plt.ylabel('推力 (N)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(np.linspace(0, sol.value(T1), N1), np.degrees(psi1_values), label='阶段一 (多项式拟合)')
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