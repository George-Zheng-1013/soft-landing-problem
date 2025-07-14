import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

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
# 控制变量：推力角 psi (相对于切向)
psi1 = opti.variable(1, N1)
# 阶段时长
T1 = opti.variable()

# --- 阶段二：缓速下降段变量 ---
# 状态变量 [r, v_r, v_t, m]
X2 = opti.variable(4, N2 + 1)
r2, vr2, vt2, m2 = X2[0, :], X2[1, :], X2[2, :], X2[3, :]
# 控制变量：推力大小 F 和推力角 psi
F2 = opti.variable(1, N2)
psi2 = opti.variable(1, N2)
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
# 阶段一：动力学约束 (推力固定为最大值)
dt1 = T1 / N1
for k in range(N1):
    k1 = dynamics(X1[:, k], F_max_dim, psi1[k])
    k2 = dynamics(X1[:, k] + dt1 / 2 * k1, F_max_dim, psi1[k])
    k3 = dynamics(X1[:, k] + dt1 / 2 * k2, F_max_dim, psi1[k])
    k4 = dynamics(X1[:, k] + dt1 * k3, F_max_dim, psi1[k])
    x_next = X1[:, k] + dt1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X1[:, k+1] == x_next)

# 阶段二：动力学约束 (推力可变)
dt2 = T2 / N2
for k in range(N2):
    k1 = dynamics(X2[:, k], F2[k], psi2[k])
    k2 = dynamics(X2[:, k] + dt2 / 2 * k1, F2[k], psi2[k])
    k3 = dynamics(X2[:, k] + dt2 / 2 * k2, F2[k], psi2[k])
    k4 = dynamics(X2[:, k] + dt2 * k3, F2[k], psi2[k])
    x_next = X2[:, k] + dt2 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X2[:, k+1] == x_next)

# 边界条件：初始和最终
opti.subject_to(r1[0] == r0_dim)
opti.subject_to(vr1[0] == vr0_dim)
opti.subject_to(vt1[0] == vt0_dim)
opti.subject_to(m1[0] == m0_dim)
opti.subject_to(r1[N1] == R_moon_dim + 30)  # 阶段一结束时高度

opti.subject_to(r2[N2] == R_moon_dim+4) # 最终高度为月面
opti.subject_to(vr2[N2] == 0)         # 最终垂直速度为0 (软着陆)
opti.subject_to(vt2[N2] == 0)         # 最终水平速度为0 (软着陆)

# 【核心】阶段缝合约束 (Stitching Constraints)
# 强制阶段一的终点等于阶段二的起点
opti.subject_to(X1[:, N1] == X2[:, 0])

# 路径和控制变量的约束
opti.subject_to(opti.bounded(0, psi1, np.pi)) # 阶段一推力角
opti.subject_to(opti.bounded(0, F2, F_max_dim)) # 阶段二推力大小，允许为0
#opti.subject_to(psi2 == np.pi/2) # 阶段二推力角固定为竖直向下
angle_range = 15 * np.pi / 180 
opti.subject_to(opti.bounded(np.pi/2 - angle_range, psi2, np.pi/2 + angle_range))
opti.subject_to(r1 >= R_moon_dim) # 路径高度约束
opti.subject_to(r2 >= R_moon_dim) # 路径高度约束
opti.subject_to(m2[N2] >= 500) # 保证最小干重

# 时间约束
opti.subject_to(opti.bounded(300, T1, 700))
opti.subject_to(opti.bounded(50, T2, 200))

# --- 6. 设置初始猜测值 ---
# 为整个复杂问题提供一个合理的初始猜测是至关重要的
# 阶段一猜测
opti.set_initial(T1, 450)
opti.set_initial(psi1, np.linspace(0.1, np.pi/2, N1))
opti.set_initial(r1, np.linspace(r0_dim, R_moon_dim + 2000, N1 + 1))
opti.set_initial(vr1, np.linspace(vr0_dim, -80, N1 + 1))
opti.set_initial(vt1, np.linspace(vt0_dim, 100, N1 + 1))
opti.set_initial(m1, np.linspace(m0_dim, m0_dim - 1100, N1 + 1))

# 阶段二猜测 (起点必须与阶段一终点猜测一致)
opti.set_initial(T2, 100)
opti.set_initial(F2, F_max_dim * 0.8)
opti.set_initial(psi2, np.pi/2)
opti.set_initial(r2, np.linspace(R_moon_dim + 2000, R_moon_dim, N2 + 1))
opti.set_initial(vr2, np.linspace(-80, 0, N2 + 1))
opti.set_initial(vt2, np.linspace(100, 0, N2 + 1))
opti.set_initial(m2, np.linspace(m0_dim - 1100, m0_dim - 1300, N2 + 1))

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
    plt.step(np.linspace(0, sol.value(T1), N1), np.full(N1, F_max_dim), where='post', label='阶段一 (最大推力)')
    plt.step(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), sol.value(F2), where='post', label='阶段二 (可变推力)')
    plt.title('推力剖面'), plt.xlabel('时间 (s)'), plt.ylabel('推力 (N)'), plt.grid(True), plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(np.linspace(0, sol.value(T1), N1), np.degrees(sol.value(psi1)), label='阶段一')
    plt.plot(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), np.degrees(sol.value(psi2)), label='阶段二')
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