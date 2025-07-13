import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- 1. 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义物理常量和初始/终端条件 (参考论文 A10009072) ---
# 物理常量
G_const = 6.67430e-11  # 万有引力常数 (m^3/kg/s^2)
M_moon = 7.3477e22     # 月球质量 (kg)
mu_moon = G_const * M_moon # 月球引力常数 (m^3/s^2)
v_e = 2940.0           # 发动机等效排气速度, 对应比冲 (m/s)

# 初始状态 (SI 单位)
R_moon_dim = 1737013.0 # 月球平均半径 (m)
H0_dim = 15000.0       # 初始高度 (m)
r0_dim = R_moon_dim + H0_dim # 初始轨道半径 (m)
vr0_dim = 0.0          # 初始径向速度 (m/s)
vt0_dim = 1692.46      # 初始切向速度 (m/s)
m0_dim = 2400.0        # 初始质量 (kg)

# 终端状态 (SI 单位)
rf_dim = 2400.0+R_moon_dim
vtf_dim = 0.0          # 切向速度要求为0

# 推力约束 (SI 单位)
F_max_dim = 7500.0     # N

# --- 3. 优化问题构建 (CasADi) ---
N = 400  # 离散点数
opti = ca.Opti()

# 定义状态变量 X = [r, v_r, v_t, m]
X = opti.variable(4, N + 1)
r    = X[0, :]  # 径向距离 (m)
v_r  = X[1, :]  # 径向速度 (m/s)
v_t  = X[2, :]  # 切向速度 (m/s)
m    = X[3, :]  # 质量 (kg)

# 【修改】控制变量只剩下推力角 psi
psi = opti.variable(1, N)  # 推力角, 相对于切向 (rad)

# 定义总时间 T_final (s)
T_final = opti.variable()

# 目标函数：最大化最终质量 (等价于最小化飞行时间)
opti.minimize(-m[N])

# --- 4. 动力学约束 ---
dt = T_final / N

def dynamics(x_state, psi_k):
    """
    基于论文的动力学模型, 但推力F是常数
    x_state: [r, v_r, v_t, m]
    psi_k: 推力角控制
    """
    r_k, vr_k, vt_k, m_k = x_state[0], x_state[1], x_state[2], x_state[3]
    
    # 【修改】推力F固定为最大值
    F_k = F_max_dim
    
    # 动力学方程
    r_dot   = vr_k
    vr_dot  = (vt_k**2 / r_k) - (mu_moon / r_k**2) + (F_k / m_k) * ca.sin(psi_k)
    vt_dot  = -(vr_k * vt_k / r_k) - (F_k / m_k) * ca.cos(psi_k)
    m_dot   = -F_k / v_e
    
    return ca.vertcat(r_dot, vr_dot, vt_dot, m_dot)

# 使用RK4积分施加动力学约束
for k in range(N):
    k1 = dynamics(X[:, k], psi[k])
    k2 = dynamics(X[:, k] + dt / 2 * k1, psi[k])
    k3 = dynamics(X[:, k] + dt / 2 * k2, psi[k])
    k4 = dynamics(X[:, k] + dt * k3, psi[k])
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k+1] == x_next)

# --- 5. 边界和路径约束 ---
# 初始条件
opti.subject_to(r[0] == r0_dim)
opti.subject_to(v_r[0] == vr0_dim)
opti.subject_to(v_t[0] == vt0_dim)
opti.subject_to(m[0] == m0_dim)

# 终端条件
opti.subject_to(r[N] == rf_dim)
opti.subject_to(v_t[N] == vtf_dim)

# 路径约束
opti.subject_to(r >= rf_dim)
opti.subject_to(m >= 500.0)

# 控制变量约束
opti.subject_to(opti.bounded(-np.pi/2, psi, np.pi*0.5))

# 时间约束
opti.subject_to(opti.bounded(300, T_final, 1200))

# --- 6. 设置初始猜测值 ---
opti.set_initial(T_final, 400) # 最大推力下，时间会更短
# 猜测角度从纯制动(0)线性过渡到纯垂直(pi/2)
opti.set_initial(psi, np.linspace(0, np.pi/2, N))
opti.set_initial(r, np.linspace(r0_dim, rf_dim, N + 1))
opti.set_initial(v_r, np.linspace(vr0_dim, -50, N + 1)) # 猜测撞击速度会更大
opti.set_initial(v_t, np.linspace(vt0_dim, vtf_dim, N + 1))
opti.set_initial(m, np.linspace(m0_dim, m0_dim - 1000, N + 1)) # 猜测消耗更多燃料

# --- 7. 求解器设置 ---
s_opts = {"max_iter": 500, "print_level": 5}
opti.solver('ipopt', {}, s_opts)

# --- 8. 求解并后处理 ---
try:
    sol = opti.solve()
    print("\n--- 优化求解成功！ ---\n")
    
    # 提取最优解
    r_opt = sol.value(r)
    vr_opt = sol.value(v_r)
    vt_opt = sol.value(v_t)
    m_opt = sol.value(m)
    psi_opt = sol.value(psi)
    T_final_opt = sol.value(T_final)
    
    print(f"最优飞行时间: {T_final_opt:.2f} s")
    print(f"最终质量: {m_opt[-1]:.2f} kg")
    print(f"燃料消耗: {m0_dim - m_opt[-1]:.2f} kg")
    print(f"终端径向速度 (撞击速度): {vr_opt[-1]:.3f} m/s")
    print(f"终端切向速度: {vt_opt[-1]:.3f} m/s")
    
    # 绘图
    t_axis = np.linspace(0, T_final_opt, N + 1)
    t_ctrl_axis = np.linspace(0, T_final_opt, N)
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'全程最大推力轨迹优化 (飞行时间: {T_final_opt:.1f}s)', fontsize=16)
    
    # 高度变化
    plt.subplot(2, 3, 1)
    plt.plot(t_axis, (r_opt - R_moon_dim)/1000)
    plt.title('高度变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (km)')
    plt.grid(True)
    
    # 速度分量
    plt.subplot(2, 3, 2)
    plt.plot(t_axis, vr_opt, label='径向速度 (v_r)')
    plt.plot(t_axis, vt_opt, label='切向速度 (v_t)')
    plt.title('速度分量')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    # 质量变化
    plt.subplot(2, 3, 3)
    plt.plot(t_axis, m_opt)
    plt.title('质量变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('质量 (kg)')
    plt.grid(True)
    
    # 推力变化
    plt.subplot(2, 3, 4)
    plt.plot(t_ctrl_axis, np.full_like(t_ctrl_axis, F_max_dim), 'r')
    plt.title('推力变化 (恒定最大)')
    plt.xlabel('时间 (s)')
    plt.ylabel('推力 (N)')
    plt.ylim(0, F_max_dim * 1.1)
    plt.grid(True)
    
    # 推力角度变化
    plt.subplot(2, 3, 5)
    plt.plot(t_ctrl_axis, np.degrees(psi_opt))
    plt.title('推力角 ψ (相对切向)')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (°)')
    plt.grid(True)
    
    # 轨迹投影
    plt.subplot(2, 3, 6)
    d_theta_rad = (vt_opt[:-1] / r_opt[:-1]) * (T_final_opt / N)
    theta_traj_rad = np.cumsum(np.concatenate(([0], d_theta_rad)))
    x_traj = r_opt * np.cos(theta_traj_rad)
    y_traj = r_opt * np.sin(theta_traj_rad)
    plt.plot(x_traj/1000, y_traj/1000, label='轨迹')
    moon_angle = np.linspace(theta_traj_rad[0], theta_traj_rad[-1], 100)
    plt.plot(R_moon_dim/1000 * np.cos(moon_angle), R_moon_dim/1000 * np.sin(moon_angle), 'k--', label='月面')
    plt.title('轨迹投影 (月心系)')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
except Exception as e:
    print(f"\n--- 求解失败: {e} ---\n")
    if opti.debug:
        print(opti.debug.show_infeasibilities())