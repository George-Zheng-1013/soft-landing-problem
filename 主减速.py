import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义物理常量和初始/终端条件 (有单位) ---
G= 6.67430e-11  # 万有引力常数 (m^3/kg/s^2)
M=7.34767309e22  # 月球质量 (kg)
Isp = 316.0      # 发动机比冲 (s)
g_e = 9.81     # 地球海平面重力加速度 (m/s^2)
mu=2940

# 初始状态 (SI 单位)
R0_dim = 1753000.0   # m (初始轨道半径)
V_y0_dim = 0.0       # m/s (径向速度)
V_x0_dim = 1692.0    # m/s (切向速度)
M0_dim = 750.0       # kg

# 终端状态 (SI 单位)
Rf_dim = 1741030.0   # m
Vy_f_max = 1.5       # m/s (径向速度约束)
Vx_f_max = 3.8       # m/s (切向速度约束)

# 推力约束 (SI 单位)
F_max_dim = 1750.0   # N
F_min_dim = 0.0      # N

# --- 3. 无量纲化 (Scaling) ---
DU = R0_dim
MU = M0_dim
TU = np.sqrt(DU**3 / mu)
VU = DU / TU
FU = MU * DU / TU**2
Rf_norm = Rf_dim / DU
Vy0_norm = V_y0_dim / VU
Vx0_norm = V_x0_dim / VU
Vy_f_norm_max = Vy_f_max / VU
Vx_f_norm_max = Vx_f_max / VU
F_max_norm = F_max_dim / FU
F_min_norm = F_min_dim / FU
C_thrust = (TU * FU) / (MU * Isp * g_e)

# --- 4. 优化问题构建 ---
N = 400
opti = ca.Opti()

# 定义状态变量 X_norm (无量纲)
X_norm = opti.variable(5, N + 1)
r= X_norm[0, :]  # 无量纲半径
v_x= X_norm[1, :]  # 无量纲切向速度
v_y= X_norm[2, :]  # 无量纲径向速度
m= X_norm[3, :]  # 无量纲质量
v= X_norm[4, :]  # 无量纲速度

# 定义控制变量 U_ctrl (无量纲推力 + 角度)
U_ctrl = opti.variable(2, N)
f_norm = U_ctrl[0, :]
Theta  = U_ctrl[1, :]

# 定义总时间 T_final (物理单位 s)
T_final = opti.variable()

# 目标函数：最大化最终质量
opti.minimize(-m[N])

# --- 5. 动力学约束 ---
dt_norm = (T_final / TU) / N

def dynamics_norm_2d(X_state, U_control):
    r, v_y, v_x, m, v = X_state
    f, Theta = U_control
    r_dot = v_y
    m_dot = f / mu
    v_dot = f / (M0_dim - m_dot * t)
    v_y_dot=-G*M / r**2 + v_x**2 / r + v_dot*np.sin(Theta)
    v_x_dot = v_dot * np.cos(Theta) - v_y * v_x / r
    return ca.vertcat(r_dot, v_y_dot, v_x_dot, m_dot, v_dot)

for k in range(N):
    k1 = dynamics_norm_2d(X_norm[:, k], U_ctrl[:, k])
    k2 = dynamics_norm_2d(X_norm[:, k] + dt_norm / 2 * k1, U_ctrl[:, k])
    k3 = dynamics_norm_2d(X_norm[:, k] + dt_norm / 2 * k2, U_ctrl[:, k])
    k4 = dynamics_norm_2d(X_norm[:, k] + dt_norm * k3, U_ctrl[:, k])
    X_next = X_norm[:, k] + dt_norm / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X_norm[:, k+1] == X_next)

# --- 6. 边界和路径约束 (与之前相同) ---
opti.subject_to(r[0] == 1.0)
opti.subject_to(v_y[0] == Vy0_norm)
opti.subject_to(v_x[0] == Vx0_norm)
opti.subject_to(m[0] == 1.0)
opti.subject_to(r[N] == Rf_norm)
opti.subject_to(ca.fabs(v_y[N]) <= Vy_f_norm_max)
opti.subject_to(ca.fabs(v_x[N]) <= Vx_f_norm_max)
opti.subject_to(opti.bounded(F_min_norm, f_norm, F_max_norm))
opti.subject_to(r >= Rf_norm)
opti.subject_to(m >= 100.0 / MU)
opti.subject_to(opti.bounded(500, T_final, 800))
opti.subject_to(opti.bounded(-np.pi, Theta, 0))

# 遗传算法找到的最优解 (N=50)
T_final_guess = 790.2278
f_guess_ga = np.array([0.6702,1.0898,0.918 ,1.056 ,0.8027,1.1581,0.7835,0.7112,0.9954,1.0826,
 1.244 ,0.7488,1.2345,1.246 ,0.4875,1.1859,0.8387,1.1612,0.9443,0.5768,
 1.1293,1.2643,1.3281,0.9447,1.2372,1.0965,1.1969,1.1387,1.2866,1.0324,
 1.1521,1.1302,0.8652,1.4291,1.3702,1.2897,1.1933,1.1567,1.37  ,1.0867,
 1.4624,1.4509,1.3279,1.0311,1.4624,1.4613,1.0783,1.4624,1.3269,1.4624])
theta_guess_ga = np.array([-1.7701,-1.6486,-1.2898,-1.3523,-1.0958,-1.5215,-1.2352,-1.1459,-1.303 ,
 -1.3449,-1.2663,-1.5481,-1.3056,-1.1996,-1.6984,-1.4512,-1.2433,-1.2134,
 -1.4197,-1.5555,-1.2374,-1.326 ,-1.2748,-0.9982,-1.2032,-1.2612,-1.2766,
 -1.2747,-1.2256,-1.2757,-1.2371,-0.9881,-1.2096,-1.2097,-1.1492,-1.1448,
 -0.848 ,-0.955 ,-1.0875,-1.1208,-0.915 ,-1.0213,-0.9279,-0.8395,-1.0503,
 -0.6862,-0.9348,-1.0166,-1.1408,-1.3125])

N_ga = 50
f_interp = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_ga), f_guess_ga)
theta_interp = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_ga), theta_guess_ga)

opti.set_initial(T_final, T_final_guess)
opti.set_initial(f_norm, f_interp)
opti.set_initial(Theta, theta_interp)

opti.set_initial(r, np.linspace(1, Rf_norm, N + 1))
opti.set_initial(v_y, np.linspace(Vy0_norm, -Vy_f_norm_max, N + 1))
opti.set_initial(v_x, np.linspace(Vx0_norm, Vx_f_norm_max, N + 1))
opti.set_initial(m, np.linspace(1, (M0_dim - 200) / MU, N + 1)) # 猜测消耗200kg燃料

# --- 8. 求解器设置 ---
s_opts = {"max_iter": 3000, "print_level": 5}
opti.solver('ipopt', {}, s_opts)

# --- 9. 求解并后处理 ---
try:
    sol = opti.solve()
    print("\n--- 优化求解成功！ ---\n")
    
    # 提取最优解
    r_opt = sol.value(r) * DU
    vy_opt = sol.value(v_y) * VU
    vx_opt = sol.value(v_x) * VU
    m_opt = sol.value(m) * MU
    f_opt = sol.value(f_norm) * FU
    Theta_opt = sol.value(Theta)
    T_final_opt = sol.value(T_final)
    
    print(f"最优飞行时间: {T_final_opt:.2f} s")
    print(f"最终质量: {m_opt[-1]:.2f} kg")
    print(f"燃料消耗: {M0_dim - m_opt[-1]:.2f} kg")
    print(f"终端速度: V_y={vy_opt[-1]:.3f} m/s, V_x={vx_opt[-1]:.3f} m/s")
    
    # 绘图
    t = np.linspace(0, T_final_opt, N + 1)
    t_control = np.linspace(0, T_final_opt, N)
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'月球软着陆轨迹优化 (飞行时间: {T_final_opt:.1f}s)', fontsize=16)
    
    plt.subplot(2, 3, 1)
    plt.plot(t, (r_opt - Rf_dim)/1000)
    plt.title('高度变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (km)')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(t, vy_opt, label='V（y轴速度）')
    plt.plot(t, vx_opt, label='U（x轴速度）')
    plt.title('速度分量')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(t, m_opt)
    plt.title('质量变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('质量 (kg)')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.step(t_control, f_opt, where='post')
    plt.title('推力变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('推力 (N)')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.step(t_control, Theta_opt, where='post')
    plt.title('推力角度')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (rad)')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    d_theta_rad = (vx_opt[:-1] / r_opt[:-1]) * (T_final_opt / N)
    theta_traj_rad = np.cumsum(np.concatenate(([0], d_theta_rad)))
    x_traj = r_opt * np.cos(theta_traj_rad)
    y_traj = r_opt * np.sin(theta_traj_rad)
    plt.plot(x_traj/1000, y_traj/1000, label='轨迹')
    moon_angle = np.linspace(theta_traj_rad[0], theta_traj_rad[-1], 100)
    plt.plot(Rf_dim/1000 * np.cos(moon_angle), Rf_dim/1000 * np.sin(moon_angle), 'k--', label='月面')
    plt.title('轨迹投影')
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