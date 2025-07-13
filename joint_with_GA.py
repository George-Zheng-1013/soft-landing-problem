import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- 1. 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义物理常量和初始条件 ---
G_const = 6.67430e-11
M_moon = 7.3477e22
mu_moon = G_const * M_moon
v_e = 3038
R_moon_dim = 1737013.0
m0_dim = 3762
F_max_dim = 7500.0
F_min_dim = 1500.0

# 初始状态
H0_dim = 11700.0
r0_dim = R_moon_dim + H0_dim
vr0_dim = 0.0
vt0_dim = 1762.83

# --- 3. 优化问题构建 ---
opti = ca.Opti()

# 定义两个阶段的离散点数
N1 = 500  # 阶段一：主减速段
N2 = 300  # 阶段二：缓速下降段

# --- 阶段一：主减速段变量 ---
X1 = opti.variable(4, N1 + 1)
r1, vr1, vt1, m1 = X1[0, :], X1[1, :], X1[2, :], X1[3, :]
psi1 = opti.variable(1, N1)
T1 = opti.variable()

# --- 阶段二：缓速下降段变量 ---
X2 = opti.variable(4, N2 + 1)
r2, vr2, vt2, m2 = X2[0, :], X2[1, :], X2[2, :], X2[3, :]
F2 = opti.variable(1, N2)
psi2 = opti.variable(1, N2)
T2 = opti.variable()

# --- 目标函数：最大化最终质量 ---
opti.minimize(-m2[N2])

# --- 4. 动力学模型定义 ---
def dynamics(x_state, F_k, psi_k):
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
    k1 = dynamics(X1[:, k], F_max_dim, psi1[k])
    k2 = dynamics(X1[:, k] + dt1 / 2 * k1, F_max_dim, psi1[k])
    k3 = dynamics(X1[:, k] + dt1 / 2 * k2, F_max_dim, psi1[k])
    k4 = dynamics(X1[:, k] + dt1 * k3, F_max_dim, psi1[k])
    x_next = X1[:, k] + dt1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X1[:, k+1] == x_next)

# 阶段二：动力学约束
dt2 = T2 / N2
for k in range(N2):
    k1 = dynamics(X2[:, k], F2[k], psi2[k])
    k2 = dynamics(X2[:, k] + dt2 / 2 * k1, F2[k], psi2[k])
    k3 = dynamics(X2[:, k] + dt2 / 2 * k2, F2[k], psi2[k])
    k4 = dynamics(X2[:, k] + dt2 * k3, F2[k], psi2[k])
    x_next = X2[:, k] + dt2 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X2[:, k+1] == x_next)

# 边界条件
opti.subject_to(r1[0] == r0_dim)
opti.subject_to(vr1[0] == vr0_dim)
opti.subject_to(vt1[0] == vt0_dim)
opti.subject_to(m1[0] == m0_dim)
opti.subject_to(r1[N1] == R_moon_dim + 3000)

opti.subject_to(r2[N2] == R_moon_dim + 4)
opti.subject_to(vr2[N2] == 0)
opti.subject_to(vt2[N2] == 0)

# 阶段缝合约束
opti.subject_to(X1[:, N1] == X2[:, 0])

# 路径和控制变量的约束
opti.subject_to(opti.bounded(0, psi1, np.pi))
opti.subject_to(opti.bounded(0, F2, F_max_dim))
opti.subject_to(psi2 == np.pi/2)
opti.subject_to(r1 >= R_moon_dim)
opti.subject_to(r2 >= R_moon_dim)
opti.subject_to(m2[N2] >= 500)

# 时间约束
opti.subject_to(opti.bounded(300, T1, 700))
opti.subject_to(opti.bounded(50, T2, 200))


# --- 6. 从CUDA遗传算法设置初始猜测值 ---
# [GA生成的初始值] 阶段一飞行时间
T1_guess = 700.000000

# [GA生成的初始值] 阶段一推力角序列
psi1_guess_ga = np.array([2.91934 ,3.141593,3.141593,3.13385 ,3.141593,2.830652,3.104269,3.141593,3.141593,2.692453,3.141593,0.      ,3.141593,
 0.      ,2.52705 ,3.141593,3.141593,3.141593,2.65433 ,3.141593,2.540848,2.975149,3.141593,3.141593,3.141593,3.141593,
 3.141593,3.141593,0.615321,0.      ,3.141593,3.141593,2.794217,3.118908,0.      ,2.596475,0.011965,3.141593,2.781018,
 1.831919,2.229005,3.141593,3.141593,3.141593,3.141593,3.141593,1.899636,3.141593,2.76046 ,3.141593,3.141593,3.141593,
 2.292971,3.141593,3.141593,3.093468,0.      ,3.141593,2.999439,2.775214,3.07867 ,3.141593,0.      ,2.666642,0.      ,
 3.141593,3.055211,2.984853,0.      ,3.141593,0.958526,1.524813,3.141593,2.78424 ,3.141593,2.843801,0.      ,3.141593,
 0.      ,2.082339,3.141593,3.141593,3.141593,1.619032,2.367922,3.141593,3.141593,3.141593,3.141593,3.141593,3.141593,
 2.443135,2.292083,3.141593,3.141593,3.141593,3.141593,3.141593,3.141593,3.141593,0.      ,0.751901,0.      ,1.608544,
 3.028373,3.141593,3.141593,3.141593,3.141593,2.199115,3.141593,3.141593,2.528445,0.442379,3.141593,3.041407,3.103338,
 0.528007,3.141593,3.141593,3.141593,0.058585,3.141593,3.141593,2.15756 ,3.141593,1.975465,0.      ,3.073353,3.141593,
 2.884223,2.271748,0.      ,3.045809,3.141593,0.866053,2.623547,3.141593,0.      ,2.485902,0.      ,1.993287,0.199083,
 2.842545,2.216257,0.      ,1.531408,3.141593,3.141593,3.134771])

# --- 7. 应用初始猜测值 ---
print("正在应用GA生成的初始猜测值...")

# 设置阶段一的初始值
opti.set_initial(T1, T1_guess)

# 对推力角进行插值，以匹配CasADi的离散点数N1
# 这是因为GA脚本中的N可能与此处的N1不同
N_ga = len(psi1_guess_ga)
psi1_guess_interp = np.interp(
    np.linspace(0, 1, N1),      # 新的坐标点 (0到1, N1个点)
    np.linspace(0, 1, N_ga),    # 原始坐标点 (0到1, N_ga个点)
    psi1_guess_ga               # 原始数据
)
opti.set_initial(psi1, psi1_guess_interp)

# 为状态变量提供一个线性的初始猜测，这通常足够好
r1_guess = np.linspace(r0_dim, R_moon_dim + 3000, N1 + 1)
vr1_guess = np.linspace(vr0_dim, -80, N1 + 1)
vt1_guess = np.linspace(vt0_dim, 100, N1 + 1)
m1_guess = np.linspace(m0_dim, m0_dim - (m0_dim - 500) * 0.6, N1 + 1)

opti.set_initial(r1, r1_guess)
opti.set_initial(vr1, vr1_guess)
opti.set_initial(vt1, vt1_guess)
opti.set_initial(m1, m1_guess)

# 为阶段二提供简单的初始猜测
opti.set_initial(T2, 100)
opti.set_initial(F2, F_max_dim * 0.6)
opti.set_initial(psi2, np.pi/2)
# 阶段二的起点与阶段一的终点猜测保持一致
opti.set_initial(r2, np.linspace(r1_guess[-1], R_moon_dim + 4, N2 + 1))
opti.set_initial(vr2, np.linspace(vr1_guess[-1], 0, N2 + 1))
opti.set_initial(vt2, np.linspace(vt1_guess[-1], 0, N2 + 1))
opti.set_initial(m2, np.linspace(m1_guess[-1], m1_guess[-1] - 200, N2 + 1))

# --- 8. 求解 ---
s_opts = {"max_iter": 5000, "print_level": 5, "tol": 1e-6, "acceptable_tol": 1e-4}
opti.solver('ipopt', {}, s_opts)

try:
    sol = opti.solve()
    print("\n--- 联合优化求解成功！ ---\n")
    
    # 结果提取和绘图代码 (与您原始脚本相同)
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
    
    m_final = sol.value(m2[N2])
    total_fuel_consumed = m0_dim - m_final
    print("\n--- 全局最优结果 ---")
    print(f"总燃料消耗: {total_fuel_consumed:.2f} kg")
    print(f"总飞行时间: {sol.value(T1) + sol.value(T2):.2f} s")
    
    t1_opt = np.linspace(0, sol.value(T1), N1 + 1)
    t2_opt = np.linspace(sol.value(T1), sol.value(T1) + sol.value(T2), N2 + 1)
    t_axis = np.concatenate((t1_opt, t2_opt[1:]))
    
    r_opt = np.concatenate((sol.value(r1), sol.value(r2)[1:]))
    vr_opt = np.concatenate((sol.value(vr1), sol.value(vr2)[1:]))
    vt_opt = np.concatenate((sol.value(vt1), sol.value(vt2)[1:]))
    m_opt = np.concatenate((sol.value(m1), sol.value(m2)[1:]))
    
    plt.figure(figsize=(18, 10))
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
    f1_opt = np.full(N1, F_max_dim)
    f2_opt = sol.value(F2)
    t_f_axis = np.concatenate((t1_opt[:-1], t2_opt[:-1]))
    f_opt = np.concatenate((f1_opt, f2_opt))
    plt.step(t_f_axis, f_opt, where='post')
    plt.axvline(x=sol.value(T1), color='r', linestyle='--')
    plt.title('推力剖面'), plt.xlabel('时间 (s)'), plt.ylabel('推力 (N)'), plt.grid(True)

    plt.subplot(2, 3, 5)
    psi1_opt = sol.value(psi1)
    psi2_opt = sol.value(psi2)
    psi_opt = np.concatenate((psi1_opt, psi2_opt))
    plt.plot(np.linspace(0, sol.value(T1), N1), np.degrees(psi1_opt))
    plt.plot(np.linspace(sol.value(T1), sol.value(T1)+sol.value(T2), N2), np.degrees(psi2_opt))
    plt.axvline(x=sol.value(T1), color='r', linestyle='--')
    plt.title('推力角 ψ (相对切向)'), plt.xlabel('时间 (s)'), plt.ylabel('角度 (°)'), plt.grid(True)

    plt.subplot(2, 3, 6)
    dt_traj = np.diff(t_axis)
    # 确保vt_opt和r_opt的长度与dt_traj匹配
    vt_for_theta = vt_opt[:-1]
    r_for_theta = r_opt[:-1]
    d_theta_rad = (vt_for_theta / r_for_theta) * dt_traj
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
    print("--- 提示：如果求解失败，可以尝试：")
    print("1. 重新运行CUDA脚本，看是否能找到一个适应度值更低的解。")
    print("2. 调整 `CUDA_Initial_Guess_Generator.py` 中的惩罚权重或DE算法参数。")
    print("3. 检查CasADi模型中的约束是否过于严格。")