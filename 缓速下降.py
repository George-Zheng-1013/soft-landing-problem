# 1. 导入库并设置参数
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# --- 问题参数设定 ---
g_m = 1.62
g_E = 9.81

Isp = 311.0

F_min = 0.0
F_max = 7500.0

y0 = 2400.0
vy0 = -70.0
m0 = 2500.0

yf = 0.0
vyf = 0.0

T = 100.0        
N = 400
dt = T / N

# --- CasADi 优化问题构建 ---
opti = ca.Opti()

X = opti.variable(3, N + 1)
y_pos   = X[0, :]
vy      = X[1, :]
m       = X[2, :]

F_thrust = opti.variable(1, N)

opti.minimize(-m[N])

#定义动力学约束 (使用前向欧拉法进行离散化)
for k in range(N):
    X_k = X[:, k]
    U_k = F_thrust[k]

    x_dot = ca.vertcat(
        X_k[1],                                 # vy
        U_k / X_k[2] - g_m,                     # ay = F/m - g_m
        -U_k / (Isp * g_E)                      # m_dot
    )

    # 离散化约束：X_{k+1} = X_k + dt * f(X_k, U_k)
    X_next = X_k + dt * x_dot
    opti.subject_to(X[:, k+1] == X_next)

opti.subject_to(y_pos[0] == y0)
opti.subject_to(vy[0] == vy0)
opti.subject_to(m[0] == m0)

opti.subject_to(y_pos[N] == yf)
opti.subject_to(vy[N] == vyf)

opti.subject_to(opti.bounded(F_min, F_thrust, F_max)) # 推力约束
opti.subject_to(y_pos >= 0)                         # 高度不能为负
opti.subject_to(m >= 1500)                          # 放宽质量约束

#设置更好的初始猜测值
y_guess = []
vy_guess = []
m_guess = []
for i in range(N + 1):
    t = i * dt
    # 假设前半段自由落体，后半段减速
    if t < T/2:
        y_est = y0 + vy0 * t - 0.5 * g_m * t**2
        vy_est = vy0 - g_m * t
    else:
        # 后半段线性减速到0
        t_brake = t - T/2
        y_mid = y0 + vy0 * (T/2) - 0.5 * g_m * (T/2)**2
        vy_mid = vy0 - g_m * (T/2)
        y_est = y_mid + vy_mid * t_brake - 0.5 * (vy_mid/(T/2)) * t_brake**2
        vy_est = vy_mid - (vy_mid/(T/2)) * t_brake
    
    y_guess.append(max(0, y_est))
    vy_guess.append(vy_est)
    m_guess.append(m0 - (m0 - 1500) * t/T)  # 线性质量减少

opti.set_initial(y_pos, y_guess)
opti.set_initial(vy, vy_guess)
opti.set_initial(m, m_guess)

# 推力初始猜测 - 前半段小推力，后半段大推力
F_guess = []
for i in range(N):
    t = i * dt
    if t < T/2:
        F_guess.append(F_max * 0.2)  # 前半段小推力
    else:
        F_guess.append(F_max * 0.8)  # 后半段大推力

opti.set_initial(F_thrust, F_guess)

opti.solver('ipopt', {}, {
    'print_level': 5,
    'max_iter': 1000,
    'tol': 1e-6,
    'acceptable_tol': 1e-4,
    'acceptable_constr_viol_tol': 1e-3
})

try:
    sol = opti.solve()
    
    y_opt = sol.value(y_pos)
    vy_opt = sol.value(vy)
    m_opt = sol.value(m)
    F_opt = sol.value(F_thrust)
    
    print(f"\n=== 优化结果 ===")
    print(f"初始质量: {m_opt[0]:.2f} kg")
    print(f"最终质量: {m_opt[-1]:.2f} kg")
    print(f"消耗燃料: {m_opt[0] - m_opt[-1]:.2f} kg")
    print(f"飞行时间: {T:.2f} s")
    print(f"最终高度: {y_opt[-1]:.2f} m")
    print(f"最终速度: {vy_opt[-1]:.2f} m/s")
    
    time = np.linspace(0, T, N + 1)
    time_u = np.linspace(0, T, N)
    
    plt.figure(figsize=(12, 10))
    plt.suptitle("月球软着陆优化结果", fontsize=16)
    
    plt.subplot(2, 2, 1)
    plt.plot(time, y_opt, 'b-', linewidth=2)
    plt.title("高度随时间变化")
    plt.xlabel("时间 (s)")
    plt.ylabel("高度 (m)")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(time, vy_opt, 'r-', linewidth=2)
    plt.title("垂直速度随时间变化")
    plt.xlabel("时间 (s)")
    plt.ylabel("速度 (m/s)")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(time, m_opt, 'g-', linewidth=2)
    plt.title("质量随时间变化")
    plt.xlabel("时间 (s)")
    plt.ylabel("质量 (kg)")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.step(time_u, F_opt, where='post', linewidth=2)
    plt.title("推力随时间变化")
    plt.xlabel("时间 (s)")
    plt.ylabel("推力 (N)")
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
except Exception as e:
    print(f"求解失败: {e}")
    print("\n尝试调试...")
    
    print("当前变量值:")
    try:
        print(f"高度范围: [{opti.debug.value(y_pos).min():.2f}, {opti.debug.value(y_pos).max():.2f}]")
        print(f"速度范围: [{opti.debug.value(vy).min():.2f}, {opti.debug.value(vy).max():.2f}]")
        print(f"质量范围: [{opti.debug.value(m).min():.2f}, {opti.debug.value(m).max():.2f}]")
        print(f"推力范围: [{opti.debug.value(F_thrust).min():.2f}, {opti.debug.value(F_thrust).max():.2f}]")
    except:
        print("无法获取调试信息")