import numpy as np
from scipy.optimize import differential_evolution
import time

# --- 1. 复制主脚本的物理模型和参数 ---
# 物理常量
G_const = 6.67430e-11
M_moon = 7.3477e22
mu_moon = G_const * M_moon
v_e = 2940.0

# 初始状态 (SI 单位)
R_moon_dim = 1737013.0
H0_dim = 15000.0
r0_dim = R_moon_dim + H0_dim
vr0_dim = 0.0
vt0_dim = 1692.46
m0_dim = 2400.0

# 终端状态 (SI 单位)
# 修正后的终端高度：月球半径 + 2.4km
rf_dim = R_moon_dim + 2400.0
vtf_dim = 0.0

# 推力 (恒定最大)
F_max_dim = 7500.0

# 离散点数 (GA中可以使用较少的点以加速)
N = 50

# --- 2. 定义适应度函数 (目标函数) ---
def objective_function(chromosome):
    """
    计算一个控制策略（染色体）的成本。
    染色体结构: [psi_0, ..., psi_{N-1}, T_final]
    """
    # 2.1 解码染色体
    psi_controls = chromosome[0:N]
    T_final = chromosome[N]

    # 2.2 模拟轨迹 (使用简单的欧拉积分)
    dt = T_final / N
    
    # 初始状态
    r, v_r, v_t, m = r0_dim, vr0_dim, vt0_dim, m0_dim
    
    for i in range(N):
        psi_k = psi_controls[i]
        F_k = F_max_dim

        # 动力学方程
        r_dot   = v_r
        vr_dot  = (v_t**2 / r) - (mu_moon / r**2) + (F_k / m) * np.sin(psi_k)
        vt_dot  = -(v_r * v_t / r) - (F_k / m) * np.cos(psi_k)
        m_dot   = -F_k / v_e
        
        # 改进：使用RK4积分替代欧拉积分
        def rk4_step(r, v_r, v_t, m, psi, dt):
            def dynamics(r, v_r, v_t, m, psi):
                F_k = F_max_dim
                r_dot = v_r
                vr_dot = (v_t**2 / r) - (mu_moon / r**2) + (F_k / m) * np.sin(psi)
                vt_dot = -(v_r * v_t / r) - (F_k / m) * np.cos(psi)
                m_dot = -F_k / v_e
                return r_dot, vr_dot, vt_dot, m_dot
            
            k1 = dynamics(r, v_r, v_t, m, psi)
            k2 = dynamics(r + dt/2*k1[0], v_r + dt/2*k1[1], v_t + dt/2*k1[2], m + dt/2*k1[3], psi)
            k3 = dynamics(r + dt/2*k2[0], v_r + dt/2*k2[1], v_t + dt/2*k2[2], m + dt/2*k2[3], psi)
            k4 = dynamics(r + dt*k3[0], v_r + dt*k3[1], v_t + dt*k3[2], m + dt*k3[3], psi)
            
            r_new = r + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            v_r_new = v_r + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            v_t_new = v_t + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
            m_new = m + dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
            
            return r_new, v_r_new, v_t_new, m_new
        
        r, v_r, v_t, m = rk4_step(r, v_r, v_t, m, psi_k, dt)
        
        # 增加数值稳定性检查
        if r < R_moon_dim or m <= 0 or np.isnan(r) or np.isinf(r):
            return 1e9

    # 2.3 计算成本 - 改进版本
    cost = -m  # 主要目标：最大化最终质量
    
    # 惩罚项权重调整
    W_pos = 1e-3   # 位置误差权重 (降低)
    W_vel = 1e-1   # 速度误差权重
    W_vr = 1e-2    # 径向速度惩罚 (新增)
    
    # 位置误差惩罚
    pos_error = abs(r - rf_dim)
    cost += W_pos * pos_error**2
    
    # 切向速度误差惩罚
    vel_t_error = abs(v_t - vtf_dim)
    cost += W_vel * vel_t_error**2
    
    # 径向速度惩罚 (软着陆需要小的径向速度)
    cost += W_vr * v_r**2
    
    return cost

# --- 3. 设置遗传算法的边界 ---
bounds = []
# 推力角 psi 的边界
for _ in range(N):
    bounds.append((-np.pi/2, np.pi/2))
# 总时间 T_final 的边界
bounds.append((300, 1200))

# --- 4. 运行差分进化算法 ---
print("--- 开始运行差分进化算法进行全局搜索 (这可能需要几分钟)... ---")
start_time = time.time()

# popsize: 种群大小, 越大搜索越广但越慢
# maxiter: 最大迭代代数
# tol: 收敛容差
# workers=-1: 使用所有CPU核心并行计算
result = differential_evolution(objective_function, bounds, 
                                strategy='best1bin', maxiter=500, 
                                popsize=25, tol=0.01, 
                                disp=True, updating='deferred', workers=-1)

end_time = time.time()
print(f"--- 搜索完成！耗时: {end_time - start_time:.2f} 秒 ---")

# --- 5. 打印最优结果 ---
best_chromosome = result.x
best_fitness = result.fun

print(f"\n找到的最佳适应度值 (成本): {best_fitness:.6f}")
print("\n--- 请将以下初始猜测值复制到您的 `主减速_o1.py` 脚本中 ---")

# 格式化输出，方便复制
psi_guess_ga = best_chromosome[0:N]
T_final_guess = best_chromosome[N]

print("\n# 初始猜测值 (来自遗传算法)")
print(f"opti.set_initial(T_final, {T_final_guess:.4f})")

# 构造一个可复制的 numpy 数组字符串
print("# 对于控制变量，需要进行插值")
print(f"psi_guess_ga = np.array({np.array2string(psi_guess_ga, precision=4, separator=',')})")
print("# CasADi脚本中的N值 (例如N=400)")
print("# N_casadi = 400")
print("# N_ga = 50")
print("# psi_interp = np.interp(np.linspace(0, 1, N_casadi), np.linspace(0, 1, N_ga), psi_guess_ga)")
print("# opti.set_initial(psi, psi_interp)")