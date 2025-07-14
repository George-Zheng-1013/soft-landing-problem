import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import time

# --- 1. 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 系统参数 (与joint_F.py完全一致) ==========
G_const = 6.67430e-11
M_moon = 7.3477e22
mu_moon = G_const * M_moon
R_moon_dim = 1737013.0

m0_dim = 3762.0
F_max_dim = 7500.0
F_min_dim = 1500.0
v_e = 3038.0

# 初始状态 (极坐标)
H0_dim = 11700.0
r0_dim = R_moon_dim + H0_dim
vr0_dim = 0.0
vt0_dim = 1762.83
X0 = np.array([r0_dim, vr0_dim, vt0_dim, m0_dim])

# 终端约束
r_target = R_moon_dim + 4.0
vr_target = 0.0
vt_target = 0.0

# ========== 2. CVP (控制变量参数化) 设置 ==========
# 阶段一：主减速段 (最大推力，优化推力角)
N1_segments = 6
psi1_bounds = (0.0, np.pi)
T1_bounds = (300.0, 700.0)

# 阶段二：缓速下降段 (可变推力，固定垂直角度)
N2_segments = 4
F2_bounds = (0.0, F_max_dim)
T2_bounds = (50.0, 200.0)

# 转折点高度也作为优化变量
r_transition_bounds = (R_moon_dim + 2000, R_moon_dim + 5000)

# 全局缓存，避免重复积分计算
last_params_vec = None
last_solution = None

# ========== 3. 核心函数定义 ==========

def dynamics(r, vr, vt, m, F, psi):
    """
    极坐标动力学方程 - 与joint_F.py完全一致。
    """
    # 增加安全检查
    if m <= 0 or r <= R_moon_dim * 0.9: # 允许轻微穿透，给求解器容错
        return np.array([0, 0, 0, 1e6]) # 返回一个惩罚性的质量变化率

    r_dot = vr
    vr_dot = (vt**2 / r) - (mu_moon / r**2) + (F / m) * np.sin(psi)
    vt_dot = -(vr * vt / r) - (F / m) * np.cos(psi)
    m_dot = -F / v_e
    
    return np.array([r_dot, vr_dot, vt_dot, m_dot])

def simulate_trajectory(params_vec):
    """
    “打靶”函数：根据给定的参数向量，积分出完整轨迹。
    """
    global last_params_vec, last_solution
    # 如果参数和上次相同，直接返回缓存的结果，极大提高效率
    if last_params_vec is not None and np.array_equal(params_vec, last_params_vec):
        return last_solution

    # 1. 解包参数
    psi1_params, F2_params, T1, T2, r_transition = unpack_params(params_vec)
    
    # 2. 模拟阶段一
    dt1 = T1 / N1_segments
    def stage1_dynamics(t, X):
        k = min(int(t / dt1), N1_segments - 1) if dt1 > 0 else 0
        return dynamics(X[0], X[1], X[2], X[3], F_max_dim, psi1_params[k])
    
    sol1 = solve_ivp(stage1_dynamics, [0, T1], X0, method='RK45', rtol=1e-7, atol=1e-7)
    
    if not sol1.success or sol1.y[3, -1] < 0: # 如果积分失败或质量为负
        return None

    # 3. 模拟阶段二
    X1_final = sol1.y[:, -1].copy()
    # 强制阶段一的终点高度为我们正在优化的转折点高度
    # 这是打靶法处理阶段衔接的一种方式，虽然不如直接配置法严谨
    X1_final[0] = r_transition 
    
    dt2 = T2 / N2_segments
    def stage2_dynamics(t, X):
        k = min(int((t - T1) / dt2), N2_segments - 1) if dt2 > 0 else 0
        return dynamics(X[0], X[1], X[2], X[3], F2_params[k], np.pi / 2)
        
    sol2 = solve_ivp(stage2_dynamics, [T1, T1 + T2], X1_final, method='RK45', rtol=1e-7, atol=1e-7)

    if not sol2.success or sol2.y[3, -1] < 0:
        return None

    # 4. 缓存并返回结果
    last_params_vec = np.copy(params_vec)
    last_solution = {'sol1': sol1, 'sol2': sol2, 'params': params_vec}
    return last_solution

def objective(params_vec):
    """目标函数：最大化最终质量"""
    sim_result = simulate_trajectory(params_vec)
    if sim_result is None:
        return 1e6  # 积分失败，返回一个巨大的惩罚值
    
    final_mass = sim_result['sol2'].y[3, -1]
    return -final_mass

def constraints(params_vec):
    """约束函数：终端状态必须满足着陆条件"""
    sim_result = simulate_trajectory(params_vec)
    if sim_result is None:
        return np.full(3, 1e6) # 积分失败，返回巨大的约束违反值

    final_state = sim_result['sol2'].y[:, -1]
    
    # 返回一个包含所有等式约束残差的列表
    # 求解器会尝试让这个列表中的所有值都变为0
    return [
        final_state[0] - r_target,
        final_state[1] - vr_target,
        final_state[2] - vt_target
    ]

def unpack_params(x):
    """将优化变量向量解析为有意义的参数"""
    psi1 = x[:N1_segments]
    F2 = x[N1_segments : N1_segments + N2_segments]
    T1 = x[N1_segments + N2_segments]
    T2 = x[N1_segments + N2_segments + 1]
    r_transition = x[N1_segments + N2_segments + 2]
    return psi1, F2, T1, T2, r_transition

# ========== 4. 主优化流程 ==========
def cvp_optimization():
    # 初始化参数猜测
    initial_psi1 = np.linspace(0.2, np.pi / 2, N1_segments)
    initial_F2 = np.linspace(F_max_dim * 0.8, F_max_dim * 0.5, N2_segments)
    initial_T1 = 450.0
    initial_T2 = 120.0
    initial_r_transition = R_moon_dim + 3000.0
    
    x0 = np.concatenate([initial_psi1, initial_F2, [initial_T1, initial_T2, initial_r_transition]])
    
    bounds = ([psi1_bounds] * N1_segments +
              [F2_bounds] * N2_segments +
              [T1_bounds, T2_bounds, r_transition_bounds])
    
    cons = [{'type': 'eq', 'fun': constraints}]
    
    print("--- 开始使用CVP和直接打靶法进行优化 ---")
    start_time = time.time()
    
    result = minimize(
        fun=objective,
        x0=x0,
        method='SLSQP',
        constraints=cons,
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-7, 'disp': True}
    )
    
    print(f"--- 优化用时: {time.time() - start_time:.2f} 秒 ---")
    return result

# ========== 5. 结果处理与可视化 ==========
def post_process(result):
    if not result.success:
        print("\n优化失败:", result.message)
        return

    print("\n--- CVP优化成功! ---")
    
    # 使用最优参数重新进行一次高精度仿真
    sim_result = simulate_trajectory(result.x)
    sol1 = sim_result['sol1']
    sol2 = sim_result['sol2']
    opt_psi1, opt_F2, opt_T1, opt_T2, opt_r_transition = unpack_params(result.x)
    
    # 合并轨迹数据
    t_axis = np.concatenate([sol1.t, sol2.t])
    y_matrix = np.concatenate([sol1.y, sol2.y], axis=1)
    r_opt, vr_opt, vt_opt, m_opt = y_matrix
    
    final_mass = m_opt[-1]
    fuel_consumed = m0_dim - final_mass
    
    print(f"\n--- 最优结果 (CVP方法) ---")
    print(f"最优燃料消耗: {fuel_consumed:.2f} kg")
    print(f"总飞行时间: {opt_T1 + opt_T2:.2f} s")
    print(f"最优转折点高度: {opt_r_transition - R_moon_dim:.2f} m")

if __name__ == "__main__":
    result = cvp_optimization()
    post_process(result)