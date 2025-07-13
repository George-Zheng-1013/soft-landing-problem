import numpy as np
from scipy.optimize import differential_evolution
import time

def main():
    # --- 1. 从原脚本复制必要的物理常量和无量纲化参数 ---
    # 物理常量
    mu = 4.903e12
    Isp = 316.0
    g_e = 9.81

    # 初始/终端条件 (有单位)
    R0_dim = 1753000.0
    V_r0_dim = 0.0
    V_t0_dim = 1692.0
    M0_dim = 750.0
    Rf_dim = 1741030.0
    Vr_f_max = 1.5
    Vt_f_max = 3.8
    F_max_dim = 1750.0

    # 无量纲化
    DU = R0_dim
    MU = M0_dim
    TU = np.sqrt(DU**3 / mu)
    VU = DU / TU
    FU = MU * DU / TU**2
    Rf_norm = Rf_dim / DU
    Vr0_norm = V_r0_dim / VU
    Vt0_norm = V_t0_dim / VU
    Vr_f_norm_max = Vr_f_max / VU
    Vt_f_norm_max = Vt_f_max / VU
    F_max_norm = F_max_dim / FU
    C_thrust = (TU * FU) / (MU * Isp * g_e)

    N = 50  # GA中使用较少的离散点

    # --- 2. 定义适应度函数 ---
    def objective_function(chromosome):
        """
        计算给定控制策略的成本/适应度。
        """
        # 2.1 解码染色体
        f_controls = chromosome[0:N]
        theta_controls = chromosome[N:2*N]
        T_final = chromosome[2*N]

        # 2.2 模拟轨迹
        dt_norm = (T_final / TU) / N
        
        # 初始状态
        r, vr, vt, m = 1.0, Vr0_norm, Vt0_norm, 1.0
        
        for i in range(N):
            f_n = f_controls[i]
            Theta_n = theta_controls[i]

            # 动力学
            r_dot = vr
            vr_dot = (f_n * np.cos(Theta_n)) / m - 1 / r**2 + vt**2 / r
            vt_dot = (f_n * np.sin(Theta_n)) / m - vr * vt / r
            m_dot = -C_thrust * f_n
            
            # 欧拉积分更新状态
            r += r_dot * dt_norm
            vr += vr_dot * dt_norm
            vt += vt_dot * dt_norm
            m += m_dot * dt_norm

        # 2.3 计算成本
        cost = -m  # 最大化最终质量
        
        # 惩罚项
        W_pos = 1e4
        W_vel = 1e4
        
        # 位置误差惩罚
        pos_error = abs(r - Rf_norm)
        cost += W_pos * pos_error**2
        
        # 速度误差惩罚
        vel_r_error = max(0, abs(vr) - Vr_f_norm_max)
        vel_t_error = max(0, abs(vt) - Vt_f_norm_max)
        cost += W_vel * (vel_r_error**2 + vel_t_error**2)
        
        return cost

    # --- 3. 设置遗传算法的边界 ---
    bounds = []
    # 推力 f_norm 的边界
    for _ in range(N):
        bounds.append((0, F_max_norm))
    # 推力角度 Theta 的边界
    for _ in range(N):
        bounds.append((-np.pi, 0))
    # 总时间 T_final 的边界
    bounds.append((500, 800))

    # --- 4. 运行差分进化算法 ---
    print("--- 开始运行差分进化算法进行全局搜索 (这可能需要几分钟)... ---")
    start_time = time.time()

    # 移除多进程参数以避免Windows错误
    result = differential_evolution(objective_function, bounds, 
                                    strategy='best1bin', maxiter=100,  # 减少迭代次数
                                    popsize=15, tol=0.01, 
                                    disp=True, seed=42)  # 移除workers参数

    end_time = time.time()
    print(f"--- 搜索完成！耗时: {end_time - start_time:.2f} 秒 ---")

    # --- 5. 打印最优结果 ---
    best_chromosome = result.x
    best_fitness = result.fun

    print(f"\n找到的最佳适应度值: {best_fitness:.6f}")
    print("\n--- 请将以下初始猜测值复制到您的 [主减速.py](http://_vscodecontentref_/0) 脚本中 ---")

    # 格式化输出
    f_guess = best_chromosome[0:N]
    theta_guess = best_chromosome[N:2*N]
    T_final_guess = best_chromosome[2*N]

    print("\n# 初始猜测值 (来自遗传算法)")
    print(f"opti.set_initial(T_final, {T_final_guess:.4f})")

    print("# 对于控制变量，您可能需要进行插值或使用重复值")
    print("# 这是一个示例，展示如何使用这些值")
    print(f"f_guess_ga = np.array({np.array2string(f_guess, precision=4, separator=',')})")
    print(f"theta_guess_ga = np.array({np.array2string(theta_guess, precision=4, separator=',')})")
    
    print("\n# 插值到您的N值（例如N=200）:")
    print("# f_interp = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, 50), f_guess_ga)")
    print("# theta_interp = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, 50), theta_guess_ga)")
    print("# opti.set_initial(f_norm, f_interp)")
    print("# opti.set_initial(Theta, theta_interp)")

    # 简化的插值示例
    print(f"\n# 简化版本 - 使用平均值:")
    print(f"f_avg = {np.mean(f_guess):.4f}")
    print(f"theta_avg = {np.mean(theta_guess):.4f}")
    print("# opti.set_initial(f_norm, f_avg)")
    print("# opti.set_initial(Theta, theta_avg)")

if __name__ == '__main__':
    main()