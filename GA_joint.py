import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 0. 配置 ---
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# --- 1. 物理常量和参数 (与 joint.py 的第一阶段对齐) ---
G_const = 6.67430e-11
M_moon = 7.3477e22
mu_moon = G_const * M_moon
v_e = 3038.0             # 与 joint.py 对齐
F_max_dim = 7500.0         # 主减速段使用最大推力

# 初始状态 (与 joint.py 对齐)
R_moon_dim = 1737013.0
H0_dim = 11700.0
r0_dim = R_moon_dim + H0_dim
vr0_dim = 0.0
vt0_dim = 1762.83
m0_dim = 3762.0

# 终端目标状态 (主减速段的目标)
# 注意：这些是软约束，通过惩罚函数实现
rf_dim_target = R_moon_dim + 3000.0  # 目标高度
vtf_dim_target = 100.0               # 目标切向速度 (一个合理的值)
vrf_dim_target = -80.0               # 目标径向速度 (一个合理的值)

# 离散点数 (GPU可以处理更多点以获得更精确的轨迹)
N = 150

# 转换为torch张量
mu_moon_t = torch.tensor(mu_moon, dtype=torch.float32, device=device)
v_e_t = torch.tensor(v_e, dtype=torch.float32, device=device)
R_moon_dim_t = torch.tensor(R_moon_dim, dtype=torch.float32, device=device)
F_max_dim_t = torch.tensor(F_max_dim, dtype=torch.float32, device=device)
r0_dim_t = torch.tensor(r0_dim, dtype=torch.float32, device=device)
vr0_dim_t = torch.tensor(vr0_dim, dtype=torch.float32, device=device)
vt0_dim_t = torch.tensor(vt0_dim, dtype=torch.float32, device=device)
m0_dim_t = torch.tensor(m0_dim, dtype=torch.float32, device=device)
rf_dim_target_t = torch.tensor(rf_dim_target, dtype=torch.float32, device=device)
vtf_dim_target_t = torch.tensor(vtf_dim_target, dtype=torch.float32, device=device)
vrf_dim_target_t = torch.tensor(vrf_dim_target, dtype=torch.float32, device=device)


# --- 2. 动力学函数 ---
def dynamics(state, psi):
    """
    计算动力学导数 (批量操作)
    state: [batch_size, 4] -> [r, v_r, v_t, m]
    psi:   [batch_size]
    """
    r, v_r, v_t, m = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    
    r_dot = v_r
    vr_dot = (v_t**2 / r) - (mu_moon_t / r**2) + (F_max_dim_t / m) * torch.sin(psi)
    vt_dot = -(v_r * v_t / r) + (F_max_dim_t / m) * torch.cos(psi) # 注意：joint.py中cos项为负
    
    # --- BUG修复 ---
    # 原代码: m_dot 是一个标量，而其他导数是向量，导致torch.stack尺寸不匹配
    # m_dot_scalar = -F_max_dim_t / v_e_t
    # 修正后: 将标量结果扩展为与批次大小相同的张量
    m_dot = (-F_max_dim_t / v_e_t).expand_as(r)
    
    return torch.stack([r_dot, vr_dot, vt_dot, m_dot], dim=1)

def rk4_step_batch(states, psi, dt):
    """
    批量RK4积分步骤
    states: [batch_size, 4]
    psi:    [batch_size]
    dt:     [batch_size]
    """
    dt_expanded = dt.unsqueeze(1)
    k1 = dynamics(states, psi)
    k2 = dynamics(states + dt_expanded/2 * k1, psi)
    k3 = dynamics(states + dt_expanded/2 * k2, psi)
    k4 = dynamics(states + dt_expanded * k3, psi)
    
    return states + dt_expanded/6 * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. 批量轨迹仿真 ---
def simulate_trajectory_batch(psi_controls, T_final):
    """
    批量仿真轨迹
    psi_controls: [batch_size, N]
    T_final:      [batch_size]
    """
    batch_size = psi_controls.shape[0]
    
    # 初始状态
    states = torch.zeros(batch_size, 4, device=device)
    states[:, 0] = r0_dim_t
    states[:, 1] = vr0_dim_t
    states[:, 2] = vt0_dim_t
    states[:, 3] = m0_dim_t
    
    # 时间步长
    dt = T_final / N
    
    for i in range(N):
        psi_k = psi_controls[:, i]
        states = rk4_step_batch(states, psi_k, dt)
        
        # 检查约束违反
        invalid_mask = (states[:, 0] < R_moon_dim_t) | (states[:, 3] <= 500) | \
                      torch.isnan(states).any(dim=1) | torch.isinf(states).any(dim=1)
        
        if invalid_mask.any():
            # 对违反约束的个体直接设置一个非常差的状态，使其在评估中被淘汰
            states[invalid_mask, 3] = 0 # 质量设为0作为惩罚标志
    
    return states

# --- 4. 目标函数 ---
def objective_function_batch(chromosomes):
    """
    批量计算目标函数
    chromosomes: [batch_size, N+1] (psi_controls and T_final)
    """
    psi_controls = chromosomes[:, :N]
    T_final = chromosomes[:, N]
    
    final_states = simulate_trajectory_batch(psi_controls, T_final)
    
    r_final, vr_final, vt_final, m_final = final_states.T
    
    # 主要目标：最大化最终质量
    cost = -m_final
    
    # 惩罚项权重
    W_pos = 1e-3
    W_velt = 1e-1
    W_velr = 1e-2
    
    # 位置误差惩罚
    pos_error = torch.abs(r_final - rf_dim_target_t)
    cost += W_pos * pos_error**2
    
    # 切向速度误差惩罚
    velt_error = torch.abs(vt_final - vtf_dim_target_t)
    cost += W_velt * velt_error**2
    
    # 径向速度误差惩罚
    velr_error = torch.abs(vr_final - vrf_dim_target_t)
    cost += W_velr * velr_error**2
    
    # 对无效解施加大惩罚
    invalid_mask = (m_final <= 0) | torch.isnan(cost) | torch.isinf(cost)
    cost[invalid_mask] = 1e12 # 使用一个非常大的数值
    
    return cost

# --- 5. 差分进化算法 ---
class DifferentialEvolution:
    def __init__(self, bounds, popsize=100, F=0.6, CR=0.9, maxiter=500):
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        self.popsize = popsize
        self.F = F
        self.CR = CR
        self.maxiter = maxiter
        self.dim = len(bounds)
        
        # 初始化种群
        self.population = torch.rand(popsize, self.dim, device=device) * \
                          (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        
        self.fitness = objective_function_batch(self.population)
        self.best_idx = torch.argmin(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_solution = self.population[self.best_idx].clone()
        
    def evolve(self):
        # 创建试验向量
        indices = torch.arange(self.popsize, device=device)
        # 为每个个体随机选择三个不重复且不等于自身的索引
        choices = torch.cat([torch.randperm(self.popsize-1, device=device)[:3].unsqueeze(0) for _ in range(self.popsize)])
        choices[choices >= indices.unsqueeze(1)] += 1
        
        a, b, c = self.population[choices[:, 0]], self.population[choices[:, 1]], self.population[choices[:, 2]]
        
        mutant = a + self.F * (b - c)
        mutant = torch.max(torch.min(mutant, self.bounds[:, 1]), self.bounds[:, 0]) # 边界处理
        
        cross_points = torch.rand(self.popsize, self.dim, device=device) < self.CR
        # 确保至少一个维度被交叉
        rand_dim = torch.randint(0, self.dim, (self.popsize,), device=device)
        cross_points[indices, rand_dim] = True
        
        trial = torch.where(cross_points, mutant, self.population)
        
        trial_fitness = objective_function_batch(trial)
        
        improvement_mask = trial_fitness < self.fitness
        self.population[improvement_mask] = trial[improvement_mask]
        self.fitness[improvement_mask] = trial_fitness[improvement_mask]
        
        current_best_idx = torch.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[current_best_idx]
            self.best_solution = self.population[current_best_idx].clone()

    def optimize(self):
        print(f"初始最佳适应度: {self.best_fitness.item():.6f}")
        start_time = time.time()
        for generation in range(self.maxiter):
            self.evolve()
            if (generation + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"第 {generation+1}/{self.maxiter} 代, 最佳适应度: {self.best_fitness.item():.6f}, 已用时: {elapsed:.2f}s")
        return self.best_solution, self.best_fitness

# --- 6. 运行优化 ---
if __name__ == "__main__":
    # 设置边界
    bounds = []
    # 推力角 psi 边界 (0 到 pi)
    for _ in range(N):
        bounds.append([0, np.pi])
    # 时间 T1 边界
    bounds.append([300, 700])
    
    print("--- 开始运行PyTorch差分进化算法生成初始猜测值 ---")
    
    optimizer = DifferentialEvolution(bounds, popsize=256, F=0.7, CR=0.9, maxiter=800)
    best_solution, best_fitness = optimizer.optimize()
    
    print(f"\n--- 优化完成！---")
    
    best_solution_cpu = best_solution.cpu().numpy()
    
    print(f"\n找到的最佳适应度值 (成本): {best_fitness.item():.6f}")
    
    psi_guess_ga = best_solution_cpu[:N]
    T_final_guess = best_solution_cpu[N]
    
    # --- 格式化输出，方便复制 ---
    print("\n" + "="*60)
    print("--- 请将以下代码复制到 `Joint_Optimization_with_GA_Guess.py` 中 ---")
    print("="*60 + "\n")
    
    print("# [GA生成的初始值] 阶段一飞行时间")
    print(f"T1_guess = {T_final_guess:.6f}\n")
    
    print("# [GA生成的初始值] 阶段一推力角序列")
    print(f"psi1_guess_ga = np.array({np.array2string(psi_guess_ga, max_line_width=120, precision=6, separator=',')})\n")
    
    print("#" + "="*58)
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.suptitle("GA生成的初始猜测值", fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, T_final_guess, N), np.degrees(psi_guess_ga))
    plt.title('最优推力角序列 (psi)')
    plt.xlabel('时间 (s)')
    plt.ylabel('推力角 (度)')
    plt.grid(True)
    
    # 仿真并绘制轨迹
    final_states_vis = simulate_trajectory_batch(best_solution[:N].unsqueeze(0), best_solution[N].unsqueeze(0))
    final_states_vis = final_states_vis.cpu().numpy().squeeze()
    
    print("\n--- 最终状态预览 ---")
    print(f"最终高度: {(final_states_vis[0] - R_moon_dim):.2f} m")
    print(f"最终径向速度: {final_states_vis[1]:.2f} m/s")
    print(f"最终切向速度: {final_states_vis[2]:.2f} m/s")
    print(f"最终质量: {final_states_vis[3]:.2f} kg")

    plt.subplot(1, 2, 2)
    plt.text(0.05, 0.9, f"T_final: {T_final_guess:.2f} s", transform=plt.gca().transAxes)
    plt.text(0.05, 0.8, f"H_final: {(final_states_vis[0] - R_moon_dim):.1f} m", transform=plt.gca().transAxes)
    plt.text(0.05, 0.7, f"Vr_final: {final_states_vis[1]:.1f} m/s", transform=plt.gca().transAxes)
    plt.text(0.05, 0.6, f"Vt_final: {final_states_vis[2]:.1f} m/s", transform=plt.gca().transAxes)
    plt.text(0.05, 0.5, f"M_final: {final_states_vis[3]:.1f} kg", transform=plt.gca().transAxes)
    plt.title("最终状态")
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()