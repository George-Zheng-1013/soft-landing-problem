import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 1. 物理常量和参数 ---
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
rf_dim = R_moon_dim + 2400.0
vtf_dim = 0.0

# 推力 (恒定最大)
F_max_dim = 7500.0

# 离散点数
N = 100  # 可以增加更多点，GPU能处理

# 转换为torch张量
mu_moon_t = torch.tensor(mu_moon, dtype=torch.float32, device=device)
v_e_t = torch.tensor(v_e, dtype=torch.float32, device=device)
R_moon_dim_t = torch.tensor(R_moon_dim, dtype=torch.float32, device=device)
F_max_dim_t = torch.tensor(F_max_dim, dtype=torch.float32, device=device)
r0_dim_t = torch.tensor(r0_dim, dtype=torch.float32, device=device)
vr0_dim_t = torch.tensor(vr0_dim, dtype=torch.float32, device=device)
vt0_dim_t = torch.tensor(vt0_dim, dtype=torch.float32, device=device)
m0_dim_t = torch.tensor(m0_dim, dtype=torch.float32, device=device)
rf_dim_t = torch.tensor(rf_dim, dtype=torch.float32, device=device)
vtf_dim_t = torch.tensor(vtf_dim, dtype=torch.float32, device=device)

# --- 2. 动力学函数 ---
def dynamics(state, psi):
    """
    计算动力学导数
    state: [r, v_r, v_t, m]
    psi: 推力角
    """
    r, v_r, v_t, m = state
    
    r_dot = v_r
    vr_dot = (v_t**2 / r) - (mu_moon_t / r**2) + (F_max_dim_t / m) * torch.sin(psi)
    vt_dot = -(v_r * v_t / r) - (F_max_dim_t / m) * torch.cos(psi)
    m_dot = -F_max_dim_t / v_e_t
    
    return torch.stack([r_dot, vr_dot, vt_dot, m_dot])

def rk4_step(state, psi, dt):
    """
    RK4积分步骤
    """
    k1 = dynamics(state, psi)
    k2 = dynamics(state + dt/2 * k1, psi)
    k3 = dynamics(state + dt/2 * k2, psi)
    k4 = dynamics(state + dt * k3, psi)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. 批量轨迹仿真 ---
def simulate_trajectory_batch(psi_controls, T_final):
    """
    批量仿真轨迹
    psi_controls: [batch_size, N]
    T_final: [batch_size]
    """
    batch_size = psi_controls.shape[0]
    
    # 初始状态
    states = torch.zeros(batch_size, 4, device=device)
    states[:, 0] = r0_dim_t
    states[:, 1] = vr0_dim_t
    states[:, 2] = vt0_dim_t
    states[:, 3] = m0_dim_t
    
    # 时间步长
    dt = T_final / N  # [batch_size]
    
    for i in range(N):
        psi_k = psi_controls[:, i]  # [batch_size]
        dt_k = dt  # [batch_size]
        
        # 并行RK4积分
        new_states = torch.zeros_like(states)
        for j in range(batch_size):
            new_states[j] = rk4_step(states[j], psi_k[j], dt_k[j])
        
        states = new_states
        
        # 检查约束违反
        invalid_mask = (states[:, 0] < R_moon_dim_t) | (states[:, 3] <= 0) | \
                      torch.isnan(states).any(dim=1) | torch.isinf(states).any(dim=1)
        
        if invalid_mask.any():
            # 对违反约束的个体设置惩罚
            states[invalid_mask, 3] = 0  # 质量设为0
    
    return states

# --- 4. 目标函数 ---
def objective_function_batch(chromosomes):
    """
    批量计算目标函数
    chromosomes: [batch_size, N+1]
    """
    batch_size = chromosomes.shape[0]
    
    psi_controls = chromosomes[:, :N]  # [batch_size, N]
    T_final = chromosomes[:, N]        # [batch_size]
    
    # 仿真轨迹
    final_states = simulate_trajectory_batch(psi_controls, T_final)
    
    # 提取最终状态
    r_final = final_states[:, 0]
    vr_final = final_states[:, 1]
    vt_final = final_states[:, 2]
    m_final = final_states[:, 3]
    
    # 计算成本
    cost = -m_final  # 主要目标：最大化最终质量
    
    # 惩罚项权重
    W_pos = 1e-3
    W_vel = 1e-1
    W_vr = 1e-2
    
    # 位置误差惩罚
    pos_error = torch.abs(r_final - rf_dim_t)
    cost += W_pos * pos_error**2
    
    # 切向速度误差惩罚
    vel_t_error = torch.abs(vt_final - vtf_dim_t)
    cost += W_vel * vel_t_error**2
    
    # 径向速度惩罚
    cost += W_vr * vr_final**2
    
    # 对无效解施加大惩罚
    invalid_mask = (m_final <= 0) | torch.isnan(cost) | torch.isinf(cost)
    cost[invalid_mask] = 1e9
    
    return cost

# --- 5. 差分进化算法 ---
class DifferentialEvolution:
    def __init__(self, bounds, popsize=50, F=0.5, CR=0.9, maxiter=500):
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        self.popsize = popsize
        self.F = F
        self.CR = CR
        self.maxiter = maxiter
        self.dim = len(bounds)
        
        # 初始化种群
        self.population = torch.rand(popsize, self.dim, device=device)
        for i in range(self.dim):
            self.population[:, i] = (self.bounds[i, 1] - self.bounds[i, 0]) * \
                                   self.population[:, i] + self.bounds[i, 0]
        
        # 评估初始种群
        self.fitness = objective_function_batch(self.population)
        self.best_idx = torch.argmin(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_solution = self.population[self.best_idx].clone()
        
    def evolve(self):
        """进化一代"""
        # 创建试验向量
        trial_population = torch.zeros_like(self.population)
        
        for i in range(self.popsize):
            # 选择三个不同的随机个体
            candidates = torch.randperm(self.popsize, device=device)
            candidates = candidates[candidates != i][:3]
            a, b, c = candidates
            
            # 变异
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
            
            # 边界处理
            mutant = torch.clamp(mutant, self.bounds[:, 0], self.bounds[:, 1])
            
            # 交叉
            trial = self.population[i].clone()
            crossover_mask = torch.rand(self.dim, device=device) < self.CR
            trial[crossover_mask] = mutant[crossover_mask]
            
            # 确保至少有一个维度被交叉
            if not crossover_mask.any():
                rand_idx = torch.randint(0, self.dim, (1,), device=device)
                trial[rand_idx] = mutant[rand_idx]
            
            trial_population[i] = trial
        
        # 评估试验种群
        trial_fitness = objective_function_batch(trial_population)
        
        # 选择
        improvement_mask = trial_fitness < self.fitness
        self.population[improvement_mask] = trial_population[improvement_mask]
        self.fitness[improvement_mask] = trial_fitness[improvement_mask]
        
        # 更新最优解
        current_best_idx = torch.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[current_best_idx]
            self.best_solution = self.population[current_best_idx].clone()
            self.best_idx = current_best_idx
    
    def optimize(self):
        """优化过程"""
        print(f"初始最佳适应度: {self.best_fitness:.6f}")
        
        for generation in range(self.maxiter):
            self.evolve()
            
            if generation % 50 == 0:
                print(f"第 {generation} 代, 最佳适应度: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness

# --- 6. 运行优化 ---
if __name__ == "__main__":
    # 设置边界
    bounds = []
    # 推力角边界
    for _ in range(N):
        bounds.append([-np.pi/2, np.pi/2])
    # 时间边界
    bounds.append([300, 1200])
    
    print("--- 开始运行PyTorch版本的差分进化算法 ---")
    start_time = time.time()
    
    # 创建优化器
    optimizer = DifferentialEvolution(bounds, popsize=50, F=0.5, CR=0.9, maxiter=500)
    # 运行优化
    best_solution, best_fitness = optimizer.optimize()
    
    end_time = time.time()
    print(f"--- 优化完成！耗时: {end_time - start_time:.2f} 秒 ---")
    
    # 转换回CPU和numpy
    best_solution_cpu = best_solution.cpu().numpy()
    
    print(f"\n找到的最佳适应度值 (成本): {best_fitness:.6f}")
    print("\n--- 请将以下初始猜测值复制到您的 `主减速_o1.py` 脚本中 ---")
    
    # 格式化输出
    psi_guess_ga = best_solution_cpu[:N]
    T_final_guess = best_solution_cpu[N]
    
    print("\n# 初始猜测值 (来自PyTorch遗传算法)")
    print(f"opti.set_initial(T_final, {T_final_guess:.4f})")
    
    print("# 对于控制变量，需要进行插值")
    print(f"psi_guess_ga = np.array({np.array2string(psi_guess_ga, precision=4, separator=',')})")
    print("# CasADi脚本中的N值 (例如N=400)")
    print("# N_casadi = 400")
    print(f"# N_ga = {N}")
    print("# psi_interp = np.interp(np.linspace(0, 1, N_casadi), np.linspace(0, 1, N_ga), psi_guess_ga)")
    print("# opti.set_initial(psi, psi_interp)")
    
    # 可视化结果
    if len(psi_guess_ga) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.degrees(psi_guess_ga))
        plt.title('最优推力角序列')
        plt.xlabel('时间步')
        plt.ylabel('推力角 (度)')
        plt.grid(True)
        plt.show()