import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 1. 设置中文字体和CUDA设备 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置失败，将使用默认字体: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 使用设备: {device} ---")

# ========== 2. 系统参数 ==========
G_const = 6.67430e-11
M_moon = 7.3477e22
mu_moon = G_const * M_moon
R_moon_dim = 1737013.0
m0_dim = 3762.0
F_max_dim = 7500.0
v_e = 3038.0
r0_dim = R_moon_dim + 11700.0
vr0_dim = 0.0
vt0_dim = 1762.83

# --- 终端和转折点目标 ---
r_target_final = R_moon_dim + 4.0
vr_target_final = 0.0
vt_target_final = 0.0

r_target_stage1 = R_moon_dim + 3000.0
vt_target_stage1 = 0.0
print(f"--- 转折点目标: 高度={(r_target_stage1 - R_moon_dim):.0f}m, 切向速度={vt_target_stage1:.1f}m/s ---")


# 将常量转换为PyTorch张量并移到GPU
X0 = torch.tensor([r0_dim, vr0_dim, vt0_dim, m0_dim], dtype=torch.float32, device=device)
mu_moon_t = torch.tensor(mu_moon, dtype=torch.float32, device=device)
F_max_dim_t = torch.tensor(F_max_dim, dtype=torch.float32, device=device)
v_e_t = torch.tensor(v_e, dtype=torch.float32, device=device)
R_moon_dim_t = torch.tensor(R_moon_dim, dtype=torch.float32, device=device)
r_target_t = torch.tensor(r_target_final, dtype=torch.float32, device=device)
vr_target_t = torch.tensor(vr_target_final, dtype=torch.float32, device=device)
vt_target_t = torch.tensor(vt_target_final, dtype=torch.float32, device=device)
r1_target_t = torch.tensor(r_target_stage1, dtype=torch.float32, device=device)
vt1_target_t = torch.tensor(vt_target_stage1, dtype=torch.float32, device=device)


# ========== 3. CVP 设置 ==========
N1_segments = 6
N2_segments = 4
# 总优化变量数: 6 (psi1) + 4 (F2) + 1 (T1) + 1 (T2) = 12
n_params = N1_segments + N2_segments + 2

# ========== 4. 并行化动力学与仿真 ==========

def dynamics_batch(states, F, psi):
    r, vr, vt, m = states.T
    safe_mask = (m > 500) & (r > R_moon_dim_t * 0.9)
    r_dot, vr_dot, vt_dot, m_dot = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
    if safe_mask.any():
        m_s, r_s, vr_s, vt_s, F_s, psi_s = m[safe_mask], r[safe_mask], vr[safe_mask], vt[safe_mask], F[safe_mask], psi[safe_mask]
        r_dot[safe_mask] = vr_s
        vr_dot[safe_mask] = (vt_s**2 / r_s) - (mu_moon_t / r_s**2) + (F_s / m_s) * torch.sin(psi_s)
        vt_dot[safe_mask] = -(vr_s * vt_s / r_s) - (F_s / m_s) * torch.cos(psi_s)
        m_dot[safe_mask] = -F_s / v_e_t
    return torch.stack([r_dot, vr_dot, vt_dot, m_dot], dim=1)

def rk4_step_batch(states, F, psi, dt):
    dt_exp = dt.unsqueeze(1)
    k1 = dynamics_batch(states, F, psi)
    k2 = dynamics_batch(states + dt_exp / 2 * k1, F, psi)
    k3 = dynamics_batch(states + dt_exp / 2 * k2, F, psi)
    k4 = dynamics_batch(states + dt_exp * k3, F, psi)
    return states + dt_exp / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate_trajectory_batch(population):
    pop_size = population.shape[0]
    psi1 = population[:, :N1_segments]
    F2 = population[:, N1_segments : N1_segments + N2_segments]
    T1 = population[:, -2]
    T2 = population[:, -1]
    
    states = X0.expand(pop_size, -1)
    dt1 = T1 / N1_segments
    for i in range(N1_segments):
        states = rk4_step_batch(states, F_max_dim_t.expand(pop_size), psi1[:, i], dt1)
    
    stage1_final_states = states.clone()

    # 在这里，我们不再强制阶段一的终点高度，而是让适应度函数去惩罚偏差
    # states[:, 0] = r_target_stage1 

    dt2 = T2 / N2_segments
    for i in range(N2_segments):
        states = rk4_step_batch(states, F2[:, i], torch.full_like(T1, np.pi / 2), dt2)
        
    return states, stage1_final_states

# ========== 5. 遗传算法 (含新的适应度函数) ==========

class CUDAGeneticAlgorithm:
    def __init__(self, pop_size, n_params, bounds, mutation_rate=0.2, crossover_rate=0.8):
        self.pop_size = pop_size
        self.n_params = n_params
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=device).T
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.bounds[0] + torch.rand(pop_size, n_params, device=device) * (self.bounds[1] - self.bounds[0])
        
    def fitness(self):
        """
        新的适应度函数，精确地惩罚对新约束的违反
        """
        final_states, stage1_final_states = simulate_trajectory_batch(self.population)
        
        # --- 解包状态 ---
        r1_final, vr1_final, vt1_final, m1_final = stage1_final_states.T
        r2_final, vr2_final, vt2_final, m2_final = final_states.T
        
        # --- 惩罚权重 ---
        # 阶段一约束的权重
        W_r1 = 10.0   # 强制到达转折高度
        W_vt1 = 5.0   # 强制转折时切向速度为0
        W_vr1 = 0.01  # 轻微惩罚径向速度，使其不要过大
        
        # 阶段二（最终着陆）约束的权重
        W_r2 = 1.0
        W_vr2 = 10.0
        W_vt2 = 10.0

        # --- 计算惩罚 ---
        # 惩罚第一阶段违反约束
        penalty_stage1 = (W_r1 * (r1_final - r1_target_t)**2 +
                          W_vt1 * (vt1_final - vt1_target_t)**2 +
                          W_vr1 * vr1_final**2)
        
        # 惩罚第二阶段违反约束
        penalty_stage2 = (W_r2 * (r2_final - r_target_t)**2 +
                          W_vr2 * (vr2_final - vr_target_t)**2 +
                          W_vt2 * (vt2_final - vt_target_t)**2)

        # --- 综合适应度 ---
        # 目标: 最大化最终质量 (最小化-m2_final)
        # 同时加上所有约束的惩罚
        fitness_scores = -m2_final + penalty_stage1 + penalty_stage2
        
        # 对物理上不可能的解给予巨大惩罚
        fitness_scores[m2_final <= 500] = 1e9
        
        return fitness_scores

    def evolve(self):
        fitness_scores = self.fitness()
        new_population = torch.zeros_like(self.population)
        for i in range(self.pop_size):
            p1_idx, p2_idx = torch.randperm(self.pop_size, device=device)[:2]
            winner_idx = p1_idx if fitness_scores[p1_idx] < fitness_scores[p2_idx] else p2_idx
            new_population[i] = self.population[winner_idx]
        for i in range(0, self.pop_size, 2):
            if torch.rand(1) < self.crossover_rate:
                p1, p2 = new_population[i], new_population[i+1]
                u = torch.rand(self.n_params, device=device)
                eta_c = 20.0
                beta = torch.where(u <= 0.5, (2 * u)**(1.0 / (eta_c + 1.0)), (1.0 / (2.0 * (1.0 - u)))**(1.0 / (eta_c + 1.0)))
                c1, c2 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                new_population[i], new_population[i+1] = c1, c2
        mutation_mask = torch.rand(self.pop_size, self.n_params, device=device) < self.mutation_rate
        eta_m = 20.0
        u = torch.rand(self.pop_size, self.n_params, device=device)
        delta = torch.where(u < 0.5, (2*u)**(1.0/(eta_m+1.0)) - 1.0, 1.0 - (2*(1-u))**(1.0/(eta_m+1.0)))
        new_population += mutation_mask * delta * (self.bounds[1] - self.bounds[0])
        self.population = torch.max(torch.min(new_population, self.bounds[1]), self.bounds[0])

    def optimize(self, generations):
        print("--- 开始基于CUDA的并行遗传算法优化 ---")
        start_time = time.time()
        best_fitness_overall = float('inf')
        best_solution_overall = None
        for gen in range(generations):
            self.evolve()
            if (gen + 1) % 10 == 0:
                current_fitness = self.fitness()
                best_fitness_gen = torch.min(current_fitness)
                if best_fitness_gen < best_fitness_overall:
                    best_fitness_overall = best_fitness_gen
                    best_idx = torch.argmin(current_fitness)
                    best_solution_overall = self.population[best_idx].clone()
                print(f"第 {gen+1}/{generations} 代 | 最佳适应度: {best_fitness_overall.item():.2f} | 用时: {time.time() - start_time:.2f}s")
        return best_solution_overall

# ========== 6. 结果分析与可视化 ==========
def unpack_params(solution_vec):
    psi1 = solution_vec[:N1_segments]
    F2 = solution_vec[N1_segments : N1_segments + N2_segments]
    T1 = solution_vec[-2]
    T2 = solution_vec[-1]
    return psi1, F2, T1, T2

def simulate_single_trajectory(solution_vec, num_points_per_stage=200):
    psi1, F2, T1, T2 = unpack_params(solution_vec)
    states = X0.unsqueeze(0)
    dt1 = T1 / num_points_per_stage
    history1 = [states.clone()]
    for i in range(num_points_per_stage):
        segment_idx = min(int((i / num_points_per_stage) * N1_segments), N1_segments - 1)
        psi1_k, F1_k = psi1[segment_idx].expand(1), F_max_dim_t.expand(1)
        states = rk4_step_batch(states, F1_k, psi1_k, dt1.expand(1))
        history1.append(states.clone())
    
    stage1_final_state_sim = states.clone()

    dt2 = T2 / num_points_per_stage
    history2 = [states.clone()]
    for i in range(num_points_per_stage):
        segment_idx = min(int((i / num_points_per_stage) * N2_segments), N2_segments - 1)
        F2_k, psi2_k = F2[segment_idx].expand(1), torch.full((1,), np.pi / 2, device=device)
        states = rk4_step_batch(states, F2_k, psi2_k, dt2.expand(1))
        history2.append(states.clone())
    return torch.cat(history1).squeeze(1), torch.cat(history2).squeeze(1), stage1_final_state_sim.squeeze(0)

def post_process(best_solution):
    print("\n--- 开始进行结果分析与可视化 ---")
    traj1_gpu, traj2_gpu, stage1_final_gpu = simulate_single_trajectory(best_solution)
    traj1, traj2, s1_final, solution_np = traj1_gpu.cpu().numpy(), traj2_gpu.cpu().numpy(), stage1_final_gpu.cpu().numpy(), best_solution.cpu().numpy()
    opt_psi1, opt_F2, opt_T1, opt_T2 = unpack_params(solution_np)
    full_traj = np.vstack([traj1, traj2[1:]])
    r_opt, vr_opt, vt_opt, m_opt = full_traj.T
    t1_axis, t2_axis = np.linspace(0, opt_T1, len(traj1)), np.linspace(opt_T1, opt_T1 + opt_T2, len(traj2))
    t_axis = np.concatenate([t1_axis, t2_axis[1:]])
    final_state = full_traj[-1, :]
    fuel_consumed = m0_dim - final_state[3]
    print(f"\n--- 最优结果 (CUDA GA + CVP) ---")
    print(f"最优燃料消耗: {fuel_consumed:.2f} kg")
    print(f"总飞行时间: {opt_T1 + opt_T2:.2f} s (T1={opt_T1:.2f}s, T2={opt_T2:.2f}s)")
    print("\n--- 转折点状态 ---")
    print(f"高度误差: {s1_final[0] - r_target_stage1:.2f} m")
    print(f"切向速度误差: {s1_final[2] - vt_target_stage1:.2f} m/s")
    print(f"径向速度: {s1_final[1]:.2f} m/s")
    print("\n--- 终端状态 ---")
    print(f"着陆高度误差: {final_state[0] - r_target_final:.2f} m")
    print(f"径向速度误差: {final_state[1] - vr_target_final:.2f} m/s")
    print(f"切向速度误差: {final_state[2] - vt_target_final:.2f} m/s")
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f'CUDA遗传算法优化结果 (燃料消耗: {fuel_consumed:.1f} kg)', fontsize=18)
    axs[0, 0].plot(t_axis, (r_opt - R_moon_dim) / 1000)
    axs[0, 0].axvline(opt_T1, color='r', linestyle='--', label=f'转折点 T={opt_T1:.1f}s')
    axs[0, 0].set_title('高度 vs. 时间'), axs[0, 0].set_xlabel('时间 (s)'), axs[0, 0].set_ylabel('高度 (km)'), axs[0, 0].grid(True), axs[0, 0].legend()
    axs[0, 1].plot(t_axis, vr_opt, label='径向速度 $v_r$'), axs[0, 1].plot(t_axis, vt_opt, label='切向速度 $v_t$'), axs[0, 1].axvline(opt_T1, color='r', linestyle='--'), axs[0, 1].set_title('速度 vs. 时间'), axs[0, 1].set_xlabel('时间 (s)'), axs[0, 1].set_ylabel('速度 (m/s)'), axs[0, 1].grid(True), axs[0, 1].legend()
    axs[0, 2].plot(t_axis, m_opt), axs[0, 2].axvline(opt_T1, color='r', linestyle='--'), axs[0, 2].set_title('质量 vs. 时间'), axs[0, 2].set_xlabel('时间 (s)'), axs[0, 2].set_ylabel('质量 (kg)'), axs[0, 2].grid(True)
    t1_control_axis, t2_control_axis = np.linspace(0, opt_T1, N1_segments), np.linspace(opt_T1, opt_T1 + opt_T2, N2_segments)
    axs[1, 0].step(t1_control_axis, np.full(N1_segments, F_max_dim), where='post', label='阶段一 (最大推力)'), axs[1, 0].step(t2_control_axis, opt_F2, where='post', label='阶段二 (可变推力)'), axs[1, 0].set_title('推力剖面'), axs[1, 0].set_xlabel('时间 (s)'), axs[1, 0].set_ylabel('推力 (N)'), axs[1, 0].grid(True), axs[1, 0].legend()
    axs[1, 1].step(t1_control_axis, np.degrees(opt_psi1), where='post', label='阶段一 (可变角度)'), axs[1, 1].step(t2_control_axis, np.full(N2_segments, 90.0), where='post', label='阶段二 (固定90°)'), axs[1, 1].set_title(r'推力角 $\psi$ (相对切向)'), axs[1, 1].set_xlabel('时间 (s)'), axs[1, 1].set_ylabel('角度 (°)') , axs[1, 1].grid(True), axs[1, 1].legend()
    dt_traj = np.diff(t_axis)
    vt_for_theta, r_for_theta = vt_opt[:-1], r_opt[:-1]
    d_theta_rad = (vt_for_theta / r_for_theta) * dt_traj
    theta_traj_rad = np.cumsum(np.concatenate(([0], d_theta_rad)))
    x_traj, y_traj = r_opt * np.cos(theta_traj_rad) / 1000, r_opt * np.sin(theta_traj_rad) / 1000
    axs[1, 2].plot(x_traj, y_traj, label='轨迹'), axs[1, 2].plot(x_traj[len(traj1)-1], y_traj[len(traj1)-1], 'r*', markersize=12, label='最优转折点'), axs[1, 2].set_title('轨迹投影'), axs[1, 2].set_xlabel('X (km)'), axs[1, 2].set_ylabel('Y (km)'), axs[1, 2].axis('equal'), axs[1, 2].grid(True), axs[1, 2].legend()
    plt.show()

# ========== 7. 主流程 ==========
if __name__ == "__main__":
    psi1_bounds = (0.0, np.pi)
    F2_bounds = (0.0, F_max_dim)
    T1_bounds = (300.0, 600.0)
    T2_bounds = (50.0, 250.0)
    all_bounds = ([psi1_bounds] * N1_segments + [F2_bounds] * N2_segments + [T1_bounds, T2_bounds])
    ga_optimizer = CUDAGeneticAlgorithm(pop_size=4096, n_params=n_params, bounds=all_bounds)
    best_solution = ga_optimizer.optimize(generations=300)
    if best_solution is not None:
        post_process(best_solution)
    else:
        print("\n--- 未找到有效解 ---")

