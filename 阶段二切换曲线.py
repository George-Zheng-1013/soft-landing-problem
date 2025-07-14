import numpy as np
import matplotlib.pyplot as plt

# --- 1. 设置中文字体和绘图风格 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-notebook')

# --- 2. 定义物理常量 (与论文和您的代码对齐) ---
# 注意：论文中的模型是简化的，这里的参数是为了演示
g = 1.62         # 月球表面重力加速度 (m/s^2)
m0 = 3762        # 初始总质量 (kg)
ms = 1900        # 飞船干重 (kg), 这是m_bar的下限
# 论文中的 k 是质量消耗率，与 F_max/v_e 相关。我们这里假设一个值来演示。
# k = F_max / v_e = 7500 / 3038 ≈ 2.468
k = 2.468

# --- 3. 定义计算切换曲线的函数 ---
def calculate_switching_curve(m_bar_range):
    """
    根据一系列的最终质量 m_bar，计算切换曲线的 (v, h) 坐标。
    公式来源：论文 Eq. (2.12)
    """
    m_bar = m_bar_range
    # 速度 v 的计算
    v_switch = (g / k) * (m0 - m_bar) + (1 / k) * np.log(m_bar / m0)
    
    # 高度 h 的计算
    h_switch = -(m0 - m_bar) / k**2 - (g / (2 * k**2)) * (m0 - m_bar)**2 - (m0 / k**2) * np.log(m_bar / m0)
    
    return v_switch, h_switch

# --- 4. 定义计算一条安全着陆曲线的函数 (用于对比) ---
def calculate_safe_landing_curve(m_bar_final, t_flight):
    """
    计算给定最终质量 m_bar_final 的安全着陆曲线。
    公式来源：论文 Eq. (2.8)
    """
    sigma = np.linspace(0, t_flight, 500) # sigma = τ - t
    m_t = m_bar_final + k * sigma
    
    # 速度 v 的计算
    v_landing = g * sigma - (1 / k) * np.log(m_t / m_bar_final)
    
    # 高度 h 的计算
    h_landing = (v_landing * sigma) + (m_t / k**2) * np.log(m_t / m_bar_final) - (m_t - m_bar_final) / k**2
    # 论文中的h(t)公式有点复杂，这里用一个等价形式 h(t) = ∫v(t)dt
    # h_landing = -sigma/k - (g/2)*sigma**2 + (m_t/k**2)*np.log(m_t/m_bar_final)
    
    return v_landing, h_landing

# --- 5. 执行计算 ---
# 创建一系列的最终质量 m_bar 值，从干重到接近初始总重
m_bar_values = np.linspace(ms, m0 * 0.999, 1000)

# 计算切换曲线
v_sc, h_sc = calculate_switching_curve(m_bar_values)

# 计算一条示例安全着陆曲线 (假设最终剩余质量为2500kg)
m_final_example = 2500
# 这条曲线的起点应该在切换曲线上
v_start_example = np.interp(m_final_example, m_bar_values, v_sc)
h_start_example = np.interp(m_final_example, m_bar_values, h_sc)
# 从起点反推出飞行时间
t_flight_example = (m0 - m_final_example) / k
v_land_ex, h_land_ex = calculate_safe_landing_curve(m_final_example, t_flight_example)


# --- 6. 绘图 ---
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制切换曲线
ax.plot(v_sc, h_sc / 1000, label='切换曲线 (Switching Curve)', color='crimson', linewidth=2.5, zorder=10)

# 绘制示例安全着陆曲线
ax.plot(v_land_ex, h_land_ex / 1000, label=f'安全着陆曲线 (m_final={m_final_example}kg)', color='royalblue', linestyle='--', linewidth=2)

# 绘制一个自由下落的抛物线，展示它如何与切换曲线相交
v_freefall = np.linspace(v_start_example, 50, 200)
h_freefall = h_start_example + (v_start_example**2 - v_freefall**2) / (2 * g)
ax.plot(v_freefall, h_freefall / 1000, label='自由下落轨迹', color='green', linestyle=':', linewidth=2)
ax.scatter(v_start_example, h_start_example / 1000, color='magenta', s=100, zorder=11, label='切换点 (Switch Point)')


# 设置图表样式
ax.set_title('月球着陆切换曲线与安全着陆曲线', fontsize=18)
ax.set_xlabel('速度 v (m/s)', fontsize=14)
ax.set_ylabel('高度 h (km)', fontsize=14)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.set_xlim(left=min(v_sc)-100, right=100)
ax.set_ylim(bottom=-1)

# 添加注释
ax.annotate('自由下落阶段\n(发动机关闭)', xy=(0, 10), xytext=(-400, 15),
            arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7),
            fontsize=12, color='green', ha='center')

ax.annotate('动力下降阶段\n(发动机全开)', xy=(-200, 2), xytext=(-600, 6),
            arrowprops=dict(facecolor='royalblue', shrink=0.05, alpha=0.7),
            fontsize=12, color='royalblue', ha='center')

plt.show()

