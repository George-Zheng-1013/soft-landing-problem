import os
import subprocess
import re
import pandas as pd

# --- 1. 定义多组着陆器参数 ---
# 每组参数是一个字典，包含了需要被修改的初始条件
parameter_sets = [
    {
        "name": "标准着陆器",
        "params": {
            "m0_dim": 3762,
            "F_max_dim": 7500.0,
            "F_min_dim": 1500.0,
            "H0_dim": 11700.0,
            "vt0_dim": 1762.83
        }
    },
    {
        "name": "轻型着陆器",
        "params": {
            "m0_dim": 3200,
            "F_max_dim": 7000.0,
            "F_min_dim": 1400.0,
            "H0_dim": 11700.0,
            "vt0_dim": 1762.83
        }
    },
    {
        "name": "强劲引擎着陆器",
        "params": {
            "m0_dim": 4000,
            "F_max_dim": 8500.0,
            "F_min_dim": 2000.0,
            "H0_dim": 11700.0,
            "vt0_dim": 1762.83
        }
    },
    {
        "name": "高轨道着陆器",
        "params": {
            "m0_dim": 3762,
            "F_max_dim": 7500.0,
            "F_min_dim": 1500.0,
            "H0_dim": 15000.0,
            "vt0_dim": 1700.0 # 对应较高轨道的较低速度
        }
    }
]

# --- 2. 定义运行优化策略的函数 ---
def run_simulation(strategy_script, params_config, output_filename):
    """
    通过修改和执行脚本来运行指定的优化策略。

    Args:
        strategy_script (str): 要运行的Python脚本文件名 (e.g., 'joint_F.py').
        params_config (dict): 包含参数名称和值的字典。
        output_filename (str): 保存输出数据的文件名。
    """
    set_name = params_config['name']
    params = params_config['params']
    
    print(f"--- 正在准备运行: {strategy_script} | 参数组: {set_name} ---")
    
    try:
        with open(strategy_script, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except FileNotFoundError:
        print(f"错误：脚本文件 {strategy_script} 未找到。")
        return

    # 动态修改代码字符串中的参数
    new_code = original_code
    
    # 1. 修改输出文件名
    # 使用正则表达式匹配 'output_filename = "..."'
    new_code = re.sub(
        r'output_filename\s*=\s*".*?"', 
        f'output_filename = r"{output_filename}"', 
        new_code
    )

    # 2. 修改物理参数
    for key, value in params.items():
        # 正则表达式匹配 'key = value' 或 'key_dim = value'
        # \s* 匹配任意空白字符, ([0-9eE.+-]+) 捕获数值部分
        pattern = re.compile(rf"({key}\s*=\s*)[0-9eE.+-]+")
        replacement = rf"\g<1>{value}"
        # 检查是否成功替换，如果失败则打印警告
        if not pattern.search(new_code):
            print(f"警告: 在 {strategy_script} 中未找到参数 '{key}' 的匹配项。")
        new_code = pattern.sub(replacement, new_code)

    # 创建一个临时脚本来运行
    temp_script_name = f"temp_runner_{os.path.basename(strategy_script)}"
    with open(temp_script_name, 'w', encoding='utf-8') as f:
        f.write(new_code)

    # 运行临时脚本
    print(f"--- 开始执行: {temp_script_name} ---")
    try:
        # 使用 -u 参数确保实时输出
        process = subprocess.Popen(
            ['python', '-u', temp_script_name], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8'
        )
        
        # 实时打印子进程的输出
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.wait() # 等待子进程结束
        
        if process.returncode == 0:
            print(f"--- {strategy_script} ({set_name}) 运行成功 ---")
        else:
            print(f"--- !!! {strategy_script} ({set_name}) 运行出错, 返回码: {process.returncode} !!! ---")

    except Exception as e:
        print(f"--- !!! 运行 {temp_script_name} 时发生异常: {e} !!! ---")
    finally:
        # 清理临时文件
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)
            print(f"--- 已清理临时文件: {temp_script_name} ---")


# --- 3. 主执行循环 ---
if __name__ == "__main__":
    # 确保数据输出目录存在
    output_dir = "simulation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 对每一组参数，运行两种策略
    for i, params_config in enumerate(parameter_sets):
        set_name = params_config['name']
        print(f"\n{'='*60}")
        print(f"开始处理第 {i+1} 组参数: {set_name}")
        print(f"{'='*60}\n")

        # 恒F策略
        output_f = os.path.join(output_dir, f"data_{i+1}_F_{set_name}.xlsx").replace('\\', '/')
        run_simulation('joint_F.py', params_config, output_f)

        # 恒a策略
        output_a = os.path.join(output_dir, f"data_{i+1}_a_{set_name}.xlsx").replace('\\', '/')
        run_simulation('joint_a.py', params_config, output_a)

    print("\n--- 所有模拟运行完毕 ---")
    print(f"输出数据已保存至 '{output_dir}' 文件夹。")
    print("接下来，请在Jupyter Notebook中运行新的绘图文件进行分析。")
