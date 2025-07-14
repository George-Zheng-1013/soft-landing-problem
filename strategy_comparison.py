import subprocess
import json
import pandas as pd
import os

# --- 1. 定义多组着陆器参数 ---
# 每组参数是一个字典，包含了需要被修改的初始条件
parameter_sets = [
    {
        "name": "标准着陆器",
        "m0_dim": 3762,
        "F_max_dim": 7500.0,
        "F_min_dim": 1500.0,
        "H0_dim": 11700.0,
        "vt0_dim": 1762.83
    },
    {
        "name": "轻型着陆器",
        "m0_dim": 3200,
        "F_max_dim": 7500.0,
        "F_min_dim": 1500.0,
        "H0_dim": 11700.0,
        "vt0_dim": 1762.83
    },
    {
        "name": "强劲引擎着陆器",
        "m0_dim": 3762,
        "F_max_dim": 8500.0,
        "F_min_dim": 2000.0,
        "H0_dim": 11700.0,
        "vt0_dim": 1762.83
    }
]

# --- 2. 定义运行优化策略的函数 ---
def run_simulation(strategy_script, params, output_filename):
    """
    通过命令行运行指定的策略脚本，并传递参数。

    Args:
        strategy_script (str): 要运行的Python脚本文件名 (e.g., 'joint_F.py').
        params (dict): 包含着陆器参数的字典。
        output_filename (str): 保存输出数据的文件名。
    """
    print(f"--- 正在运行: {strategy_script} 使用参数: {params['name']} ---")
    
    # 将参数转换为JSON字符串以便于命令行传递
    params_json = json.dumps(params)
    
    # 构建命令行
    # 我们将参数作为命令行参数传递给一个修改后的脚本
    # 为了不修改源文件，我们先读取源文件，修改它，保存为临时文件，然后运行
    
    with open(strategy_script, 'r', encoding='utf-8') as f:
        original_code = f.read()

    # 替换输出文件名
    new_code = original_code.replace(
        '"lunar_landing_optimization_data.xlsx"', f'"{output_filename}"'
    ).replace(
        '"lunar_landing_joint_a_data.xlsx"', f'"{output_filename}"'
    )

    # 替换参数值
    for key, value in params.items():
        # 寻找类似 `m0_dim = 3762` 的行并替换它
        new_code = new_code.replace(
            f'{key}_dim = {pd.read_csv(pd.io.common.StringIO(original_code), sep="=").set_index(0).loc[f"{key}_dim"][1]}',
            f'{key}_dim = {value}'
        )

    # 创建一个临时脚本来运行
    temp_script_name = f"temp_{strategy_script}"
    with open(temp_script_name, 'w', encoding='utf-8') as f:
        f.write(new_code)

    # 运行临时脚本
    try:
        result = subprocess.run(['python', temp_script_name], capture_output=True, text=True, check=True, encoding='utf-8')
        print(f"--- {strategy_script} ({params['name']}) 运行成功 ---")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"--- !!! {strategy_script} ({params['name']}) 运行失败 !!! ---")
        print(e.stderr)
    finally:
        # 清理临时文件
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)


# --- 3. 主执行循环 ---
if __name__ == "__main__":
    # 确保数据输出目录存在
    output_dir = "data_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 对每一组参数，运行两种策略
    for i, params in enumerate(parameter_sets):
        print(f"\n=================================================")
        print(f"开始处理第 {i+1} 组参数: {params['name']}")
        print(f"=================================================\n")

        # 恒F策略
        output_f = os.path.join(output_dir, f"data_set_{i+1}_const_F.xlsx")
        run_simulation('joint_F.py', params, output_f)

        # 恒a策略
        output_a = os.path.join(output_dir, f"data_set_{i+1}_const_a.xlsx")
        run_simulation('joint_a.py', params, output_a)

    print("\n--- 所有模拟运行完毕 ---")
    print(f"输出数据已保存至 '{output_dir}' 文件夹。")

