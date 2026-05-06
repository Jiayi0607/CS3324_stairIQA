import numpy as np

# 替换为你的npz文件路径
npz_file_path = "all_experiments_summary.npz"

# 读取并打印所有内容
with np.load(npz_file_path, allow_pickle=True) as data:
    for key in data.files:
        print(f"=== 数组名称: {key} ===")
        print(f"内容:\n{data[key]}\n")