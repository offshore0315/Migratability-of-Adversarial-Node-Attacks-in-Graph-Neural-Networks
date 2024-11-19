import numpy as np

def adjust_data_format(source_path, target_path, output_path):
    # 加载数据
    source_data = np.load(source_path)
    target_data = np.load(target_path)

    # 输出数据的形状
    print("Source data shape:", source_data.shape)
    print("Target data shape:", target_data.shape)

    # 获取源和目标数据的行数和列数
    source_rows, source_cols = source_data.shape
    target_rows, target_cols = target_data.shape

    # 计算总元素数量
    source_size = source_rows * source_cols
    target_size = target_rows * target_cols

    if source_size < target_size:
        # 扩展源数据，重复填充
        repeat_rows = (target_rows + source_rows - 1) // source_rows
        repeat_cols = (target_cols + source_cols - 1) // source_cols

        # 使用 np.tile 进行重复填充
        expanded_data = np.tile(source_data, (repeat_rows, repeat_cols))

        # 截取到目标形状
        adjusted_data = expanded_data[:target_rows, :target_cols]
    elif source_size > target_size:
        # 缩减源数据，随机选择元素
        selected_rows = np.random.choice(source_rows, target_rows, replace=False)
        selected_cols = np.random.choice(source_cols, target_cols, replace=False)

        adjusted_data = source_data[np.ix_(selected_rows, selected_cols)]
    else:
        # 大小相同，无需调整
        adjusted_data = source_data

    # 保存处理后的数据
    np.save(output_path, adjusted_data)

    # 输出修改后的数据格式
    print("Modified data shape:", adjusted_data.shape)
    print("Modified data dtype:", adjusted_data.dtype)

# 文件路径变量
model = "GAT_"
source = "pubmed"
target = "pubmed"
source_file = './results/adversarial_support_' + model + source + '.npy'
target_file = './results/adversarial_support_' + model + target + '.npy'
output_file = './results/adversarial_support_' + model + source + '_to_' + target + '.npy'

# 调用函数
adjust_data_format(source_file, target_file, output_file)
