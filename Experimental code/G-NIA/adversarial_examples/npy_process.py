import numpy as np

def adjust_adv_examples_format(source_path, target_path, output_path):
    # 加载源（cora）和目标（citeseer）的对抗样本数据
    source_data = np.load(source_path, allow_pickle=True)
    target_data = np.load(target_path, allow_pickle=True)

    print(f"Original source data type: {type(source_data)}")
    print(f"Original source data dtype: {source_data.dtype}")
    print(f"Original source data shape: {source_data.shape}")
    print(f"Original target data type: {type(target_data)}")
    print(f"Original target data dtype: {target_data.dtype}")
    print(f"Original target data shape: {target_data.shape}")

    # 提取 source_data 字典
    if isinstance(source_data, np.ndarray) and source_data.size == 1:
        source_data = source_data.item()
        print("Extracted source data as dict.")
    elif isinstance(source_data, dict):
        print("Source data is already a dict.")
    else:
        print("Source data is in unexpected format. Exiting.")
        return

    # 提取 target_data 字典并获取目标特征维度
    if isinstance(target_data, np.ndarray) and target_data.size == 1:
        target_data = target_data.item()
        print("Extracted target data as dict.")
    elif isinstance(target_data, dict):
        print("Target data is already a dict.")
    else:
        print("Target data is in unexpected format. Exiting.")
        return

    # 从 target_data 中获取目标特征维度
    target_inj_feats = []
    for dset in ['train', 'val', 'test']:
        if dset in target_data and 'inj_feat' in target_data[dset]:
            inj_feats = target_data[dset]['inj_feat']
            print(f"{dset} - Number of inj_feats: {len(inj_feats)}")
            for idx, inj_feat in enumerate(inj_feats):
                inj_feat = np.array(inj_feat)  # 将 inj_feat 转换为 NumPy 数组
                print(f"target inj_feat[{idx}] shape: {inj_feat.shape}")
                target_inj_feats.append(inj_feat)
    if len(target_inj_feats) == 0:
        print("No inj_feat found in target_data.")
        return
    # 检查 inj_feat 的形状是否至少有两个维度
    print(f"First target_inj_feats[0] shape: {target_inj_feats[0].shape}")
    if len(target_inj_feats[0].shape) < 2:
        print(f"Error: inj_feat has shape {target_inj_feats[0].shape}, expected at least 2 dimensions.")
        # 尝试调整 inj_feat 的形状
        target_inj_feats = [inj_feat.reshape(1, -1) if len(inj_feat.shape) == 1 else inj_feat for inj_feat in target_inj_feats]
        print(f"After reshaping, target_inj_feats[0] shape: {target_inj_feats[0].shape}")
        if len(target_inj_feats[0].shape) < 2:
            print("Unable to reshape inj_feat to have at least 2 dimensions. Exiting.")
            return
    target_feat_dim = target_inj_feats[0].shape[1]
    print(f"Target feature dimension: {target_feat_dim}")

    # 调整 source_data 中的 inj_feat
    for dset in ['train', 'val', 'test']:
        if dset in source_data and 'inj_feat' in source_data[dset]:
            adjusted_inj_feats = []
            for idx, inj_feat in enumerate(source_data[dset]['inj_feat']):
                inj_feat = np.array(inj_feat)
                print(f"source inj_feat[{idx}] original shape: {inj_feat.shape}")
                if len(inj_feat.shape) == 1:
                    inj_feat = inj_feat.reshape(1, -1)
                    print(f"source inj_feat[{idx}] reshaped to: {inj_feat.shape}")
                # if inj_feat.shape[1] < target_feat_dim:
                #     # 进行零填充
                #     padding = np.zeros((inj_feat.shape[0], target_feat_dim - inj_feat.shape[1]))
                #     inj_feat = np.hstack((inj_feat, padding))
                # elif inj_feat.shape[1] > target_feat_dim:
                #     # 进行截断
                #     inj_feat = inj_feat[:, :target_feat_dim]
                if inj_feat.shape[1] < target_feat_dim:
                    # 通过重复特征来扩充特征维度
                    times = (target_feat_dim + inj_feat.shape[1] - 1) // inj_feat.shape[1]
                    expanded_feat = np.tile(inj_feat, (1, times))
                    inj_feat = expanded_feat[:, :target_feat_dim]
                elif inj_feat.shape[1] > target_feat_dim:
                    # 通过随机选择特征来缩减特征维度
                    indices = np.random.choice(inj_feat.shape[1], target_feat_dim, replace=False)
                    inj_feat = inj_feat[:, indices]

                # 否则，特征维度匹配，无需调整
                print(f"Adjusted source inj_feat[{idx}] shape: {inj_feat.shape}")
                adjusted_inj_feats.append(inj_feat)
            source_data[dset]['inj_feat'] = adjusted_inj_feats

    # 保存调整后的 source_data
    np.save(output_path, source_data, allow_pickle=True)
    print(f"Adjusted source data saved to {output_path}")

# 文件路径变量
model = "gcn"
source = "citeseer"
target = "citeseer"
source_file = source + "_" + model + ".npy"
target_file = target + "_" + model + ".npy"
output_file = source + "_" + model + '_to_' + target + ".npy"

# 调整 cora_gcn.npy 以匹配 citeseer_gcn.npy
adjust_adv_examples_format(source_file, target_file, output_file)
