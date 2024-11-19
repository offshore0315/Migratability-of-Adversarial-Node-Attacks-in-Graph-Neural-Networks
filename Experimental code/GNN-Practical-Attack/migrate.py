import torch as th
import numpy as np
from model import GCN, GAT  # 根据需要导入您的模型
from utils import load_data, split_data


# 加载数据
migrate_dataset_name = "cora"  # 源，指对抗网络
dataset_name = "pubmed"  # 目标，指保存的模型

migrate_model_name = "GCN"  # 源，指对抗网络
model_name = "GAT"  # 目标，指保存的模型


# 初始化模型
data = load_data(dataset=dataset_name)

def get_model_and_args(model_name, data):
    if model_name == "GCN":
        model_args = {
            "in_feats": data.features.shape[1],
            "out_feats": data.num_labels,
            "n_units": 64,  # 设置为训练时使用的隐藏单元数量
            "dropout": 0.5,  # 设置为训练时的dropout
            "activation": "relu",
        }
        model = GCN(**model_args)
    elif model_name == "GAT":
        model_args = {
            "in_feats": data.features.shape[1],
            "out_feats": data.num_labels,
            "n_units": 8,       # 设置为训练时使用的隐藏单元数量
            "num_heads": 8,     # 设置为训练时使用的注意力头数量
            "dropout": 0.6,     # 设置为训练时的dropout
            "activation": "elu",# 设置为训练时的激活函数
        }
        model = GAT(**model_args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

# 获取目标模型
model = get_model_and_args(model_name, data)
# 加载训练好的模型
model.load_state_dict(th.load(f"./results/trained_{model_name}_{dataset_name}.pth"))
model.eval()  # 设置模型为评估模式

# 获取对抗模型（用于加载对抗特征时使用）
# 由于对抗特征是基于对抗模型生成的，我们需要知道对抗模型的参数
migrate_data = data  # 对于相同的数据集，可以直接使用
migrate_model = get_model_and_args(migrate_model_name, migrate_data)

# 评估模型
def evaluate(model, data, mask):
    with th.no_grad():
        logits = model(data)
        _, indices = th.max(logits[mask], dim=1)
        labels = data.labels[mask]
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# 确保参数为整数
NumTrain = int(data.size * 0.8)
NumTest = int(data.size * 0.1)
NumVal = data.size - NumTrain - NumTest  # 确保所有样本都被使用

# 获取训练、验证和测试索引
idx_train, idx_val, idx_test = split_data(data, NumTrain, NumVal, NumTest)

# 在替换对抗特征之前，评估模型
test_accuracy_original = evaluate(model, data, idx_test)
print("Test Accuracy on Original Samples: {:.4f}".format(test_accuracy_original))

# 加载对抗特征
adversarial_features_path = f"./results/adversarial_support_{migrate_model_name}_{migrate_dataset_name}_to_{dataset_name}.npy"
adversarial_features = np.load(adversarial_features_path)
data.features = th.tensor(adversarial_features, dtype=th.float32)

# 评估对抗攻击效果
test_accuracy = evaluate(model, data, idx_test)
print("Test Accuracy on Adversarial Samples: {:.4f}".format(test_accuracy))
