import numpy as np
from sklearn.model_selection import train_test_split

dataset_name = "cora"

# Load the npz file data
npz_file_path = f"{dataset_name}.npz"
npz_data = np.load(npz_file_path)

# Extract labels and node indices from the npz file
labels = npz_data['labels']
node_indices = np.arange(len(labels))  # Assuming node indices are sequential

train_indices, temp_indices = train_test_split(
    node_indices, test_size=0.2, stratify=labels, random_state=42)

val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=labels[temp_indices], random_state=42)

# Create the split dictionary
split_data = {
    'train': train_indices,
    'val': val_indices,
    'test': test_indices
}

# Save the split data to an npy file
split_npy_path = f'{dataset_name}_split.npy'
np.save(split_npy_path, split_data)


def check_split_ratios(npy_file_path):
    # Load the split data from the npy file
    split_data = np.load(npy_file_path, allow_pickle=True).item()

    # Extract the indices for train, val, and test
    train_indices = split_data['train']
    val_indices = split_data['val']
    test_indices = split_data['test']

    # Calculate lengths
    total_nodes = len(train_indices) + len(val_indices) + len(test_indices)
    train_ratio = len(train_indices) / total_nodes
    val_ratio = len(val_indices) / total_nodes
    test_ratio = len(test_indices) / total_nodes

    # Print ratios
    print(f'Train Ratio: {train_ratio:.2%}')
    print(f'Validation Ratio: {val_ratio:.2%}')
    print(f'Test Ratio: {test_ratio:.2%}')

# Example usage
npy_file_path = f'{dataset_name}_split.npy'  # 输入分割文件路径
check_split_ratios(npy_file_path)
