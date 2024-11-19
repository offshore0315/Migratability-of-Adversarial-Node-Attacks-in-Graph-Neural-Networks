import numpy as np


def adjust_data_format(source_path, target_path, output_path, method='repeat'):
    # Load data
    source_data = np.load(source_path)
    target_data = np.load(target_path)

    # Get shapes
    source_shape = source_data.shape  # e.g., (3327, 1851)
    target_shape = target_data.shape  # e.g., (2708, 716)

    # Initialize new data
    new_data = None

    if method == 'repeat':
        # Adjust the number of rows (samples)
        if source_shape[0] < target_shape[0]:
            # Repeat rows
            repeats = (target_shape[0] + source_shape[0] - 1) // source_shape[0]
            repeated_rows = np.tile(source_data, (repeats, 1))
            new_data = repeated_rows[:target_shape[0], :]
        elif source_shape[0] > target_shape[0]:
            # Truncate rows
            new_data = source_data[:target_shape[0], :]
        else:
            new_data = source_data

        # Adjust the number of columns (features)
        if source_shape[1] < target_shape[1]:
            # Repeat columns
            repeats = (target_shape[1] + source_shape[1] - 1) // source_shape[1]
            repeated_cols = np.tile(new_data, (1, repeats))
            new_data = repeated_cols[:, :target_shape[1]]
        elif source_shape[1] > target_shape[1]:
            # Truncate columns
            new_data = new_data[:, :target_shape[1]]
        # Else, shapes match; do nothing

    elif method == 'zero':
        # Initialize a new array with zeros
        new_data = np.zeros(target_shape, dtype=source_data.dtype)
        # Determine the minimum size along each dimension
        min_rows = min(source_shape[0], target_shape[0])
        min_cols = min(source_shape[1], target_shape[1])
        # Copy the source data into the new array
        new_data[:min_rows, :min_cols] = source_data[:min_rows, :min_cols]

    else:
        # Random truncation if source data is larger
        # Adjust rows
        if source_shape[0] > target_shape[0]:
            rows = np.random.choice(source_shape[0], target_shape[0], replace=False)
            new_data = source_data[rows, :]
        else:
            new_data = source_data

        # Adjust columns
        if source_shape[1] > target_shape[1]:
            cols = np.random.choice(source_shape[1], target_shape[1], replace=False)
            new_data = new_data[:, cols]

    # Save the processed data
    np.save(output_path, new_data)

    # Output the modified data format
    print("Modified data shape:", new_data.shape)
    print("Modified data dtype:", new_data.dtype)


# File path variables
model = "GCN"
source = "cora"
target = "pubmed"
source_file = f"save_model/main/{model}/{source}30/adversarial_samples"
target_file = f"save_model/main/{model}/{target}30/adversarial_samples"
output_file = source_file + '_to_' + target

# Example call
adjust_data_format(source_file + ".npy", target_file + ".npy", output_file + ".npy", method='repeat')
