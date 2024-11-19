import numpy as np

def adjust_data_format(source_path, target_path, output_path, method='repeat'):
    # Load data
    source_data = np.load(source_path)

    # Check if source and target paths are the same
    if source_path == target_path:
        # Save the source data directly to the output path
        np.save(output_path, source_data)
        print("Source and target are identical. File has been saved with new name:", output_path)
        return

    # Load target data if paths are different
    target_data = np.load(target_path)

    # Get the shape of the target data, ignoring the first dimension
    target_shape = target_data.shape[1:]

    # Adjust the shape of the source data, assuming first dimension is batch size
    source_reshaped = source_data[0]

    if source_reshaped.shape[0] < target_shape[0] or source_reshaped.shape[1] < target_shape[1]:
        if method == 'repeat':
            # Extend source data by repeating
            repeated_rows = (target_shape[0] + source_reshaped.shape[0] - 1) // source_reshaped.shape[0]
            repeated_cols = (target_shape[1] + source_reshaped.shape[1] - 1) // source_reshaped.shape[1]
            expanded_data = np.tile(source_reshaped, (repeated_rows, repeated_cols))
            source_reshaped = expanded_data[:target_shape[0], :target_shape[1]]
        elif method == 'zero':
            # Pad with zeros
            new_data = np.zeros(target_shape, dtype=np.float32)
            new_data[:source_reshaped.shape[0], :source_reshaped.shape[1]] = source_reshaped
            source_reshaped = new_data
    else:
        # Random truncation if source data is larger
        rows = np.random.choice(source_reshaped.shape[0], target_shape[0], replace=False)
        cols = np.random.choice(source_reshaped.shape[1], target_shape[1], replace=False)
        source_reshaped = source_reshaped[rows][:, cols]

    # Add a new dimension
    final_data = np.expand_dims(source_reshaped, axis=0)

    # Save the processed data
    np.save(output_path, final_data)

    # Output the modified data format
    print("Modified data shape:", final_data.shape)
    print("Modified data dtype:", final_data.dtype)

# File path variables
model = "GAT"
source = "citeseer"
target = "citeseer"
source_file = 'adversarial_support_' + model + "_" + source
target_file = 'adversarial_support_' + model + "_" + target
output_file = source_file + '_to_' + target

# Example call
adjust_data_format(source_file + ".npy", target_file + ".npy", output_file + ".npy", method='repeat')
