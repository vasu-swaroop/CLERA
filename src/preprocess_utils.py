import pandas as pd
import numpy as np
def window_average_with_stride(df, window_size, stride, allowed_clusters, num_genes=2000):
    results = []
    # Assuming the last column is categorical and the rest are numeric
    numeric_indices = list(range(num_genes))  # Adjust if your indices start from 1 or another number
    categorical_col = df.columns[num_genes]  # Adjust based on zero-indexing

    for i in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[i:i+window_size]
        aggregated_row = {}
        # Aggregate numeric columns
        poss=False
        # Aggregate the categorical column
        cell_types=set(window[categorical_col])
        if  cell_types in allowed_clusters:
            # If all elements belong to allowed clusters, compute mode, else NaN or other logic
            value_counts = window[categorical_col].value_counts()
            max_count_category = value_counts.idxmax()
            aggregated_row[categorical_col] = max_count_category
            poss=True
        if poss:
            aggregated_row.update(window.iloc[:, numeric_indices].mean().to_dict())
        results.append(aggregated_row)
    result_df = pd.DataFrame(results)
    return result_df
def split_data(data_dict, validation_ratio=0.2, seed=None):
    """
    Splits the data dictionary into training and validation sets.

    Parameters:
        data_dict (dict): The data dictionary with 'x' and 'dx' arrays.
        validation_ratio (float): The ratio of validation data (default: 0.2).
        seed (int): Seed for random number generator (optional).

    Returns:
        tuple: Two dictionaries: (training_dict, validation_dict).
    """
    

    x_array = data_dict['x']
    dx_array = data_dict['dx']
    class_ = data_dict['classes']

    # Get the number of samples
    num_samples = x_array.shape[0]
        
    # Shuffle the indices of the samples
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Calculate the size of the validation set
    validation_size = int(num_samples * validation_ratio)

    # Get the indices for validation and training sets
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    # Create the training and validation dictionaries
    training_dict = {
        'x': x_array[training_indices],
        'dx': dx_array[training_indices],
        'classes': class_[training_indices]
    }

    validation_dict = {
        'x': x_array[validation_indices],
        'dx': dx_array[validation_indices],
        'classes': class_[validation_indices]

    }

    return training_dict, validation_dict
def xavier_initializer(shape):
    """
    Xavier initializer for weights.
    
    Arguments:
        shape (tuple): Shape of the weight tensor.
    
    Returns:
        np.ndarray: Initialized weight tensor.
    """
    variance = 2.0 / (shape[0] + shape[1])  # Xavier variance calculation
    stddev = np.sqrt(variance)
    return np.random.normal(0, stddev, shape)

def build_network_layers(input_dim, output_dim, widths, name):
    """
    Construct one portion of the network (either encoder or decoder).
    Arguments:
        input (Tensor): 2D TensorFlow array, input to the network (shape is [?, input_dim]).
        input_dim (int): Integer, number of state variables in the input to the first layer.
        output_dim (int): Integer, number of state variables to output from the final layer.
        widths (List[int]): List of integers representing how many units are in each network layer.
        activation (function): TensorFlow activation function to be used at each layer.
        name (str): Prefix to be used in naming the TensorFlow variables.
    Returns:
        output (Tensor): TensorFlow array, output of the network layers (shape is [?, output_dim]).
        weights (List[Tensor]): List of TensorFlow arrays containing the network weights.
        biases (List[Tensor]): List of TensorFlow arrays containing the network biases.
    """
    weights = []
    last_width = input_dim
    for i, n_units in enumerate(widths):
        W = np.float32(xavier_initializer(shape=[last_width, n_units]))
        last_width = n_units
        weights.append(W)
    W = np.float32(xavier_initializer(shape=[last_width, output_dim]))
    weights.append(W)
    if name=="encoder":
        W = np.float32(xavier_initializer(shape=[last_width, output_dim]))
        weights.append(W)
    
    return weights
def coefficient_innit(library_dim, latent_dim):
    sindy_coefficients = xavier_initializer([library_dim,latent_dim])
    return np.float32(sindy_coefficients)

