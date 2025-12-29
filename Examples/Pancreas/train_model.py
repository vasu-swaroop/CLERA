import os
import pickle
import pandas as pd
import numpy as np
import torch

from src.torch_impl.config_utils import load_yaml_config, build_training_config
from src.torch_impl.training import train_network
from src.core.preprocess_utils import split_data

# Use the directory where this script is located as the default data path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Data is stored in the project's centralized data directory
project_root = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(project_root, 'data', 'Pancreas')

# Load YAML configuration
config_path = os.path.join(script_dir, 'train_config.yaml')
config = load_yaml_config(config_path)

def get_data():
    # Loading data from pickle files
    gene_names_path = os.path.join(data_path, 'gene_names.pkl')
    time_series_path = os.path.join(data_path, 'time_series.pkl')
    
    if not os.path.exists(gene_names_path):
        raise FileNotFoundError(f"Data file not found: {gene_names_path}")
    if not os.path.exists(time_series_path):
        raise FileNotFoundError(f"Data file not found: {time_series_path}")

    with open(gene_names_path, 'rb') as f:
        # Note: If these files are Git LFS pointers, pickle.load will fail.
        gene_names = pickle.load(f)
    with open(time_series_path, 'rb') as f:
        time_series = pickle.load(f)
    
    # Preparing data dictionary
    data_dict = {
        'x': time_series['x'],
        'dx': time_series['dx'],
        'classes': time_series['classes']
    }
    print("Successfully loaded data from disk.")
    return data_dict

# Load actual data
data_dict = get_data()

# Data splitting
training_data, val_data = split_data(data_dict, validation_ratio=0.1)

print(f"Training data shapes: x={training_data['x'].shape}, dx={training_data['dx'].shape}, classes={training_data['classes'].shape}")
print(f"Validation data shapes: x={val_data['x'].shape}, dx={val_data['dx'].shape}, classes={val_data['classes'].shape}")

# Class counts
print("Training class counts:\n", pd.Series(training_data['classes'].flatten()).value_counts())
print("Validation class counts:\n", pd.Series(val_data['classes'].flatten()).value_counts())

# Training Configuration
training_config, experiment_path = build_training_config(config, script_dir)

print(f"Experiment outputs will be saved to: {experiment_path}")

# Start training
print("Starting training...")
train_network(training_data, val_data, training_config)
