# Inference Utils

Shared utility functions for both TensorFlow and PyTorch inference notebooks.

## Usage

```python
from utils.inference_utils import load_data, load_model_components

# Load dataset
data = load_data('Pancreas')

# Load model components
components = load_model_components('path/to/chosen_exp_components')
```

## Functions

- `load_data(dataset_name)` - Load preprocessed time series
- `load_model_components(components_dir)` - Load saved model weights and components
- `calculate_errors(predictions, targets)` - Calculate relative errors
- `print_equations(active_terms, ...)` - Print SINDy equations
- `save_components(save_dir, **kwargs)` - Save model components
