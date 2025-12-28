import torch


def save_checkpoint(model, optimizer, epoch, save_path):
    import torch
    torch.save({
        'model': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    import torch
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch']


@torch.no_grad()
def apply_coefficient_thresholding(model, threshold):
    coeffs = model.sindy.coefficients
    new_mask = (torch.abs(coeffs) > threshold).float()
    model.sindy.coefficient_mask.data *= new_mask
    return int(model.sindy.coefficient_mask.sum().item())


def load_for_transfer_learning(model, load_path, load_classifier=False):
    """
    Load pretrained weights for transfer learning.
    
    Loads:
      - Encoder weights
      - Decoder weights  
      - SINDy coefficients
      - Optionally: classifier weights
    
    Does NOT load:
      - Coefficient mask (fresh mask allows new sparsity pattern discovery)
      - Optimizer state
    
    Args:
        model: SINDyAE model to load weights into
        load_path: Path to checkpoint file
        load_classifier: Whether to load classifier head weights (default: False)
    
    Returns:
        None
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    pretrained_state = checkpoint['model']
    
    # Get current model state
    model_state = model.state_dict()
    
    # Keys to load (everything except coefficient_mask)
    keys_to_load = []
    
    for key in pretrained_state.keys():
        # Skip coefficient mask - we want a fresh mask for transfer learning
        if 'coefficient_mask' in key:
            continue
        
        # Skip classifier if not requested
        if not load_classifier and 'classification_head' in key:
            continue
            
        # Check if key exists in current model with matching shape
        if key in model_state:
            if pretrained_state[key].shape == model_state[key].shape:
                keys_to_load.append(key)
            else:
                print(f"  Skipping {key}: shape mismatch "
                      f"(pretrained {pretrained_state[key].shape} vs "
                      f"model {model_state[key].shape})")
    
    # Load the filtered state dict
    filtered_state = {k: pretrained_state[k] for k in keys_to_load}
    model.load_state_dict(filtered_state, strict=False)
    
    print(f"Transfer learning: loaded {len(keys_to_load)} weight tensors from {load_path}")
    print(f"  - Encoder: loaded")
    print(f"  - Decoder: loaded")
    print(f"  - SINDy coefficients: loaded")
    print(f"  - Coefficient mask: fresh (not loaded)")
    print(f"  - Classifier: {'loaded' if load_classifier else 'fresh (not loaded)'}")
