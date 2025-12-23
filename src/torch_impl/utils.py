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


def apply_coefficient_thresholding(model, threshold):
    import torch
    with torch.no_grad():
        coeffs = model.sindy.coefficients
        mask = (torch.abs(coeffs) > threshold).float()
        model.sindy.coefficient_mask.data = mask
        return int(mask.sum().item())
