from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F


def get_sindy_z_loss(dz_true: torch.Tensor, dz_sindy_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(dz_sindy_pred, dz_true)


def get_sindy_x_loss(dx_true: torch.Tensor, dx_sindy_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(dx_sindy_pred, dx_true)


def get_class_loss(class_logits: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
    """Multiclass cross-entropy loss. Handles both index labels and one-hot encoded labels."""
    if class_labels.dim() > 1:
        # One-hot encoded -> convert to class indices
        class_labels = class_labels.argmax(dim=1)
    return F.cross_entropy(class_logits, class_labels)


def get_recon_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x)


def l1_regularization(params: List[torch.nn.Parameter]) -> torch.Tensor:
    return sum(torch.sum(torch.abs(p)) for p in params)


def sindy_regularization(coefficients: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(coefficients * mask))


def compute_losses(
    model: nn.Module,
    out_dict: Dict[str, torch.Tensor],
    inp_data: Dict[str, torch.Tensor],
    loss_weights: 'LossWeights',
    include_sindy_reg: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Unified loss computation. Returns dict with individual losses and total.
    
    Args:
        include_sindy_reg: If True (training phase), include SINDy L1 regularization.
                          If False (refinement phase), exclude it.
    """
    x, dx = inp_data['x'], inp_data['dx']
    x_recon = out_dict['x_recon']
    enc_grads, dec_grads = out_dict['enc_grads'], out_dict['dec_grads']
    sindy_predict = out_dict['sindy_predict']
    
    # Core losses
    recon = get_recon_loss(x, x_recon)
    dz_true = torch.einsum('bld,bd->bl', enc_grads, dx)
    sindy_z = get_sindy_z_loss(dz_true, sindy_predict)
    dx_pred = torch.einsum('bdl,bl->bd', dec_grads, sindy_predict)
    sindy_x = get_sindy_x_loss(dx, dx_pred)
    
    # Optional classification - handle both key names
    class_loss = torch.tensor(0.0, device=x.device)
    class_labels_key = 'class_labels' if 'class_labels' in inp_data else 'classes' if 'classes' in inp_data else None
    if 'class_score' in out_dict and class_labels_key:
        class_loss = get_class_loss(out_dict['class_score'], inp_data[class_labels_key])
    
    # Regularization
    sindy_reg = sindy_regularization(model.sindy.coefficients, model.sindy.coefficient_mask) if include_sindy_reg else torch.tensor(0.0, device=x.device)
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    ae_reg = l1_regularization(ae_params) * loss_weights.l1_reg
    
    # Aggregate
    total = (
        loss_weights.recon_wt * recon +
        loss_weights.sindy_wt_z * sindy_z +
        loss_weights.sindy_wt_x * sindy_x +
        loss_weights.class_wt * class_loss +
        sindy_reg + ae_reg
    )
    
    return {
        'total': total,
        'recon': recon,
        'sindy_z': sindy_z,
        'sindy_x': sindy_x,
        'sindy_reg': sindy_reg,
        'ae_reg': ae_reg,
        'class': class_loss
    }


# Convenience wrappers for backward compatibility
def apply_sindy_ae_loss(model, loss_weights, out_dict, inp_data) -> torch.Tensor:
    """Training phase loss (with SINDy regularization)."""
    return compute_losses(model, out_dict, inp_data, loss_weights, include_sindy_reg=True)['total']


def compute_refinement_loss(model, loss_weights, out_dict, inp_data) -> torch.Tensor:
    """Refinement phase loss (without SINDy regularization)."""
    return compute_losses(model, out_dict, inp_data, loss_weights, include_sindy_reg=False)['total']


def compute_loss_components(model, out_dict, inp_data, loss_weights) -> Dict[str, float]:
    """For logging - returns dict of floats."""
    losses = compute_losses(model, out_dict, inp_data, loss_weights, include_sindy_reg=True)
    return {k: v.item() for k, v in losses.items()}