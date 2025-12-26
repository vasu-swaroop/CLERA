from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F



def get_sindy_z_loss(dz_true: torch.Tensor, dz_sindy_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(dz_sindy_pred, dz_true)


def get_sindy_x_loss(dx_true: torch.Tensor, dx_sindy_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(dx_sindy_pred, dx_true)


def get_class_loss(class_logits: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
    if class_labels.dim() == 1:
        return F.cross_entropy(class_logits, class_labels)
    else:
        return F.binary_cross_entropy_with_logits(class_logits, class_labels)


def get_recon_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x)


def l1_regularization(params: List[torch.nn.Parameter]) -> torch.Tensor:
    l1_norm = torch.tensor(0.0, device=params[0].device)
    for param in params:
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm


def sindy_regularization(sindy_coefficients: torch.Tensor, 
                         coefficient_mask: torch.Tensor) -> torch.Tensor:
    masked_coeffs = sindy_coefficients * coefficient_mask
    return torch.sum(torch.abs(masked_coeffs))


def apply_sindy_ae_loss(
    model: nn.Module,
    loss_weights: 'LossWeights',
    out_dict: Dict[str, torch.Tensor],
    inp_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    x = inp_data['x']
    dx = inp_data['dx']
    
    x_recon = out_dict['x_recon']
    z = out_dict['z']
    enc_grads = out_dict['enc_grads']
    dec_grads = out_dict['dec_grads']
    sindy_predict = out_dict['sindy_predict']
    
    loss_recon = get_recon_loss(x, x_recon)
    
    # enc_grads: (B, latent_dim, D), dx: (B, D) -> dz_true: (B, latent_dim)
    dz_true = torch.einsum('bld,bd->bl', enc_grads, dx)
    loss_sindy_z = get_sindy_z_loss(dz_true, sindy_predict)
    
    # dec_grads: (B, D, latent_dim), sindy_predict: (B, latent_dim) -> dx_sindy_pred: (B, D)
    dx_sindy_pred = torch.einsum('bdl,bl->bd', dec_grads, sindy_predict)
    loss_sindy_x = get_sindy_x_loss(dx, dx_sindy_pred)
    
    loss_class = torch.tensor(0.0, device=x.device)
    if 'class_score' in out_dict and 'class_labels' in inp_data:
        class_score = out_dict['class_score']
        class_labels = inp_data['class_labels']
        loss_class = get_class_loss(class_score, class_labels)
    
    loss_sindy_reg = sindy_regularization(
        model.sindy.coefficients, 
        model.sindy.coefficient_mask
    )
    
    ae_params = (
        list(model.encoder.parameters()) + 
        list(model.decoder.parameters())
    )
    loss_ae_reg = l1_regularization(ae_params) * loss_weights.l1_reg
    
    total_loss = (
        loss_weights.recon_wt * loss_recon +
        loss_weights.sindy_wt_z * loss_sindy_z +
        loss_weights.sindy_wt_x * loss_sindy_x +
        loss_weights.class_wt * loss_class +
        loss_sindy_reg +
        loss_ae_reg
    )
    
    return total_loss


def compute_loss_components(
    model: nn.Module,
    out_dict: Dict[str, torch.Tensor],
    inp_data: Dict[str, torch.Tensor],
    loss_weights: 'LossWeights'
) -> Dict[str, float]:
    x = inp_data['x']
    dx = inp_data['dx']
    x_recon = out_dict['x_recon']
    enc_grads = out_dict['enc_grads']
    dec_grads = out_dict['dec_grads']
    sindy_predict = out_dict['sindy_predict']
    
    loss_recon = get_recon_loss(x, x_recon)
    
    # enc_grads: (B, latent_dim, D), dx: (B, D) -> dz_true: (B, latent_dim)
    dz_true = torch.einsum('bld,bd->bl', enc_grads, dx)
    loss_sindy_z = get_sindy_z_loss(dz_true, sindy_predict)
    
    # dec_grads: (B, D, latent_dim), sindy_predict: (B, latent_dim) -> dx_sindy_pred: (B, D)
    dx_sindy_pred = torch.einsum('bdl,bl->bd', dec_grads, sindy_predict)
    loss_sindy_x = get_sindy_x_loss(dx, dx_sindy_pred)
    
    loss_class = torch.tensor(0.0, device=x.device)
    if 'class_score' in out_dict and 'class_labels' in inp_data:
        loss_class = get_class_loss(out_dict['class_score'], inp_data['class_labels'])
    
    loss_sindy_reg = sindy_regularization(model.sindy.coefficients, model.sindy.coefficient_mask)
    
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    loss_ae_reg = l1_regularization(ae_params) * loss_weights.l1_reg
    
    total = (
        loss_recon * loss_weights.recon_wt + 
        loss_sindy_z * loss_weights.sindy_wt_z + 
        loss_sindy_x * loss_weights.sindy_wt_x + 
        loss_class * loss_weights.class_wt + 
        loss_sindy_reg + 
        loss_ae_reg
    )
    
    return {
        'total': total.item(),
        'recon': loss_recon.item(),
        'sindy_z': loss_sindy_z.item(),
        'sindy_x': loss_sindy_x.item(),
        'sindy_reg': loss_sindy_reg.item(),
        'ae_reg': loss_ae_reg.item(),
        'class': loss_class.item()
    }


def compute_refinement_loss(
    model: nn.Module,
    loss_weights: 'LossWeights',
    out_dict: Dict[str, torch.Tensor],
    inp_data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    x = inp_data['x']
    dx = inp_data['dx']
    
    x_recon = out_dict['x_recon']
    enc_grads = out_dict['enc_grads']
    dec_grads = out_dict['dec_grads']
    sindy_predict = out_dict['sindy_predict']
    
    loss_recon = get_recon_loss(x, x_recon)
    
    # enc_grads: (B, latent_dim, D), dx: (B, D) -> dz_true: (B, latent_dim)
    dz_true = torch.einsum('bld,bd->bl', enc_grads, dx)
    loss_sindy_z = get_sindy_z_loss(dz_true, sindy_predict)
    
    # dec_grads: (B, D, latent_dim), sindy_predict: (B, latent_dim) -> dx_sindy_pred: (B, D)
    dx_sindy_pred = torch.einsum('bdl,bl->bd', dec_grads, sindy_predict)
    loss_sindy_x = get_sindy_x_loss(dx, dx_sindy_pred)
    
    loss_class = torch.tensor(0.0, device=x.device)
    if 'class_score' in out_dict and 'class_labels' in inp_data:
        loss_class = get_class_loss(out_dict['class_score'], inp_data['class_labels'])
    
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    loss_ae_reg = l1_regularization(ae_params) * loss_weights.l1_reg
    
    total_loss = (
        loss_weights.recon_wt * loss_recon +
        loss_weights.sindy_wt_z * loss_sindy_z +
        loss_weights.sindy_wt_x * loss_sindy_x +
        loss_weights.class_wt * loss_class +
        loss_ae_reg
    )
    
    return total_loss