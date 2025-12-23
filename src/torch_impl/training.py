from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.loss_tracker import LossTracker
from .autoencoder import SINDyAEConfig, SINDyAE
from .losses import apply_sindy_ae_loss, compute_refinement_loss, compute_loss_components
from .utils import save_checkpoint, load_checkpoint, apply_coefficient_thresholding

@dataclass
class LossWeights:
    recon_wt: float
    sindy_wt_x: float
    sindy_wt_z: float
    class_wt: float
    l1_reg: float

@dataclass
class TrainSettings:
    optimizer: str
    lr: float
    refinement_epochs: int
    num_epochs: int
    batch_size: int
    threshold_frequency: int
    coefficient_threshold: float
    sequential_thresholding: bool

@dataclass
class TrainingConfig:
    sindy_ae_config: SINDyAEConfig
    loss_weights: LossWeights
    train_settings: TrainSettings
    print_progress: bool
    print_frequency: int
    plot_loss: bool
    load_model_path: Optional[str]
    save_model_path: Optional[str]


def train_network(
    training_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    training_config: TrainingConfig
):
    model = SINDyAE(sindy_ae_config=training_config.sindy_ae_config)
    train_settings = training_config.train_settings
    optimizer = Adam(model.parameters(), lr=train_settings.lr)
    
    loss_tracker = LossTracker()
    
    epoch_start = 0
    if training_config.load_model_path is not None:
        epoch_start = load_checkpoint(model, optimizer, training_config.load_model_path)
    
    train_dataloader = DataLoader(training_data, batch_size=train_settings.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=train_settings.batch_size, shuffle=False)
    
    # Training Phase
    print("=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)
    
    for epoch in tqdm(range(epoch_start, train_settings.num_epochs), desc="Training"):
        model.train()
        for inp_data in train_dataloader:
            optimizer.zero_grad()
            x = inp_data['x']
            dx = inp_data['dx']
            
            out_dict = model(x)
            loss = apply_sindy_ae_loss(model, training_config.loss_weights, out_dict, inp_data)
            loss.backward()
            optimizer.step()
        
        if training_config.print_progress and (epoch % training_config.print_frequency == 0):
            model.eval()
            with torch.no_grad():
                train_sample = next(iter(train_dataloader))
                train_out = model(train_sample['x'])
                train_losses = compute_loss_components(model, train_out, train_sample, training_config.loss_weights)
                loss_tracker.update_losses(train_losses, 'train')
                
                val_sample = next(iter(val_dataloader))
                val_out = model(val_sample['x'])
                val_losses = compute_loss_components(model, val_out, val_sample, training_config.loss_weights)
                loss_tracker.update_losses(val_losses, 'val')
                
                loss_tracker.print_losses(epoch, 'train+val')
        
        if train_settings.sequential_thresholding and (epoch % train_settings.threshold_frequency == 0) and (epoch > 0):
            num_terms = apply_coefficient_thresholding(model, train_settings.coefficient_threshold)
            loss_tracker.sindy_model_terms.append(num_terms)
            print(f"THRESHOLDING: {num_terms} active coefficients")
    
    # Refinement Phase
    print("\n" + "=" * 50)
    print("REFINEMENT PHASE")
    print("=" * 50)
    
    for epoch in tqdm(range(train_settings.refinement_epochs), desc="Refinement"):
        model.train()
        for inp_data in train_dataloader:
            optimizer.zero_grad()
            x = inp_data['x']
            dx = inp_data['dx']
            
            out_dict = model(x)
            loss = compute_refinement_loss(model, training_config.loss_weights, out_dict, inp_data)
            loss.backward()
            optimizer.step()
        
        if training_config.print_progress and (epoch % training_config.print_frequency == 0):
            model.eval()
            with torch.no_grad():
                train_sample = next(iter(train_dataloader))
                train_out = model(train_sample['x'])
                train_losses = compute_loss_components(model, train_out, train_sample, training_config.loss_weights)
                loss_tracker.update_refinement_losses(train_losses, 'train')
                
                val_sample = next(iter(val_dataloader))
                val_out = model(val_sample['x'])
                val_losses = compute_loss_components(model, val_out, val_sample, training_config.loss_weights)
                loss_tracker.update_refinement_losses(val_losses, 'val')
                
                loss_tracker.print_losses(epoch, 'refinement')
    
    if training_config.save_model_path is not None:
        save_checkpoint(model, optimizer, train_settings.num_epochs + train_settings.refinement_epochs, 
                       training_config.save_model_path)
    
    if training_config.plot_loss:
        loss_tracker.plot_all_losses(
            save_name=training_config.save_model_path or "model",
            print_frequency=training_config.print_frequency
        )
    
    return model, loss_tracker