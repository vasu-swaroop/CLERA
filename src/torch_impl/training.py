from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn

from src.torch_impl.autoencoder import SINDyAEConfig, SINDyAE
from src.torch_impl.losses import apply_sindy_ae_loss, compute_refinement_loss, compute_loss_components, LossWeights
from src.torch_impl.utils import save_checkpoint, load_checkpoint, apply_coefficient_thresholding
from src.core.loss_tracker import LossTracker

class SINDyDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.x = torch.from_numpy(data_dict['x']).float()
        self.dx = torch.from_numpy(data_dict['dx']).float()
        self.classes = torch.from_numpy(data_dict['classes']).long()
        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'dx': self.dx[idx],
            'classes': self.classes[idx]
        }


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
    max_active_terms: int = None

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
    
    train_dataset = SINDyDataset(training_data)
    val_dataset = SINDyDataset(val_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_settings.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_settings.batch_size, shuffle=False)
    
    is_cuda= torch.cuda.is_available()
    if is_cuda:
        device= "cuda:0"
    else:
        device= "cpu"

    model.to(device)
    # Training Phase
    print("=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)

    num_terms=int(model.sindy.coefficient_mask.sum().item())
    for epoch in tqdm(range(epoch_start, train_settings.num_epochs), desc="Training"):
        model.train()
        for inp_data in train_dataloader:
            optimizer.zero_grad()
            for key, val in inp_data.items():
                inp_data[key]=val.to(device)
            x = inp_data['x']
            dx = inp_data['dx']
            
            out_dict = model(x)
            loss = apply_sindy_ae_loss(model, training_config.loss_weights, out_dict, inp_data)
            loss.backward()
            optimizer.step()
        
        if training_config.print_progress and (epoch % training_config.print_frequency == 0):
            model.eval()
            with torch.no_grad():
                val_sample = next(iter(val_dataloader))
                for key, val in val_sample.items():
                    val_sample[key]=val.to(device)

                val_out = model(val_sample['x'])
                val_losses = compute_loss_components(model, val_out, val_sample, training_config.loss_weights)
                loss_tracker.update_losses(val_losses, 'val')

        #Apply Sequential thresholding               
        if train_settings.sequential_thresholding and (epoch % train_settings.threshold_frequency == 0) and (epoch > 0):
            num_terms = apply_coefficient_thresholding(model, train_settings.coefficient_threshold)
            loss_tracker.sindy_model_terms.append(num_terms)
            print(f"THRESHOLDING: {num_terms} active coefficients")
        
        if num_terms<train_settings.max_active_terms:
            print("Maximum active terms in RHS achieved, begining the refinement phase")
            break
    
    # Refinement Phase
    print("\n" + "=" * 50)
    print("REFINEMENT PHASE")
    print("=" * 50)
    
    for epoch in tqdm(range(train_settings.refinement_epochs), desc="Refinement"):
        model.train()
        for inp_data in train_dataloader:
            optimizer.zero_grad()
            for key, val in inp_data.items():
                inp_data[key]=val.to(device)
            x = inp_data['x']
            dx = inp_data['dx']
            
            out_dict = model(x)
            loss = compute_refinement_loss(model, training_config.loss_weights, out_dict, inp_data)
            loss.backward()
            optimizer.step()
        
        if training_config.print_progress and (epoch % training_config.print_frequency == 0):
            model.eval()
            with torch.no_grad():                
                val_sample = next(iter(val_dataloader))
                for key, val in val_sample.items():
                    val_sample[key]=val.to(device)


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

def generate_dummy_data(data_dim, class_size, batch_size):
    """Generate dummy data for testing"""
    x = torch.randn(batch_size, data_dim)
    dx = torch.randn(batch_size, data_dim)
    class_batch = torch.randint(0, class_size, (batch_size,))
    class_labels = torch.nn.functional.one_hot(class_batch, num_classes=class_size).float()

    return {'x': x, 'dx': dx, 'class_labels': class_labels}

def create_dummy_training_config(data_dim=1000, latent_dim=6, num_classes=10):
    """Create a dummy training configuration for testing"""
    from .autoencoder import dummy_autoencoder
    
    # Create SINDy autoencoder config
    sindy_ae_config = dummy_autoencoder(input_dim=data_dim, latent_dim=latent_dim, num_classes=num_classes)
    
    # Create loss weights
    loss_weights = LossWeights(
        recon_wt=1.0,
        sindy_wt_x=1e-4,
        sindy_wt_z=1e-4,
        class_wt=1.0,
        l1_reg=1e-5
    )
    
    # Create training settings
    train_settings = TrainSettings(
        optimizer='adam',
        lr=1e-3,
        refinement_epochs=5,
        num_epochs=10,
        batch_size=20,
        threshold_frequency=5,
        coefficient_threshold=0.1,
        sequential_thresholding=True
    )
    
    # Create full training config
    training_config = TrainingConfig(
        sindy_ae_config=sindy_ae_config,
        loss_weights=loss_weights,
        train_settings=train_settings,
        print_progress=True,
        print_frequency=2,
        plot_loss=False,
        load_model_path=None,
        save_model_path=None
    )
    
    return training_config


def test_training():
    """Test the training pipeline with dummy data"""
    print("Testing training.py...")
    
    # Configuration
    data_dim = 100
    class_size = 5
    batch_size = 10
    latent_dim = 6
    
    # Create training config
    training_config = create_dummy_training_config(data_dim, latent_dim, class_size)
    
    # Create model
    sindy_ae = SINDyAE(training_config.sindy_ae_config)
    print(" Model created successfully")
    
    # Generate dummy input
    inp = generate_dummy_data(data_dim, class_size, batch_size)
    print(" Dummy data generated")
    
    # Test forward pass
    with torch.no_grad():
        model_out = sindy_ae(inp['x'])
        print(" Forward pass successful")
        print(f"  Output keys: {list(model_out.keys())}")
    
    # Test loss computation
    loss_weights = training_config.loss_weights
    
    # Test apply_sindy_ae_loss
    loss_train = apply_sindy_ae_loss(sindy_ae, loss_weights, model_out, inp)
    print(f" Training loss computed: {loss_train.item():.4f}")
    
    # Test compute_refinement_loss
    loss_refine = compute_refinement_loss(sindy_ae, loss_weights, model_out, inp)
    print(f" Refinement loss computed: {loss_refine.item():.4f}")
    
    # Test compute_loss_components
    loss_components = compute_loss_components(sindy_ae, model_out, inp, loss_weights)
    print(" Loss components computed:")
    for key, val in loss_components.items():
        print(f"  {key}: {val:.6f}")
    
    # Test backward pass
    optimizer = torch.optim.Adam(sindy_ae.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print(" Backward pass and optimizer step successful")
    
    print("All training tests passed!")


if __name__ == '__main__':
    test_training()