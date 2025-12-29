from typing import Dict, Optional, List, Any
import numpy as np
from .plot_utils import plot_curves, plot_training_curves


class LossTracker:
    def __init__(self):
        self.training_losses = []
        self.validation_losses = []
        self.ratios = []
        self.ref_training_losses = []
        self.ref_validation_losses = []
        self.ref_ratios = []
        self.sindy_model_terms = []
        
        self.current_losses = {}
    
    def update_losses(self, losses_dict: Dict[str, float], phase='train'):
        self.current_losses = losses_dict
        if phase == 'train':
            loss_array = np.array([
                losses_dict['total'],
                losses_dict['recon'],
                losses_dict['sindy_z'],
                losses_dict['sindy_x'],
                losses_dict['sindy_reg'],
                losses_dict['ae_reg'],
                losses_dict['class']
            ])
            self.training_losses.append(loss_array)
        elif phase == 'val':
            loss_array = np.array([
                losses_dict['total'],
                losses_dict['recon'],
                losses_dict['sindy_z'],
                losses_dict['sindy_x'],
                losses_dict['sindy_reg'],
                losses_dict['ae_reg'],
                losses_dict['class']
            ])
            self.validation_losses.append(loss_array)
    
    def update_refinement_losses(self, losses_dict: Dict[str, float], phase='train'):
        if phase == 'train':
            loss_array = np.array([
                losses_dict['total'],
                losses_dict['recon'],
                losses_dict['sindy_z'],
                losses_dict['sindy_x'],
                losses_dict.get('sindy_reg', 0.0),
                losses_dict['ae_reg'],
                losses_dict['class']
            ])
            self.ref_training_losses.append(loss_array)
        elif phase == 'val':
            loss_array = np.array([
                losses_dict['total'],
                losses_dict['recon'],
                losses_dict['sindy_z'],
                losses_dict['sindy_x'],
                losses_dict.get('sindy_reg', 0.0),
                losses_dict['ae_reg'],
                losses_dict['class']
            ])
            self.ref_validation_losses.append(loss_array)
    
    def print_losses(self, epoch: int, phase='train'):
        print(f"Epoch {epoch}")
        loss_str = ", ".join([f"{k}: {v:.6f}" for k, v in self.current_losses.items()])
        print(f"{phase.capitalize()} - {loss_str}")
    
    def plot_all_losses(self, save_name: str, print_frequency: int):
        loss_feature_names = [
            "Combined Loss",
            "Reconstruction Loss",
            "SINDy_z Loss",
            "SINDy_x Loss",
            "SINDy Regularization",
            "Autoencoder Regularization",
            "Classification Loss"
        ]
        
        # Check if we have at least one type of loss to plot
        if len(self.training_losses) > 0 or len(self.validation_losses) > 0:
            training_array = np.array(self.training_losses) if len(self.training_losses) > 0 else None
            validation_array = np.array(self.validation_losses) if len(self.validation_losses) > 0 else None
            
            # plot_training_curves expects training_array as the first argument (main curve)
            # if training_array is None, we swap them but the behavior might be unexpected
            # Usually we expect both or at least training.
            primary_array = training_array if training_array is not None else validation_array
            secondary_array = validation_array if training_array is not None else None
            
            plot_training_curves(
                training_array=primary_array,
                validation_array=secondary_array,
                feature_names=loss_feature_names,
                title=f"{save_name} - Training Phase",
                save_path=f"{save_name}_training_errors.png",
                scale=print_frequency
            )
            print(f"  Saved: {save_name}_training_errors.png")
        
        if len(self.ref_training_losses) > 0 or len(self.ref_validation_losses) > 0:
            ref_training_array = np.array(self.ref_training_losses) if len(self.ref_training_losses) > 0 else None
            ref_validation_array = np.array(self.ref_validation_losses) if len(self.ref_validation_losses) > 0 else None
            
            primary_ref = ref_training_array if ref_training_array is not None else ref_validation_array
            secondary_ref = ref_validation_array if ref_training_array is not None else None

            plot_training_curves(
                training_array=primary_ref,
                validation_array=secondary_ref,
                feature_names=loss_feature_names,
                title=f"{save_name} - Refinement Phase",
                save_path=f"{save_name}_refinement_errors.png",
                scale=print_frequency
            )
            print(f"  Saved: {save_name}_refinement_errors.png")
