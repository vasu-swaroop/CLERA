from torch.autograd.functional import jvp

from torch import nn
import torch
from Enum import enum
import math
from dataclasses import dataclass
from jaxtyping import Float, Array
from einops import einsum

class Activation(enum, nn.Module):
    RELU= nn.ReLU()
    ELU= nn.ELU()
    TANH= nn.tanh()
    IDENTITY= nn.identity

    def __call__(self):
        return self.value()

@dataclass
class MLPConfig:
    weights:list[int]
    activation:Activation
    out_dim:int
    input_dim:int

@dataclass
class SINDyAEConfig:
    encoder_config:MLPConfig
    decoder_congig:MLPConfig
    class_config:MLPConfig
    sindy_config:SINDyAEConfig

#IDEA: We can try to specifically limit some latent variables and force a functional term on them
@dataclass
class SINDyConfig:
    latent_dim:int
    model_order:int
    poly_order:int
    include_sine:callable|None
    include_tan:callable|None
    include_log:callable|None
    include_exp:callable|None
    include_reciprocal_func:callable|None

class MLP(nn.Module):
    def __init__(self, ae_config: MLPConfig):
        super().__init__()
        self.layers=[]
        ae_config.weights=[ae_config.input_dim]+ae_config.weights

        #Hidden layers
        for weights in ae_config.weights[:-1]:
            self.layers.extend([nn.Linear(), ae_config.activation])
        
        #Out layer
        self.layers.extend([nn.Linear(ae_config.weights[-1], ae_config.out_dim)])

        self.layers=nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x

# CAREFUL: The order of feature_matrix is fixed once and should be used during inference
class SINDy(nn.Module):
    def __init__(self, sindy_config:SINDyConfig):
        super().__init__()
        self.sindy_config=sindy_config
        self.library_size=self.count_library_size()

        #IDEA: Can learn a LoRA
        self.coefficients=nn.Parameter(torch.ones(self.library_size, self.sindy_config.latent_dim))
        self.coefficient_mask=nn.Parameter(torch.ones(self.library_size, self.sindy_config.latent_dim))
    
    def count_library_size(self):
        config = self.sindy_config
        count = 0
        for order in range(config.poly_order + 1):
            count += math.comb(config.latent_dim, order)
        
        # Count included nonlinear functions
        included_function_count = sum([
            config.include_sine is not None,
            config.include_tan is not None,
            config.include_log is not None,
            config.include_exp is not None,
            config.include_reciprocal_func is not None
        ])
        count += included_function_count * config.latent_dim
        return count
    
    def _add_nonlinear_terms(self, z: torch.Tensor, features: list) -> None:
        """Add nonlinear function transformations to feature list."""
        config = self.sindy_config
        latent_dim = config.latent_dim
        
        # Map of function transforms
        transforms = {
            'reciprocal': lambda zi: 1.0 / (1.0 + zi ** 2) if config.include_reciprocal_func else None,
            'tan': lambda zi: torch.tan(zi) if config.include_tan else None,
            'log': lambda zi: torch.log(zi) if config.include_log else None,
            'exp': lambda zi: torch.exp(zi) if config.include_exp else None,
            'sine': lambda zi: torch.sin(zi) if config.include_sine else None,
        }
        
        for transform_fn in transforms.values():
            if transform_fn is not None:
                for i in range(latent_dim):
                    result = transform_fn(z[:, i])
                    if result is not None:
                        features.append(result)
    
    def _add_polynomial_terms(self, z: torch.Tensor, features: list, order: int) -> None:
        """Add polynomial interaction terms of a specific order."""
        latent_dim = self.sindy_config.latent_dim
        
        def generate_indices(order: int):
            """Generate all combinations of indices for polynomial terms."""
            if order == 2:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        yield (i, j)
            elif order == 3:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            yield (i, j, k)
            elif order == 4:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            for p in range(k, latent_dim):
                                yield (i, j, k, p)
            elif order == 5:
                for i in range(latent_dim):
                    for j in range(i, latent_dim):
                        for k in range(j, latent_dim):
                            for p in range(k, latent_dim):
                                for q in range(p, latent_dim):
                                    yield (i, j, k, p, q)
        
        for indices in generate_indices(order):
            # Compute product of z[:, i] for all i in indices
            term = z[:, indices[0]]
            for idx in indices[1:]:
                term = term * z[:, idx]
            features.append(term)
    
    def get_feature_matrix(self, z: Float[Array, 'B d']) -> Float[Array, 'B F']:
        """
        Construct the SINDy feature library matrix.
        
        Args:
            z: Latent representation tensor of shape [batch_size, latent_dim]
            
        Returns:
            Feature matrix of shape [batch_size, library_size]
        """
        config = self.sindy_config
        batch_size = z.shape[0]
        device = z.device
        dtype = z.dtype
        
        features = []
        
        # Constant term
        features.append(torch.ones(batch_size, device=device, dtype=dtype))
        
        # Linear terms (order 1)
        for i in range(config.latent_dim):
            features.append(z[:, i])
        
        # Nonlinear function transformations
        self._add_nonlinear_terms(z, features)
        
        # Polynomial interaction terms (order 2 to poly_order)
        for order in range(2, config.poly_order + 1):
            self._add_polynomial_terms(z, features, order)
        
        # Stack all features along the feature dimension
        return torch.stack(features, dim=1)

    def forward(self, z):
        return self.get_feature_matrix(z)

class SINDyAE(nn.Module):
    def __init__(self, sindy_ae_config: SINDyAEConfig):
        self.encoder=MLP(sindy_ae_config.encoder_config)
        self.decoder=MLP(sindy_ae_config.decoder_congig)
        self.classification_head=MLP(sindy_ae_config.class_config)
        self.sindy=SINDy(sindy_ae_config.sindy_config)

    def forward(self, x:Float[Array, 'B D']):
        z=self.encoder(x) # B d

        z, enc_grads= jvp(self.encoder, (x,))
        x, dec_grads= jvp(self.decoder, (z,))

        class_score=self.classification_head(z) # B c
        feature_matrix=self.sindy(z) # B F
        sindy_predict=einsum(feature_matrix, self.sindy.coefficients, 'B F, F d -> B d')
        out_dict={'z':z,'enc_grads':enc_grads, 'dec_grads':dec_grads, 'x':x, 'class_score':class_score, 'sindy_predict':sindy_predict,'feature_matrix':feature_matrix}
        return out_dict
