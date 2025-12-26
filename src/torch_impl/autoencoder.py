from torch.autograd.functional import jvp
import torch
from torch import nn
from enum import Enum
import math
from dataclasses import dataclass
from jaxtyping import Float, Array
from einops import einsum

class Activation(Enum):
    RELU = 'relu'
    ELU = 'elu'
    TANH = 'tanh'
    IDENTITY = 'identity'
    
    def get_module(self):
        if self == Activation.RELU:
            return nn.ReLU()
        elif self == Activation.ELU:
            return nn.ELU()
        elif self == Activation.TANH:
            return nn.Tanh()
        elif self == Activation.IDENTITY:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {self}")

@dataclass
class MLPConfig:
    weights:list[int]
    activation:Activation
    out_dim:int
    input_dim:int

@dataclass
class SINDyAEConfig:
    encoder_config:MLPConfig
    decoder_config:MLPConfig
    class_config:MLPConfig
    sindy_config:'SINDyConfig'

#IDEA: We can try to specifically limit some latent variables and force a functional term on them
@dataclass
class SINDyConfig:
    latent_dim:int
    model_order:int
    poly_order:int
    include_sine:bool|None = False
    include_tan:bool|None = False
    include_log:bool|None = False
    include_exp:bool|None = False
    include_reciprocal_func:bool|None = False
    
    def __post_init__(self):
        assert self.model_order == 1, "only first order ODE supported as of now"

class MLP(nn.Module):
    def __init__(self, ae_config: MLPConfig):
        super().__init__()
        self.layers=[]
        weights = [ae_config.input_dim] + ae_config.weights

        #Hidden layers
        for i in range(len(weights) - 1):
            self.layers.append(nn.Linear(weights[i], weights[i+1]))
            self.layers.append(ae_config.activation.get_module())
        
        #Out layer
        self.layers.append(nn.Linear(weights[-1], ae_config.out_dim))

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
        
        # Constant term
        count += 1
        
        # Linear terms
        count += config.latent_dim
        
        # Nonlinear function transformations
        included_function_count = sum([
            config.include_sine is not False and config.include_sine is not None,
            config.include_tan is not False and config.include_tan is not None,
            config.include_log is not False and config.include_log is not None,
            config.include_exp is not False and config.include_exp is not None,
            config.include_reciprocal_func is not False and config.include_reciprocal_func is not None
        ])
        count += included_function_count * config.latent_dim
        
        for order in range(2, config.poly_order + 1):
            # Number of combinations with replacement
            count += math.comb(config.latent_dim + order - 1, order)
        
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
        super().__init__()
        self.encoder=MLP(sindy_ae_config.encoder_config)
        self.decoder=MLP(sindy_ae_config.decoder_config)
        self.classification_head=MLP(sindy_ae_config.class_config)
        self.sindy=SINDy(sindy_ae_config.sindy_config)

    def forward(self, x:Float[Array, 'B D']):
        # Create tangent vectors for jvp (ones with same shape as input)
        v_x = torch.ones_like(x)
        
        z, enc_grads = jvp(self.encoder, (x,), (v_x,))
        
        # Create tangent vector for decoder jvp
        v_z = torch.ones_like(z)
        
        x_recon, dec_grads = jvp(self.decoder, (z,), (v_z,))

        class_score = self.classification_head(z) # B c
        
        feature_matrix = self.sindy(z) # B F
        sindy_predict = einsum(feature_matrix, self.sindy.coefficients, 'B F, F d -> B d')
        
        out_dict = {
            'z': z,
            'enc_grads': enc_grads, 
            'dec_grads': dec_grads, 
            'x_recon': x_recon,
            'class_score': class_score, 
            'sindy_predict': sindy_predict,
            'feature_matrix': feature_matrix
        }
        return out_dict


def dummy_model_training():
    batch_size=10
    input_dim=1024
    latent_dim=8
    num_classes=6

    ae_weights=[512,128,32]
    classifier_weights=[8,8]
    activation=Activation.RELU

    poly_order=1
    include_sin=True
    include_log=False
    include_exp=True
    include_tan=False
    include_reciprocal=False


    encoder_config=MLPConfig(weights=ae_weights, activation=activation, out_dim=latent_dim, input_dim=input_dim)
    decoder_config=MLPConfig(weights=ae_weights[::-1], activation=activation, out_dim=input_dim, input_dim=latent_dim)
    class_config=MLPConfig(weights=classifier_weights, activation=activation, out_dim=num_classes, input_dim=latent_dim)
    sindy_config= SINDyConfig(latent_dim=latent_dim, model_order=1, poly_order=poly_order, include_sine=include_sin, include_log=include_log, include_exp=include_exp, include_tan=include_tan, include_reciprocal_func=include_reciprocal)
    
    sindy_ae_config= SINDyAEConfig(encoder_config=encoder_config, decoder_config=decoder_config, class_config=class_config, sindy_config=sindy_config)

    sindy_ae=SINDyAE(sindy_ae_config)

    x=torch.ones((batch_size, input_dim))
    with torch.no_grad():
        out=sindy_ae(x)
        for key, val in out.items():
            print(f"{key}: {val.shape}")

    
if __name__=='__main__':
    dummy_model_training()