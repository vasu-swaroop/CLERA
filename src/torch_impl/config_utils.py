import os
import yaml
from src.torch_impl.autoencoder import SINDyConfig, MLPConfig, SINDyAEConfig, Activation, Initialization
from src.torch_impl.training import LossWeights, TrainSettings, TrainingConfig

def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_training_config(config, script_dir):
    """Convert raw configuration dict to TrainingConfig and setup directories."""
    
    # Model Configuration
    model_params = config['model_config']
    input_dim = model_params['input_dim']
    latent_dim = model_params['latent_dim']
    num_classes = model_params['num_classes']
    ae_widths = model_params['ae_widths']
    classifier_widths = model_params['classifier_widths']
    activation_str = model_params.get('activation', 'relu')
    activation = Activation[activation_str.upper()]
    initialization_str = model_params.get('initialization', 'xavier')
    initialization = Initialization[initialization_str.upper()]

    encoder_config = MLPConfig(
        weights=ae_widths,
        activation=activation,
        out_dim=latent_dim,
        input_dim=input_dim,
        initialization=initialization
    )

    decoder_config = MLPConfig(
        weights=ae_widths[::-1],
        activation=activation,
        out_dim=input_dim,
        input_dim=latent_dim,
        initialization=initialization
    )

    class_config = MLPConfig(
        weights=classifier_widths,
        activation=activation,
        out_dim=num_classes,
        input_dim=latent_dim,
        initialization=initialization
    )

    sindy_params = config['sindy_config']
    sindy_config = SINDyConfig(
        latent_dim=latent_dim,
        model_order=sindy_params.get('model_order', 1),
        poly_order=sindy_params.get('poly_order', 2),
        include_sine=sindy_params.get('include_sine', True),
        include_exp=sindy_params.get('include_exp', True),
        include_reciprocal_func=sindy_params.get('include_reciprocal_func', True)
    )

    sindy_ae_config = SINDyAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        class_config=class_config,
        sindy_config=sindy_config
    )

    # Loss Configuration
    loss_params = config['loss_weights']
    loss_weights = LossWeights(
        recon_wt=loss_params.get('recon_wt', 100.0),
        sindy_wt_z=loss_params.get('sindy_wt_z', 1e-2),
        sindy_wt_x=loss_params.get('sindy_wt_x', 1e-1),
        class_wt=loss_params.get('class_wt', 1.0),
        l1_reg=loss_params.get('l1_reg', 1e-2)
    )

    # Train Settings
    train_params = config['train_settings']
    train_settings = TrainSettings(
        optimizer=train_params.get('optimizer', 'adam'),
        lr=train_params.get('lr', 1e-3),
        num_epochs=train_params.get('num_epochs', 1000),
        refinement_epochs=train_params.get('refinement_epochs', 500),
        batch_size=train_params.get('batch_size', 1024),
        threshold_frequency=train_params.get('threshold_frequency', 10),
        coefficient_threshold=train_params.get('coefficient_threshold', 0.5),
        sequential_thresholding=train_params.get('sequential_thresholding', True),
        max_active_terms=train_params.get('max_active_terms', None)
    )

    # General Training Config & Experiment Organization
    gen_params = config['training_config']
    experiments_dir = gen_params.get('experiments_dir', 'experiments')
    experiment_name = gen_params.get('experiment_name', 'default_experiment')
    experiment_path = os.path.join(script_dir, experiments_dir, experiment_name)

    os.makedirs(experiment_path, exist_ok=True)
    
    save_path = os.path.join(experiment_path, gen_params.get('save_model_path', 'model.pt'))

    training_config = TrainingConfig(
        sindy_ae_config=sindy_ae_config,
        loss_weights=loss_weights,
        train_settings=train_settings,
        print_progress=gen_params.get('print_progress', True),
        print_frequency=gen_params.get('print_frequency', 50),
        plot_loss=gen_params.get('plot_loss', True),
        load_model_path=None,
        save_model_path=save_path
    )

    return training_config, experiment_path
