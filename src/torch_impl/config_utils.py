import os
import yaml
from src.torch_impl.autoencoder import SINDyConfig, MLPConfig, SINDyAEConfig, Activation, Initialization
from src.torch_impl.losses import LossWeights
from src.torch_impl.training import TrainSettings, TrainingConfig

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
        include_reciprocal_func=sindy_params.get('include_reciprocal_func', True),
        innitialization_type=sindy_params.get('innitialization_type', None),
        innitialization_set=tuple(sindy_params.get('innitialization_set', (1,)))
    )

    sindy_ae_config = SINDyAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        class_config=class_config,
        sindy_config=sindy_config
    )

    # Loss Configuration - explicitly convert to float since YAML may parse scientific notation as strings
    loss_params = config['loss_weights']
    loss_weights = LossWeights(
        recon_wt=float(loss_params.get('recon_wt', 100.0)),
        sindy_wt_z=float(loss_params.get('sindy_wt_z', 1e-2)),
        sindy_wt_x=float(loss_params.get('sindy_wt_x', 1e-1)),
        class_wt=float(loss_params.get('class_wt', 1.0)),
        l1_reg=float(loss_params.get('l1_reg', 1e-2)),
        sindy_reg_wt=float(loss_params.get('sindy_reg_wt', 8e-3))
    )

    # Train Settings - explicitly convert types since YAML may parse values unexpectedly
    train_params = config['train_settings']
    train_settings = TrainSettings(
        optimizer=str(train_params.get('optimizer', 'adam')),
        lr=float(train_params.get('lr', 1e-3)),
        num_epochs=int(train_params.get('num_epochs', 1000)),
        refinement_epochs=int(train_params.get('refinement_epochs', 500)),
        batch_size=int(train_params.get('batch_size', 1024)),
        threshold_frequency=int(train_params.get('threshold_frequency', 10)),
        coefficient_threshold=float(train_params.get('coefficient_threshold', 0.5)),
        sequential_thresholding=bool(train_params.get('sequential_thresholding', True)),
        max_active_terms=int(train_params['max_active_terms']) if train_params.get('max_active_terms') is not None else None
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

    # Debug: print all loaded config values
    print("\n" + "="*60)
    print("LOADED CONFIGURATION:")
    print("="*60)
    print(f"\n[Model Config]")
    print(f"  input_dim: {encoder_config.input_dim}")
    print(f"  latent_dim: {encoder_config.out_dim}")
    print(f"  num_classes: {class_config.out_dim}")
    print(f"  ae_widths: {encoder_config.weights}")
    print(f"  classifier_widths: {class_config.weights}")
    print(f"  activation: {activation}")
    print(f"  initialization: {initialization}")
    
    print(f"\n[SINDy Config]")
    print(f"  model_order: {sindy_config.model_order}")
    print(f"  poly_order: {sindy_config.poly_order}")
    print(f"  include_sine: {sindy_config.include_sine}")
    print(f"  include_exp: {sindy_config.include_exp}")
    print(f"  include_reciprocal_func: {sindy_config.include_reciprocal_func}")
    print(f"  innitialization_type: {sindy_config.innitialization_type}")
    print(f"  innitialization_set: {sindy_config.innitialization_set}")
    
    print(f"\n[Loss Weights]")
    print(f"  recon_wt: {loss_weights.recon_wt}")
    print(f"  sindy_wt_z: {loss_weights.sindy_wt_z}")
    print(f"  sindy_wt_x: {loss_weights.sindy_wt_x}")
    print(f"  class_wt: {loss_weights.class_wt}")
    print(f"  l1_reg: {loss_weights.l1_reg}")
    print(f"  sindy_reg_wt: {loss_weights.sindy_reg_wt}")
    
    print(f"\n[Train Settings]")
    print(f"  optimizer: {train_settings.optimizer}")
    print(f"  lr: {train_settings.lr}")
    print(f"  num_epochs: {train_settings.num_epochs}")
    print(f"  refinement_epochs: {train_settings.refinement_epochs}")
    print(f"  batch_size: {train_settings.batch_size}")
    print(f"  threshold_frequency: {train_settings.threshold_frequency}")
    print(f"  coefficient_threshold: {train_settings.coefficient_threshold}")
    print(f"  sequential_thresholding: {train_settings.sequential_thresholding}")
    print(f"  max_active_terms: {train_settings.max_active_terms}")
    
    print(f"\n[Training Config]")
    print(f"  print_progress: {training_config.print_progress}")
    print(f"  print_frequency: {training_config.print_frequency}")
    print(f"  plot_loss: {training_config.plot_loss}")
    print(f"  save_model_path: {training_config.save_model_path}")
    print(f"  experiment_path: {experiment_path}")
    print("="*60 + "\n")

    return training_config, experiment_path
