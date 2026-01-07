"""
Shared utilities for inference notebooks across TensorFlow and PyTorch implementations.
"""

import pickle
import os
import numpy as np
import networkx as nx
from glob import glob
from natsort import natsorted


def load_data(dataset_name):
    """
    Load preprocessed time series data for a given dataset.
    
    Args:
        dataset_name: Name of dataset ('Pancreas', 'Bone Marrows', 'SERGIO')
    
    Returns:
        dict: Data dictionary with 'x', 'dx', and other arrays
    """
    # Data is now in Examples/{dataset}/data/ directory
    data_path = os.path.join('..', 'Examples', dataset_name, 'data', 'time_series.pkl')
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def load_model_components(components_dir):
    """
    Load saved model components from chosen experiment.
    
    Args:
        components_dir: Path to chosen_exp_components directory
    
    Returns:
        dict: Dictionary with weights, biases, z_values, etc.
    """
    components = {}
    
    # Load decoder weights
    with open(os.path.join(components_dir, 'decoder_weights_list.pkl'), 'rb') as f:
        components['decoder_weights'] = pickle.load(f)
    
    with open(os.path.join(components_dir, 'decoder_biases_list.pkl'), 'rb') as f:
        components['decoder_biases'] = pickle.load(f)
    
    # Load classifier weights
    with open(os.path.join(components_dir, 'classifier_weights_list.pkl'), 'rb') as f:
        components['classifier_weights'] = pickle.load(f)
    
    with open(os.path.join(components_dir, 'classifier_biases_list.pkl'), 'rb') as f:
        components['classifier_biases'] = pickle.load(f)
    
    # Load z values
    with open(os.path.join(components_dir, 'z_values.pkl'), 'rb') as f:
        components['z_values'] = pickle.load(f)
    
    # Load class indices
    with open(os.path.join(components_dir, 'index_to_class.pkl'), 'rb') as f:
        components['index_to_class'] = pickle.load(f)
    
    # Load active terms if available
    active_terms_path = os.path.join(components_dir, 'active_terms.pkl')
    if os.path.exists(active_terms_path):
        with open(active_terms_path, 'rb') as f:
            components['active_terms'] = pickle.load(f)
    
    return components


def calculate_errors(predictions, targets):
    """
    Calculate relative reconstruction errors.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
    
    Returns:
        float: Relative error (MSE / mean squared target)
    """
    mse = np.mean((predictions - targets) ** 2)
    mean_sq_target = np.mean(targets ** 2)
    return mse / mean_sq_target if mean_sq_target > 0 else float('inf')


def print_equations(active_terms, coefficients=None, library_names=None):
    """
    Print SINDy equations from active terms.
    
    Args:
        active_terms: Dict mapping LHS index to set of RHS term indices
        coefficients: Optional coefficient values
        library_names: Optional names for library terms
    """
    for lhs, rhs_terms in active_terms.items():
        eq_parts = [f"dz{lhs} = "]
        for term_idx in sorted(rhs_terms):
            if coefficients is not None:
                coeff = coefficients[lhs, term_idx]
                sign = "+" if coeff >= 0 else ""
                eq_parts.append(f"{sign}{coeff:.5f}*")
            if library_names is not None:
                eq_parts.append(library_names[term_idx])
            else:
                eq_parts.append(f"term_{term_idx}")
            eq_parts.append(" ")
        print("".join(eq_parts))


def save_components(save_dir, **kwargs):
    """
    Save model components to directory.
    
    Args:
        save_dir: Directory to save to
        **kwargs: Named components to save (e.g., z_values=z_array)
    """
    os.makedirs(save_dir, exist_ok=True)
    for name, data in kwargs.items():
        filepath = os.path.join(save_dir, f'{name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {name} to {filepath}")


def find_experiments(experiments_dir, pattern='*/model.pt'):
    """
    Find all experiment checkpoint files.
    
    Args:
        experiments_dir: Path to experiments directory
        pattern: Glob pattern for checkpoint files (default: '*/model.pt' for PyTorch)
    
    Returns:
        list: Sorted list of checkpoint file paths
    """
    search_path = os.path.join(experiments_dir, pattern)
    return natsorted(glob(search_path))


def get_experiment_info(checkpoint_path):
    """
    Extract experiment information from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        dict: Dictionary with 'name', 'path', and 'dir'
    """
    exp_dir = os.path.dirname(checkpoint_path)
    exp_name = os.path.basename(exp_dir)
    return {
        'name': exp_name,
        'path': checkpoint_path,
        'dir': exp_dir
    }


def compute_classification_metrics(predictions, targets):
    """
    Compute classification accuracy and confusion data.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
    
    Returns:
        dict: Metrics including accuracy, misclassified count, etc.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    accuracy = accuracy_score(targets, predictions)
    cm = confusion_matrix(targets, predictions)
    misclassified = np.sum(predictions != targets)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'misclassified': misclassified
    }


def build_network_from_sindy(active_terms, coefficients=None):
    """
    Build a directed graph from SINDy active terms.
    
    Args:
        active_terms: Dict mapping LHS index to set of RHS term indices
        coefficients: Optional coefficient matrix for edge weights
    
    Returns:
        nx.DiGraph: Network graph
    """
    G = nx.DiGraph()
    
    # Add nodes for all latent variables involved
    all_vars = set(active_terms.keys())
    for rhs_set in active_terms.values():
        all_vars.update(rhs_set)
    
    for var in all_vars:
        G.add_node(f'z{var}', node_type='latent')
    
    # Add edges from active terms
    for lhs, rhs_terms in active_terms.items():
        for term_idx in rhs_terms:
            weight = 1.0
            if coefficients is not None:
                weight = abs(coefficients[lhs, term_idx])
            G.add_edge(f'z{term_idx}', f'z{lhs}', weight=weight)
    
    return G


def save_network(G, save_path, format='graphml'):
    """
    Save network graph to file.
    
    Args:
        G: NetworkX graph
        save_path: Path to save file
        format: Format ('graphml', 'gexf', 'pickle')
    """
    if format == 'graphml':
        nx.write_graphml(G, save_path)
    elif format == 'gexf':
        nx.write_gexf(G, save_path)
    elif format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(G, f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Network saved to {save_path}")


def analyze_network_metrics(G):
    """
    Compute and print network analysis metrics.
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Dictionary of computed metrics
    """
    metrics = {}
    
    if G.number_of_edges() > 0:
        metrics['density'] = nx.density(G)
        metrics['in_degree'] = dict(G.in_degree())
        metrics['out_degree'] = dict(G.out_degree())
        
        # Try to compute centrality if graph is not too large
        if G.number_of_nodes() < 1000:
            metrics['betweenness'] = nx.betweenness_centrality(G)
        
        print(f"Network Density: {metrics['density']:.3f}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    else:
        print("Empty network")
    
    return metrics

