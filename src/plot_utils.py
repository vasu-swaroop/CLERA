import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

        
def extract_weights_biases(test_set_results, num_layers):
    """
    Extract weights and biases dictionaries from the test_set_results.
    
    Arguments:
        test_set_results (dict): Dictionary containing the test set results.
        num_layers (int): Number of network layers.
        
    Returns:
        encoder_weights (dict): Dictionary containing encoder weights.
        encoder_biases (dict): Dictionary containing encoder biases.
        decoder_weights (dict): Dictionary containing decoder weights.
        decoder_biases (dict): Dictionary containing decoder biases.
    """
    encoder_weights = {}
    encoder_biases = {}
    decoder_weights = {}
    decoder_biases = {}
    
    # Loop through each layer to extract weights and biases
    for layer in range(num_layers + 1):
        encoder_weights[layer] = test_set_results[f'encoder_weights'][layer]
        encoder_biases[layer] = test_set_results[f'encoder_biases'][layer]
        decoder_weights[layer] = test_set_results[f'decoder_weights'][layer]
        decoder_biases[layer] = test_set_results[f'decoder_biases'][layer]
    
    encoder_weights_list = [encoder_weights[layer] for layer in range(num_layers + 1)]
    encoder_biases_list = [encoder_biases[layer] for layer in range(num_layers + 1)]
    decoder_weights_list = [decoder_weights[layer] for layer in range(num_layers + 1)]
    decoder_biases_list = [decoder_biases[layer] for layer in range(num_layers + 1)]
    
    return encoder_weights_list, encoder_biases_list, decoder_weights_list, decoder_biases_list

# Define numpy functions for activation functions
def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def select_activation_function(activation):
    """
    Select the appropriate activation function based on the given activation string.
    Arguments:
        activation - String, activation function name ('relu', 'elu', 'sigmoid', etc.)
    Returns:
        activation_function - Numpy function, the selected activation function
    """
    if activation == 'relu':
        activation_function = relu
    elif activation == 'elu':
        activation_function = elu
    elif activation == 'sigmoid':
        activation_function = sigmoid
    else:
        activation_function = None
    
    return activation_function

def calculate_reconstruction_difference(test_set_results, latent_var, decoder_weights_list, decoder_biases_list, activation_function=None):
    # Get the number of time points and gene expressions
    time = len(test_set_results['x'])
    gene_expression = len(test_set_results['x'][0])
    
    # Get the number of layers in the decoder
    num_layers = len(decoder_weights_list)
    
    # Initialize the reconstruction difference array
    reconstruction_difference = np.zeros((latent_var, time, gene_expression))
    
    # Loop through each latent variable to zero out
    for latent_index_to_zero in range(latent_var):
        # Loop through each time index
        for index in range(time):
            # Get the original and modified input sample
            input_sample = test_set_results['x_decode'][index]
            original_z = test_set_results['z'][index]
            modified_z = original_z.copy()
            modified_z[latent_index_to_zero] = 0
            
            # Initialize modified_x_decode using the first layer weights and biases
            modified_x_decode = np.dot(modified_z, decoder_weights_list[0]) + decoder_biases_list[0]
            if activation_function:
                modified_x_decode = activation_function(modified_x_decode)
            
            # Loop through the remaining layers
            for layer in range(1, num_layers):
                modified_x_decode = np.dot(modified_x_decode, decoder_weights_list[layer]) + decoder_biases_list[layer]
                if activation_function and layer < num_layers - 1:  # Apply activation except for the last layer
                    modified_x_decode = activation_function(modified_x_decode)

            # Calculate the absolute difference between the original and modified input samples
            abs_diff = np.abs((input_sample - modified_x_decode))
            
            # Store the absolute difference in the reconstruction_difference array
            reconstruction_difference[latent_index_to_zero][index][:] = abs_diff 
            
    return reconstruction_difference


def find_max_indices_with_attributes(array, attributes):
    # Find the indices of the maximum value in each column
    max_indices = np.argmax(array, axis=0)

    # Attach the attribute (row index) to each column
    column_attributes = attributes[max_indices]
    return column_attributes

def group_max_indices_by_row(array, attributes, attribute_names):
    max_indices = np.argmax(array, axis=0)
    # Create a dictionary to store the results
    max_mean_dict = {}

    # Loop through column indices and corresponding row indices
    for col_idx, row_idx in enumerate(max_indices):
        if row_idx not in max_mean_dict:
            max_mean_dict[row_idx] = []
        
        # Append column name and its corresponding value
        col_name = attribute_names[col_idx]
        col_value = array[row_idx, col_idx]
        max_mean_dict[row_idx].append((col_name, col_value))

    return max_mean_dict

def get_dict_clusters(array, name_genes):
    attributes = np.arange(array.shape[0])  # Replace with your actual attributes

    # Find max indices with attributes
    column_attributes = find_max_indices_with_attributes(array, attributes)

    # Group max indices by row and attribute
    max_dict = group_max_indices_by_row(array, column_attributes, name_genes)
    return max_dict

def visualize_tripartite_graph(matrix, name_genes, find_num, active_terms):
    """
    Visualize a tripartite graph based on the given data.

    Parameters:
    mean_diff (numpy.ndarray): The metric used to convert time series to a single value.
    name_genes (list): List of column names for genes.
    find_num (int): Number of top genes to display for each latent variable.
    active_terms (list): List of active terms for each latent variable.

    Returns:
    displays the plot and returns the clusters in form of a dictionaty
    """
    top_columns = {}
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index, :]
        sorted_indices = np.argsort(row)[::-1][:find_num]
        top_columns[row_index] = [(name_genes[col_index], row[col_index]) for col_index in sorted_indices]

    G = nx.Graph()
    # Add nodes and specify their bipartite attribute
    for row_index in range(matrix.shape[0]):
        G.add_node(f'Latent_V_{row_index}', bipartite=0)  # Central nodes (rows)
        for col_name, col_value in top_columns[row_index]:
            G.add_node(col_name, bipartite=1)  # Surrounding nodes (columns)
            G.add_edge(f'Latent_V_{row_index}', col_name, value=col_value)

    # Add the third set of nodes and edges
    for row_index in range(matrix.shape[0]):
        for act_term in active_terms[row_index]:
            G.add_node(f'Diff_node_{act_term}', bipartite=2)  # Diff nodes
            G.add_edge(f'Latent_V_{row_index}', f'Diff_node_{act_term}')

    # Separate nodes into sets for bipartite layout
    central_nodes = {node for node in G.nodes if G.nodes[node]['bipartite'] == 0}
    surrounding_nodes = {node for node in G.nodes if G.nodes[node]['bipartite'] == 1}
    diff_nodes = {node for node in G.nodes if G.nodes[node]['bipartite'] == 2}
    pos={}
    # Layout using bipartite_layout
    # Position central nodes
    central_x_positions = np.linspace(1, 2, len(central_nodes))
    pos.update({node: (central_x_positions[i], 1) for i, node in enumerate(central_nodes)})

    # Position surrounding nodes
    surrounding_x_positions = np.linspace(0.2, 3.8, len(surrounding_nodes))
    pos.update({node: (surrounding_x_positions[i], 2) for i, node in enumerate(surrounding_nodes)})

    # Position diff nodes
    diff_x_positions = np.linspace(0.2, 3.8, len(diff_nodes))
    pos.update({node: (diff_x_positions[i], 0) for i, node in enumerate(diff_nodes)})

    # Draw the graph. Change the values based on the size required
    plt.figure(figsize=(20, 5))

    # Draw central nodes (rows) in blue
    nx.draw_networkx_nodes(G, pos, nodelist=central_nodes, node_color='blue', node_size=600, alpha=0.8)

    # Draw surrounding nodes (columns) in green
    nx.draw_networkx_nodes(G, pos, nodelist=surrounding_nodes, node_color='green', node_size=600, alpha=0.8)

    # Draw diff nodes in red
    nx.draw_networkx_nodes(G, pos, nodelist=diff_nodes, node_color='red', node_size=600, alpha=0.8)

    # Draw edges with reduced opacity
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

    # Draw labels for nodes
    labels = {node: node for node in G.nodes}  
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Latent Variables/LHS term of the ODEs'),
        Line2D([0], [0], color='green', lw=4, label='Genes most related to the Latent Variables'),
        Line2D([0], [0], color='red', lw=4, label='Latent Variable present in RHS of ODE equations')
    ]

    # Draw the legend
    plt.legend(handles=legend_elements, loc='center right', title="Node Types")

    plt.title("Tripartite Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return top_columns

def plot_top_nodes(cluster_index, top_nodes):
    genes, values = zip(*top_nodes)
    plt.figure(figsize=(20, 5))
    plt.bar(range(len(genes)), values, align='center', alpha=0.5)
    plt.xticks(range(len(genes)), genes)
    plt.ylabel('Values')
    plt.title(f'Top 40 nodes for Cluster {cluster_index}')
    plt.show()