import numpy as np
from typing import List, Optional, Any
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def plot_training_curves(
    training_array: np.ndarray,
    validation_array: Optional[np.ndarray],
    feature_names: List[str],
    title: str,
    save_path: str,
    scale: int = 1
):
    num_features = training_array.shape[1]
    num_epochs = training_array.shape[0]
    
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 8 * num_features))
    
    if num_features == 1:
        axes = [axes]
    
    for i in range(num_features):
        training_values = training_array[:, i]
        scaled_epochs = np.arange(0, num_epochs) * scale
        
        axes[i].plot(scaled_epochs, training_values, label="Training")
        
        if validation_array is not None:
            validation_values = validation_array[:, i]
            axes[i].plot(scaled_epochs, validation_values, "--", label="Validation")
        
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel("Error")
        axes[i].legend()
    
    fig.suptitle(title)
    fig.savefig(save_path)
    plt.tight_layout()
    plt.close(fig)

        

def plot_curves(
    start: int,
    title: str,
    feature_names: List[str],
    params: Any,
    axis_name: str,
    training_array: np.ndarray,
    validation_exists: bool = False,
    validation_array: Optional[np.ndarray] = None,
    scale: int = 1,
) -> None:
    """
    Plot training and validation curves for various features.
    """
    num_features = training_array.shape[1]  # Number of features
    num_epochs = training_array.shape[0]  # Number of samples

    # Prepare subplots
    fig, axes = plt.subplots(
        nrows=num_features, ncols=1, figsize=(10, 8 * num_features)
    )
    
    # If there's only one feature, axes is not a list/array but a single Axes object
    if num_features == 1:
        axes = [axes]

    # Loop through each feature
    for i in range(num_features):
        training_feature_values = training_array[:, i]

        # Multiply epochs by scale to scale the x-axis values
        scaled_epochs = np.arange(0, num_epochs) * scale

        # Plot the training curve
        axes[i].plot(scaled_epochs, training_feature_values, label="Training")

        # Plot the validation curve as dashed
        if validation_exists and validation_array is not None:
            validation_feature_values = validation_array[:, i]
            axes[i].plot(
                scaled_epochs, validation_feature_values, "--", label="Validation"
            )

        # Set plot title
        axes[i].set_title(feature_names[i])

        # Set plot labels
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(axis_name)

        # Add legend
        axes[i].legend()

    # Set the main plot title
    fig.suptitle(title)

    # Save the plot as an image
    fig.savefig(title + ".png")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Note: plt.show() blocks execution, commented out for non-interactive
    # plt.show()
    # time.sleep(1)


def print_progress(
    sess: tf.Session,
    i: int,
    loss: tf.Tensor,
    losses: Dict[str, tf.Tensor],
    train_dict: Dict[str, Any],
    validation_dict: Dict[str, Any],
    x_norm: float,
    sindy_predict_norm: float,
    z_norm: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Print loss function values to keep track of the training progress.
    """
    training_loss_vals = sess.run(
        (loss,) + tuple(losses.values()), feed_dict=train_dict
    )
    validation_loss_vals = sess.run(
        (loss,) + tuple(losses.values()), feed_dict=validation_dict
    )

    print("Epoch %d" % i)
    print("Training loss {0}, {1}".format(training_loss_vals[0], training_loss_vals[1:]))
    print(
        "Validation loss {0}, {1}".format(
            validation_loss_vals[0], validation_loss_vals[1:]
        )
    )
    decoder_losses = sess.run(
        (losses["decoder"], losses["sindy_x"], losses["sindy_z"]),
        feed_dict=validation_dict,
    )

    loss_ratios = (
        decoder_losses[0] / x_norm,
        decoder_losses[1] / sindy_predict_norm,
        decoder_losses[2] / z_norm,
    )
    print(
        "decoder loss ratio: %f, decoder SINDy loss ratio: %f, SINDy z loss ratio: %f"
        % loss_ratios
    )

    return training_loss_vals, validation_loss_vals, loss_ratios

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