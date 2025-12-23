import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import community as community_louvain
import os
import pandas as pd
from tqdm import tqdm
from gprofiler import GProfiler
def get_cell_type_data(data_type):
    """Returns cell type information for the specified data type."""
    if data_type == 'Pancreas':
        cell_types = 7
        index_to_clusters = {
            0: 'Alpha', 1: 'Beta', 2: 'Ductal', 3: 'Epsilon',
            4: 'Ngn3 high EP', 5: 'Ngn3 low EP', 6: 'Pre-endocrine'
        }
        
        index_to_all_types = {
            **index_to_clusters, 7: 'Overall'
        }

        clusters_to_dag = {
            'Ductal': 0, 'Ngn3 low EP': 1, 'Ngn3 high EP': 2,
            'Pre-endocrine': 3, 'Alpha': 4, 'Beta': 5, 'Epsilon': 6
        }

        all_to_dag = {
            **clusters_to_dag, 'Overall': 7
        }

        return cell_types, index_to_clusters, index_to_all_types, clusters_to_dag, all_to_dag
    else:
        raise ValueError("Unsupported data type!")

def plot_classifier_celltype_shap_values(data, relevant_indices, index_to_all_types, latent_vars=6):
    """
    Plots SHAP values with error bars as subplots for each cell type.

    Args:
        data (np.array): SHAP values array.
        relevant_indices (list): Indices corresponding to each cell type.
        index_to_all_types (dict): Mapping of cell type indices to names.
        latent_vars (int): Number of latent variables.
    """
    
    cell_types = len(relevant_indices)
    colors = sns.color_palette("deep", latent_vars)

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  
    axes = axes.flatten()

    for type_idx in range(cell_types):
        ax = axes[type_idx]
        
        indices = relevant_indices[type_idx]
        mean_values = np.mean(np.abs(data[indices, :, type_idx]), axis=0)
        std_values = np.std(np.abs(data[indices, :, type_idx]), axis=0)

        # Plot each latent variable as a bar with error bars
        for var_idx in range(latent_vars):
            ax.bar(var_idx, mean_values[var_idx], yerr=std_values[var_idx], 
                   color=colors[var_idx], capsize=5)

        ax.set_title(f'{index_to_all_types[type_idx]}')
        ax.set_xlabel('Latent Variables')
        ax.set_ylabel('Mean SHAP Values')
        ax.set_xticks(np.arange(latent_vars))
        
    # Hide empty subplot if cell_types < 8
    if cell_types < len(axes):
        for ax in axes[cell_types:]:
            ax.axis('off')

    # Add shared legend below the subplots
    handles = [plt.Line2D([0], [0], color=color, label=f'LV {i}') for i, color in enumerate(colors)]
    fig.legend(handles=handles, title='Latent Variables', loc='lower center', ncol=latent_vars)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Add space for legend
    plt.show()


# --- Function to plot a single legend ---
def plot_legend(latent_vars=6):
    """
    Plots a separate legend for latent variables.

    Args:
        latent_vars (int): Number of latent variables.
    """
    colors = sns.color_palette("deep", latent_vars)
    
    fig_legend, ax_legend = plt.subplots(figsize=(6, 2))
    handles = [plt.Line2D([0], [0], color=color, label=f'LV {i}') for i, color in enumerate(colors)]
    
    ax_legend.legend(handles=handles, title='Latent Variables', loc='center')
    ax_legend.axis('off')
    plt.show()

def find_top_k_as_weighted_dict(frame, name_genes, top_k=50,):
   top_indices = np.argsort(-frame, axis=-1)
   gene_set=set()
   top_values=(np.take_along_axis(frame, top_indices, axis=-1))
   
   top_indices=top_indices[:, :top_k]
   top_values=top_values[:,:top_k]
   # print(top_values)
   dict_ = {f"LV-{i}": {} for i in range(len(top_indices))}
   for i, (indices, values) in enumerate(zip(top_indices, top_values)):
      for index, value in zip(indices, values):  # Reverting values back to positive
         gene_set.add(name_genes[index])
         dict_[f"LV-{i}"][f"{name_genes[index]}"] = value
   return dict_, gene_set

def extract_clusters(graph):
    # Louvain algorithm
    clusters = louvain(graph, resolution=0.3)
    return clusters

def louvain(graph, resolution):
    partition = community_louvain.best_partition(graph,randomize=True,resolution=resolution)
    # print_partitions(partition)
    return partition
def visualize_graph_with_communities(G):
   # Find the communities
   partition = community_louvain.best_partition(G,randomize=True,resolution=0.3)
   # Create a color map for the communities
   pos = nx.spring_layout(G)
   cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
   nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
   nx.draw_networkx_edges(G, pos, alpha=0.5)
   plt.show()

import networkx as nx

def build_graph(graph_dict, active_terms):
    """
    Builds a NetworkX graph from weighted SHAP values and active terms.

    Args:
        graph_dict (dict): Dictionary containing nodes and their weighted connections.
        active_terms (dict): Dictionary of latent variable interactions.

    Returns:
        nx.Graph: Constructed NetworkX graph.
    """
    G = nx.Graph()

    # Add SHAP value-based edges
    for node, edges in graph_dict.items():
        for adj_node, weight in edges.items():
            G.add_edge(node, adj_node, weight=weight)

    # Add active term interactions
    for lhs, rhs_var in active_terms.items():
        for rhs in rhs_var:
            G.add_edge(f'LV-{lhs}', f'LV-{rhs}')

    return G


def generate_gene_lv_graphs(lv_gene_shap_values, index_to_class, index_to_clusters, 
                    active_terms,name_genes, top_k_lv=200):
    """
    Generates NetworkX graphs for each cell type based on SHAP values.

    Args:
        lv_gene_shap_values (np.array): SHAP values [LV, Samples, Genes].
        index_to_class (dict): Mapping of cell type indices to sample indices.
        index_to_clusters (dict): Mapping of cell type indices to cluster names.
        active_terms (dict): Dictionary of latent variable interactions.
        top_k_lv (int): Number of top SHAP values to include.

    Returns:
        tuple: (results, list of graphs, list of titles)
    """
    results = {}
    graphs_recon = []
    titles = []

    for cell_type_idx in range(len(index_to_clusters)):
        cell_type_name = index_to_clusters[cell_type_idx]

        # Compute mean SHAP values across samples
        mean_shap = np.mean(np.abs(lv_gene_shap_values[cell_type_idx]), axis=1)
        
        # Get top-k SHAP values as weighted dictionary
        results[cell_type_name], gene_set = find_top_k_as_weighted_dict(mean_shap, top_k=top_k_lv,name_genes=name_genes)
        
        # Build graph
        graph_dict = results[cell_type_name]
        G = build_graph(graph_dict, active_terms)

        # Append results
        graphs_recon.append(G)
        titles.append(cell_type_name)

    return results, graphs_recon, titles

# Assuming G is your graph and partition is the dictionary of community assignments
def community_layout(G, partition):
    """
    Compute the layout for a modular graph.
    """

    # Create a new graph to represent the communities
    pos_communities = _position_communities(G, partition, scale=3.)
    
    # Create a layout for individual nodes
    pos_nodes = _position_nodes(G, partition, scale=1.)
    
    # Combine positions
    pos = dict()
    for node in G.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(G, partition, scale=1.):
    # Create a weighted graph, in which each node corresponds to a community,
    # and each edge weight corresponds to the number of edges between communities
    community_graph = nx.Graph()
    for node, com_id in partition.items():
        community_graph.add_node(com_id)
        for neighbor, data in G[node].items():
            neighbor_com_id = partition[neighbor]
            if com_id != neighbor_com_id:
                if community_graph.has_edge(com_id, neighbor_com_id):
                    community_graph[com_id][neighbor_com_id]['weight'] += len(data)
                else:
                    community_graph.add_edge(com_id, neighbor_com_id, weight=len(data))
    
    # Compute the spring layout for the community graph
    pos_communities = nx.spring_layout(community_graph, scale=scale)
    
    # Set the position of each node to the position of its community
    pos = dict()
    for node, com_id in partition.items():
        pos[node] = pos_communities[com_id]
    return pos

def _position_nodes(G, partition, scale=1.):
    """
    Positions nodes within their communities.
    """
    pos = dict()
    for com in set(partition.values()):
        # Create a subgraph for each community
        subgraph = G.subgraph([node for node in partition if partition[node] == com])
        
        # Apply the spring layout to the subgraph
        pos_subgraph = nx.spring_layout(subgraph, scale=scale)
        
        pos.update(pos_subgraph)
    return pos

def get_community_colors(graph, partition, color_palette):
    """
    Get color mapping for communities.
    """
    return {node: color_palette[partition[node] % len(color_palette)] for node in graph.nodes()}

def visualize_gene_lv_graphs(graphs, titles, res, index_to_cell_type, clusters_to_dag, save_name='All_graphs_plot'):
    """
    Visualize and save the grid of gene-LV graphs with community coloring.

    Args:
        graphs (list): List of NetworkX graphs.
        titles (list): Titles for each graph.
        res (float): Resolution parameter for Louvain partitioning.
        index_to_cell_type (dict): Mapping of indices to cell types.
        clusters_to_dag (dict): Mapping of clusters to DAG positions.
        save_name (str): Directory name for saving the plots.
    """
    # Create directory
    os.makedirs(save_name, exist_ok=True)

    graph_len = len(graphs)
    partitions = []
    positions = []

    # Color palette
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ffbb78', '#98df8a', 
        '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', 
        '#9edae5', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#17becf',
        '#bcbd22', '#7f7f7f', '#e377c2', '#8c564b', '#9467bd', '#d62728',
        '#2ca02c', '#ff7f0e', '#1f77b4', '#ffbb78', '#98df8a', '#ff9896'
    ]

    # Initialize figure
    fig, axes = plt.subplots(nrows=graph_len, ncols=graph_len, figsize=(80, 80))

    # Generate partitions and positions
    for i in range(graph_len):
        partition = community_louvain.best_partition(graphs[i], resolution=res)
        partitions.append(partition)
        positions.append(nx.spring_layout(graphs[i]))

    # Plot the grid
    for i in range(graph_len):
        for j in range(graph_len):
            ax = axes[i, j]
            G = graphs[j]

            # Get colors based on communities
            community_colors = get_community_colors(G, partitions[j], color_palette)
            node_colors = [community_colors[node] for node in G.nodes()]

            pos = positions[j]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray', style='dotted', alpha=0.2, ax=ax)

            # Add community labels
            communities = set(partitions[j].values())
            for community in communities:
                community_nodes = [node for node in G.nodes() if partitions[j][node] == community]
                x_coords = [pos[node][0] for node in community_nodes]
                y_coords = [pos[node][1] for node in community_nodes]
                if x_coords and y_coords:
                    x_text = sum(x_coords) / len(x_coords)
                    y_text = sum(y_coords) / len(y_coords)
                    ax.text(x_text, y_text, f'Community {community}', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

            # Set plot titles
            ax.set_title(f'Coloring Scheme - {index_to_cell_type[i]}\nCell Type - {index_to_cell_type[j]}')
            ax.set_axis_off()

            # # Save individual plots
            # if i == j:
            #     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #     fig.savefig(f"{save_name}/{index_to_cell_type[i]}_plot.pdf", bbox_inches=extent.expanded(1.1, 1.1)) 

    plt.suptitle("Gene-LV Graph Grid Visualization", fontsize=35)
    plt.tight_layout()
    
    # Save the entire grid
    plt.savefig(f"{save_name}/All_graphs_combined.pdf")
    plt.savefig(f"{save_name}/All_graphs_combined.png")
    plt.show()

    return partitions

def enrichment_bubble_plot_colored(enrichment_results, color_dict,save_name='Default'):
    # Extract all unique names across all lists
    all_names = set()
    for key, df in enrichment_results.items():
        all_names.update(df['name'])

    # Create a DataFrame to store p_values for each name and each list
    p_values_matrix = pd.DataFrame(index=all_names, columns=list(enrichment_results.keys()))

    # Fill the matrix with p_values
    for key, df in enrichment_results.items():
        for _, row in df.iterrows():
            p_values_matrix.loc[row['name'], key] = -np.log10(row['p_value'])  # Log-transform for better visualization

    # Preparing data for bubble plot
    bubble_data = p_values_matrix.reset_index().melt(id_vars='index')
    bubble_data.columns = ['name', 'gene_cluster', 'neg_log_p_value']
    
    # Adding a new column for coloring logic
    def get_color(group, color_dict):
        return ' '.join(color_dict[group])
    
    
    # Extracting cell_type from gene_cluster
    bubble_data['cell_type'] = bubble_data['gene_cluster'].apply(lambda x: ''.join(x.split('_')[:-1]))
    
    # Sorting bubble_data by color_group to make them consecutive
    for key, val in color_dict.items():
        if len(val)>0:
            bubble_data['color_group'] = bubble_data['gene_cluster'].apply(lambda x: get_color(x, color_dict))

            bubble_data.sort_values('color_group', inplace=True)
            break
        else:
            bubble_data['color_group'] = bubble_data['gene_cluster'].apply(lambda x: ''.join(x.split('_')[:-1]))
    # Plotting the bubble plot
    plt.figure(figsize=(20, 30))
    bubble_plot = sns.scatterplot(
        data=bubble_data,
        x='gene_cluster', 
        y='name', 
        size='neg_log_p_value', 
        hue='cell_type',  # Updated to hue based on cell_type
        sizes=(20, 2000), 
        legend='brief', 
        palette='tab10',
        edgecolor='w', 
        alpha=0.6
    )
    
    # Customizing plot
    bubble_plot.set_title('Enrichment Analysis Results (-log10(p_value))')
    bubble_plot.set_xlabel('Gene Clusters')
    bubble_plot.set_ylabel('GO Biological Process')
    bubble_plot.set_xticklabels(bubble_plot.get_xticklabels(), rotation=45)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Add demarcation lines and labels between different color groups
    color_groups = bubble_data['color_group'].unique()
    xticks = []
    xticklabels = []
    count = 0
    for color_group in color_groups:
        group_clusters = bubble_data[bubble_data['color_group'] == color_group]['gene_cluster'].unique()
        group_center = count + len(group_clusters) / 2 - 0.5  # Center position for label
        for cluster in group_clusters:
            xticks.append(count)
            xticklabels.append(cluster)
            count += 1
        plt.axvline(x=count-0.5, color='black', linestyle='--', linewidth=2)
        plt.text(group_center, -1, color_group, ha='center', va='center', fontsize=12, fontweight='bold', color='black', rotation=45)
    
    # Set x-axis values
    bubble_plot.set_xticks(xticks)
    bubble_plot.set_xticklabels(xticklabels, rotation=45)
    os.makedirs(save_name, exist_ok=True)
    plt.savefig(f"{save_name}/GSEA_bubble_plot.pdf")
    plt.show()
def extract_p_values(enrichment_results):
    all_names = set()
    for key, df in enrichment_results.items():
        all_names.update(df['name'])

    # Create a DataFrame to store p_values for each name and each list
    p_values_matrix = pd.DataFrame(index=all_names, columns=list(enrichment_results.keys()))

    # Fill the matrix with p_values
    for key, df in enrichment_results.items():
        for _, row in df.iterrows():
            p_values_matrix.loc[row['name'], key] = -np.log10(row['p_value'])  # Log-transform for bette
    return p_values_matrix

def apply_g_profiler(partition_to_df,organism='mmusculus',top=None):
    gp = GProfiler(return_dataframe=True)
    enrichment_results={}
    map_lv_to_clusters={}
    for idx, rows in tqdm(partition_to_df.iteritems(),total=len(partition_to_df)):
        for community, row in enumerate(rows):
            try:
                row=eval(row)
                correct_row = {item for item in row if 'LV-' not in item}  # Filter out items containing 'LV-'
                lvs=[item for item in row if 'LV-' in item]
                if correct_row:  # Proceed if correct_row is not empty
                    df = gp.profile(organism=organism, query=list(correct_row))
                    # Filter the results to include only p_values > 5
                    filtered_df = df[(df['source'] == 'GO:BP') & (df['p_value'] < 1e-5)][['p_value', 'name']]
                    print(f"For the current cluster, Number of all cell functions {len(df[(df['source'] == 'GO:BP')])},Number of cell functions with less than e-5 p value {len(filtered_df)}")
                    if top:
                        out = filtered_df.head(top)
                    else:
                        out = filtered_df
                    enrichment_results[f'{idx}_{community}'] = out
                    map_lv_to_clusters[f'{idx}_{community}'] = lvs
            except:
                print("Error evaluating", row)
    return enrichment_results, map_lv_to_clusters
