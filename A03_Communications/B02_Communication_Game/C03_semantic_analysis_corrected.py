#!/usr/bin/env python3
"""
Corrected Semantic Detail Analysis for Translation Module
Analyzes teacher-student pairs individually and provides global summary
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.io import loadmat
import argparse
from tqdm import tqdm
import pandas as pd

# Add dependencies path (same as in the original test script)
script_dir = Path(__file__).parent
project_root = script_dir
while not (project_root / 'Deps').exists() and project_root.parent != project_root:
    project_root = project_root.parent

# Add project root to the Python path if it's not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

sys.path.append(os.path.join(project_root, "Deps", "CustomFuctions"))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import Translators

# CIFAR100 class names for visualization
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

def load_teacher_symbols():
    """Load teacher (speaker) symbols for all 100 classes"""
    print("Loading teacher symbols...")
    teacher_symbols_path = './Testing_Symbols_of_Speaker/context_ep_1999.mat'
    teacher_data = loadmat(teacher_symbols_path)
    teacher_symbols = teacher_data['context_0_1999']  # Shape: [100, symbol_dim]
    print(f"Loaded teacher symbols: {teacher_symbols.shape}")
    return teacher_symbols

def perform_hierarchical_clustering(teacher_symbols, n_clusters=10, method='ward'):
    """
    Perform hierarchical clustering on teacher symbols
    
    Args:
        teacher_symbols: Teacher symbols array [100, context_dim]
        n_clusters: Number of clusters to form
        method: Linkage method for clustering
    
    Returns:
        cluster_labels: Cluster assignment for each class
        cluster_order: New ordering based on clustering
        linkage_matrix: Linkage matrix for dendrogram
    """
    print(f"Performing hierarchical clustering with {n_clusters} clusters...")
    
    # Calculate distance matrix
    distances = pdist(teacher_symbols, metric='euclidean')
    linkage_matrix = linkage(distances, method=method)
    
    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Create new ordering based on clustering
    # Sort by cluster labels, then by original index within each cluster
    cluster_order = np.argsort(cluster_labels)
    
    print(f"Clustering complete. Cluster distribution:")
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        cluster_classes = [CIFAR100_CLASSES[i] for i in np.where(cluster_labels == cluster_id)[0]]
        print(f"  Cluster {cluster_id}: {count} classes - {cluster_classes[:5]}{'...' if count > 5 else ''}")
    
    return cluster_labels, cluster_order, linkage_matrix

def load_translation_model(test_id):
    """Load translation model for specific test_id"""
    model_path = f'./2025-06-02/checkpoint/TInet_testid_{test_id}.pth'
    if not os.path.exists(model_path):
        # Try alternative path
        alt_model_path = f"./models/student_{test_id}.pth"
        if not os.path.exists(alt_model_path):
            raise FileNotFoundError(f"Model not found: {model_path} or {alt_model_path}")
        model_path = alt_model_path
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model (using same parameters as in the original script)
    model = Translators.TImodule(context_dim=20, num_hidden_layer=10,
                                num_hidden_neuron=500, dropout_p=0.3)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])  # Note: checkpoint contains 'net' key
    else:
        model.load_state_dict(checkpoint)  # Direct state dict
    model.eval()
    model.to(device)
    
    return model

def extract_translation_features(model, teacher_symbols):
    """
    Extract features from input layer, ReLU activations and final output using real model forward pass
    
    Args:
        model: Translation model (TImodule)
        teacher_symbols: Input teacher symbols [n_samples, context_dim]
    
    Returns:
        Dictionary of features from input layer, ReLU layers and final output with shape [n_samples, hidden_dim]
    """
    features = {}
    hooks = []
    
    # Get device from model
    device = next(model.parameters()).device
    
    def create_hook(name):
        def hook_fn(module, input, output):
            # Store the activation
            features[name] = output.detach().cpu().numpy()
        return hook_fn
    
    # Register hooks for ReLU layers only
    relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hook_name = f'relu_{relu_count}'
            hook = module.register_forward_hook(create_hook(hook_name))
            hooks.append(hook)
            relu_count += 1
    
    # Convert to tensor and move to device
    x = torch.FloatTensor(teacher_symbols).to(device)
    
    with torch.no_grad():
        # Store original input data as the first feature layer
        features['input'] = teacher_symbols  # Original input data before tensor conversion
        
        # Forward pass through the model - this will trigger all hooks
        output = model(x)
        
        # Store final output
        features['output'] = output.detach().cpu().numpy()
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Print feature shapes for debugging
    print("Extracted features (input + ReLU activations + output):")
    for layer_name, feature_array in features.items():
        print(f"  {layer_name}: {feature_array.shape}")
    
    return features

def adjust_text_positions(points, labels, fig_width=12, fig_height=8, min_distance=0.05):
    """
    Adjust text positions to avoid overlaps using a simple repulsion algorithm
    
    Args:
        points: List of (x, y) coordinates for data points
        labels: List of label texts
        fig_width, fig_height: Figure dimensions for normalization
        min_distance: Minimum distance between labels
    
    Returns:
        List of adjusted (x, y) positions for labels
    """
    points = np.array(points)
    
    # Normalize coordinates to [0, 1] range for easier distance calculation
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    if x_max == x_min or y_max == y_min:
        return points.tolist()
    
    normalized_points = np.column_stack([
        (points[:, 0] - x_min) / (x_max - x_min),
        (points[:, 1] - y_min) / (y_max - y_min)
    ])
    
    # Initialize label positions with small offsets from data points
    label_positions = normalized_points + np.random.normal(0, 0.02, normalized_points.shape)
    
    # Iterative adjustment to reduce overlaps
    for iteration in range(50):
        forces = np.zeros_like(label_positions)
        
        for i in range(len(label_positions)):
            for j in range(len(label_positions)):
                if i != j:
                    diff = label_positions[i] - label_positions[j]
                    distance = np.linalg.norm(diff)
                    
                    if distance < min_distance and distance > 0:
                        # Apply repulsive force
                        force_magnitude = (min_distance - distance) / min_distance
                        force_direction = diff / distance
                        forces[i] += force_magnitude * force_direction * 0.1
        
        # Apply forces
        label_positions += forces
        
        # Keep labels within reasonable bounds
        label_positions[:, 0] = np.clip(label_positions[:, 0], -0.2, 1.2)
        label_positions[:, 1] = np.clip(label_positions[:, 1], -0.2, 1.2)
    
    # Convert back to original coordinate system
    adjusted_positions = np.column_stack([
        label_positions[:, 0] * (x_max - x_min) + x_min,
        label_positions[:, 1] * (y_max - y_min) + y_min
    ])
    
    return adjusted_positions.tolist()

def visualize_layer_features(features_dict, cluster_labels, cluster_order, class_indices, 
                            test_id, excluded_class, output_dir):
    """
    Create dimensionality reduction visualizations for each layer's features
    
    Args:
        features_dict: Dictionary of features from different layers
        cluster_labels: Cluster assignments for each class
        cluster_order: Ordering based on clustering
        class_indices: Indices of classes being analyzed
        test_id: Student test ID
        excluded_class: Excluded class index
        output_dir: Output directory for visualizations
    """
    print("Creating layer-wise visualizations...")
    
    # Create subdirectory for layer visualizations
    layer_viz_dir = os.path.join(output_dir, 'layer_visualizations')
    os.makedirs(layer_viz_dir, exist_ok=True)
    
    # Color map for clusters
    n_clusters = len(np.unique(cluster_labels))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Ensure input layer is processed first, then ReLU layers, then output
    layer_order = ['input'] + [f'relu_{i}' for i in range(20)] + ['output']
    ordered_layers = [layer for layer in layer_order if layer in features_dict]
    
    for layer_name in ordered_layers:
        features = features_dict[layer_name]
        if features.shape[1] < 2:  # Skip if feature dimension is too small
            continue
             
        print(f"  Processing {layer_name}: {features.shape}")
        
        # Perform t-SNE for visualization
        if features.shape[1] > 50:  # Use PCA first if dimension is too high
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
        tsne_results = tsne.fit_transform(features_reduced)
        
        # Create visualization
        plt.figure(figsize=(14, 10))  # Slightly larger figure for better label spacing
        
        # Collect data for label positioning
        point_positions = []
        point_info = []
        
        # Plot points colored by cluster
        for i, (x, y) in enumerate(tsne_results):
            # Use all classes (0-99) for features, but class_indices for excluded class info
            class_idx = i  # Direct mapping since features include all 100 classes
            cluster_id = cluster_labels[class_idx] - 1  # Convert to 0-based indexing
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Mark excluded class differently with enhanced visibility
            if class_idx == excluded_class:
                marker = 'X'  # Larger X marker
                size = 200    # Larger size
                alpha = 0.9   # Higher alpha for visibility
                edgecolor = 'red'  # Red edge for emphasis
                linewidth = 2
            else:
                marker = 'o'
                size = 100
                alpha = 0.7
                edgecolor = 'black'
                linewidth = 0.5
            
            plt.scatter(x, y, c=[color], s=size, alpha=alpha, marker=marker,
                       edgecolors=edgecolor, linewidths=linewidth,
                       label=f'Cluster {cluster_id+1}' if i == 0 or cluster_labels[class_idx] != cluster_labels[i-1] else "")
            
            # Store point information for label positioning
            point_positions.append((x, y))
            point_info.append({
                'class_idx': class_idx,
                'class_name': CIFAR100_CLASSES[class_idx],
                'is_excluded': class_idx == excluded_class,
                'color': '#2F2F2F'
            })
        
        # Adjust label positions to avoid overlaps
        adjusted_positions = adjust_text_positions(
            point_positions, 
            [info['class_name'] for info in point_info],
            fig_width=14, 
            fig_height=10
        )
        
        # Add labels with connecting lines
        for i, (point_pos, label_pos, info) in enumerate(zip(point_positions, adjusted_positions, point_info)):
            px, py = point_pos
            lx, ly = label_pos
            
            line_color = 'red' if info['is_excluded'] else 'gray'
            line_alpha = 0.8 if info['is_excluded'] else 0.6
            line_width = 1.2 if info['is_excluded'] else 0.8
            
            # Draw connecting line for all labels
            plt.plot([px, lx], [py, ly], 
                    color=line_color, alpha=line_alpha, linewidth=line_width, zorder=1)
            
            # Add text label
            fontsize = 9 if info['is_excluded'] else 7
            weight = 'bold' if info['is_excluded'] else 'normal'
            
            plt.annotate(info['class_name'], (lx, ly), 
                        fontsize=fontsize, 
                        color=info['color'], 
                        alpha=0.9,
                        weight=weight,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                alpha=0.8, 
                                edgecolor='red' if info['is_excluded'] else 'lightgray',
                                linewidth=1.5 if info['is_excluded'] else 0.5),
                        zorder=3)
        
        plt.title(f'Layer: {layer_name}\nTeacher-Student Pair {test_id} (Excluded: {CIFAR100_CLASSES[excluded_class]})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Add legend for clusters
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(layer_viz_dir, f'{layer_name}_tsne.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Layer visualizations saved to {layer_viz_dir}")

def compute_rdm_with_reordering(features_dict, cluster_order, output_dir):
    """
    Compute RDM (Representational Dissimilarity Matrix) for each layer and reorder based on clustering
    Apply percentile normalization for better visualization
    
    Args:
        features_dict: Dictionary of features from different layers
        cluster_order: New ordering based on clustering results
        output_dir: Output directory for RDM visualizations
    
    Returns:
        Dictionary of reordered RDMs for each layer
    """
    print("Computing RDMs with cluster-based reordering...")
    
    rdm_dir = os.path.join(output_dir, 'rdm_analysis')
    os.makedirs(rdm_dir, exist_ok=True)
    
    rdms = {}
    rdms_raw = {}  # Store raw RDMs for correlation analysis
    
    # Ensure input layer is processed first, then ReLU layers, then output
    layer_order = ['input'] + [f'relu_{i}' for i in range(20)] + ['output']
    ordered_layers = [layer for layer in layer_order if layer in features_dict]
    
    for layer_name in ordered_layers:
        features = features_dict[layer_name]
        print(f"  Computing RDM for {layer_name}: {features.shape}")
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(features)
        
        # Convert to dissimilarity matrix (1 - correlation)
        rdm = 1 - corr_matrix
        
        # Reorder based on clustering
        rdm_reordered = rdm[cluster_order][:, cluster_order]
        rdms_raw[layer_name] = rdm_reordered
        
        # Apply percentile normalization (0-100 percentile mapping)
        rdm_percentile = np.zeros_like(rdm_reordered)
        for i in range(rdm_reordered.shape[0]):
            for j in range(rdm_reordered.shape[1]):
                rdm_percentile[i, j] = (rdm_reordered <= rdm_reordered[i, j]).sum() / rdm_reordered.size * 100
        
        rdms[layer_name] = rdm_percentile
        
        # Create RDM visualization
        plt.figure(figsize=(10, 8))
        
        # Plot percentile-normalized RDM
        im = plt.imshow(rdm_percentile, cmap='viridis', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, label='Dissimilarity Percentile (%)')
        plt.title(f'RDM - {layer_name} (Cluster-ordered, Percentile-normalized)')
        plt.xlabel('Class Index (Cluster-ordered)')
        plt.ylabel('Class Index (Cluster-ordered)')
        
        # RDM visualization without text overlay to avoid blocking the matrix
        
        plt.tight_layout()
        plt.savefig(os.path.join(rdm_dir, f'{layer_name}_rdm.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comparison plot showing RDM evolution across layers
    create_rdm_comparison_plot(rdms, rdm_dir)
    
    # Create RDM correlation matrix for layer-wise comparison
    create_rdm_correlation_matrix(rdms_raw, rdm_dir)
    
    print(f"RDM analysis saved to {rdm_dir}")
    return rdms

def create_rdm_comparison_plot(rdms, output_dir):
    """
    Create a comparison plot showing RDM evolution across layers
    """
    layer_names = list(rdms.keys())
    n_layers = len(layer_names)
    
    if n_layers < 2:
        return
    
    fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    # Plot RDMs
    for i, (layer_name, rdm) in enumerate(rdms.items()):
        im = axes[0, i].imshow(rdm, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'{layer_name}')
        axes[0, i].set_xlabel('Class Index')
        axes[0, i].set_ylabel('Class Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
    
    # Plot RDM differences between consecutive layers
    for i in range(n_layers - 1):
        layer1_name = layer_names[i]
        layer2_name = layer_names[i + 1]
        
        rdm_diff = rdms[layer2_name] - rdms[layer1_name]
        
        im = axes[1, i].imshow(rdm_diff, cmap='RdBu_r', aspect='auto', 
                              vmin=-np.max(np.abs(rdm_diff)), vmax=np.max(np.abs(rdm_diff)))
        axes[1, i].set_title(f'Δ({layer2_name} - {layer1_name})')
        axes[1, i].set_xlabel('Class Index')
        axes[1, i].set_ylabel('Class Index')
        
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    # Hide the last subplot in the second row if there's an odd number of layers
    if n_layers > 1:
        axes[1, -1].axis('off')
    
    plt.suptitle('RDM Evolution Across Translation Module Layers', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rdm_evolution_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_rdm_correlation_matrix(rdms_raw, output_dir):
    """
    Create correlation matrix between RDMs from different layers
    
    Args:
        rdms_raw: Dictionary of raw (non-percentile) RDMs for each layer
        output_dir: Output directory for correlation matrix visualization
    """
    print("Computing RDM correlation matrix...")
    
    layer_names = list(rdms_raw.keys())
    n_layers = len(layer_names)
    
    if n_layers < 2:
        print("  Not enough layers for correlation analysis")
        return
    
    # Initialize correlation matrix
    correlation_matrix = np.zeros((n_layers, n_layers))
    
    # Compute pairwise correlations between RDMs
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            rdm1 = rdms_raw[layer1]
            rdm2 = rdms_raw[layer2]
            
            # Flatten upper triangular part (excluding diagonal) for correlation
            mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
            rdm1_flat = rdm1[mask]
            rdm2_flat = rdm2[mask]
            
            # Compute Spearman correlation
            correlation, _ = spearmanr(rdm1_flat, rdm2_flat)
            correlation_matrix[i, j] = correlation
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot correlation matrix
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Spearman Correlation')
    
    # Add correlation values as text
    for i in range(n_layers):
        for j in range(n_layers):
            text = plt.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white",
                           fontsize=8)
    
    # Set labels and title
    plt.xticks(range(n_layers), layer_names, rotation=45, ha='right')
    plt.yticks(range(n_layers), layer_names)
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title('RDM Correlation Matrix\n(Spearman correlation between layer representations)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rdm_correlation_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation matrix as CSV for further analysis
    correlation_df = pd.DataFrame(correlation_matrix, index=layer_names, columns=layer_names)
    correlation_df.to_csv(os.path.join(output_dir, 'rdm_correlation_matrix.csv'))
    
    print(f"  RDM correlation matrix saved to {output_dir}")
    print(f"  Average inter-layer correlation: {np.mean(correlation_matrix[np.triu_indices(n_layers, k=1)]):.3f}")
    
    return correlation_matrix

def comprehensive_semantic_analysis(teacher_symbols, test_id, excluded_class, 
                                   n_clusters=5, extract_layers=True, visualize=True):
    """
    Comprehensive semantic analysis including hierarchical clustering, layer-wise feature extraction,
    visualization, and RDM analysis
    
    Args:
        teacher_symbols: Teacher symbol representations (100, context_dim)
        test_id: Student test ID
        excluded_class: Class to exclude from analysis
        n_clusters: Number of clusters for hierarchical clustering
        extract_layers: Whether to extract layer features
        visualize: Whether to create visualizations
    
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n=== Comprehensive Semantic Analysis for Pair {test_id} ===")
    print(f"Excluded class: {CIFAR100_CLASSES[excluded_class]}")
    
    # Create output directory
    output_dir = f"./semantic_analysis_results/pair_{test_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Perform hierarchical clustering on teacher symbols
    print("\nStep 1: Performing hierarchical clustering on teacher symbols...")
    cluster_labels, cluster_order, linkage_matrix = perform_hierarchical_clustering(
        teacher_symbols, n_clusters=n_clusters
    )
    
    # Save clustering results (only essential results)
    clustering_results = {
        'cluster_labels': cluster_labels,
        'cluster_order': cluster_order,
        'linkage_matrix': linkage_matrix
    }
    np.save(os.path.join(output_dir, 'clustering_results.npy'), clustering_results)
    
    # Create dendrogram visualization
    if visualize:
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=[CIFAR100_CLASSES[i] for i in range(100)], 
                  leaf_rotation=90, leaf_font_size=8)
        plt.title(f'Hierarchical Clustering Dendrogram\nTeacher-Student Pair {test_id}')
        plt.xlabel('Class Names')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_dendrogram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Step 2: Extract translation module features if requested
    features_dict = {}
    if extract_layers:
        print("\nStep 2: Extracting translation module features...")
        
        # Load the student model
        model_path = f'./2025-06-02/checkpoint/TInet_testid_{test_id}.pth'
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping feature extraction.")
            return None
        
        # Load model (TImodule model)
        try:
            import torch
            import torch.nn as nn
            
            # Define TImodule class (from Translators.py)
            class TImodule(nn.Module):
                def __init__(self, context_dim, num_hidden_layer, num_hidden_neuron, dropout_p):
                    super(TImodule, self).__init__()
                    self.input_layer = nn.Sequential(nn.Linear(context_dim, num_hidden_neuron),
                                                     nn.BatchNorm1d(num_hidden_neuron),                                   
                                                     nn.Dropout(dropout_p),
                                                     nn.ReLU())
                    self.hidden_layers = nn.ModuleList()
                    for i in range(0, num_hidden_layer):
                        hidden_layer = nn.Linear(num_hidden_neuron, num_hidden_neuron)
                        bn_layer = nn.BatchNorm1d(num_hidden_neuron)
                        activation_layer = nn.ReLU()
                        dropout_layer = nn.Dropout(dropout_p)
                        self.hidden_layers.append(hidden_layer)
                        self.hidden_layers.append(bn_layer)
                        self.hidden_layers.append(dropout_layer)
                        self.hidden_layers.append(activation_layer)
                    self.output_layer = nn.Linear(num_hidden_neuron, context_dim)
                    self.dropout = nn.Dropout(dropout_p)
                    self.tanh = nn.Tanh()
                    self.sigmoid = nn.Sigmoid()
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.input_layer(x)
                    for m in self.hidden_layers:
                        x = m(x)
                    output = self.output_layer(x)
                    return output
            
            # Initialize model with correct parameters (from training script)
            model = TImodule(context_dim=20, num_hidden_layer=10, num_hidden_neuron=500, dropout_p=0.3)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'net' in checkpoint:
                model.load_state_dict(checkpoint['net'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            # Extract features from translation module
            features_dict = extract_translation_features(model, teacher_symbols)
            
            # Return features without saving to files
            print(f"Extracted features from {len(features_dict)} layers")
                
        except Exception as e:
            print(f"Error loading model or extracting features: {e}")
            print(f"Error details: {str(e)}")
            return None
    
    # Get class indices for analysis (excluding the specified class)
    class_indices = [i for i in range(100) if i != excluded_class]
    
    # Step 3: Create layer-wise visualizations
    if visualize and features_dict:
        print("\nStep 3: Creating layer-wise visualizations...")
        visualize_layer_features(features_dict, cluster_labels, cluster_order, 
                                class_indices, test_id, excluded_class, output_dir)
    
    # Step 4: Compute RDMs with cluster-based reordering
    rdms = {}
    if features_dict:
        print("\nStep 4: Computing RDMs with cluster-based reordering...")
        rdms = compute_rdm_with_reordering(features_dict, cluster_order, output_dir)
        
        # Save RDM results
        np.save(os.path.join(output_dir, 'rdms.npy'), rdms)
    
    # Compile comprehensive results
    results = {
        'test_id': test_id,
        'excluded_class': excluded_class,
        'clustering': clustering_results,
        'features_shapes': {name: features.shape for name, features in features_dict.items()},
        'rdm_shapes': {name: rdm.shape for name, rdm in rdms.items()},
        'output_dir': output_dir
    }
    
    # Save comprehensive results summary
    import json
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'test_id': results['test_id'],
            'excluded_class': results['excluded_class'],
            'features_shapes': results['features_shapes'],
            'rdm_shapes': results['rdm_shapes'],
            'output_dir': results['output_dir'],
            'cluster_distribution': {f'cluster_{i+1}': int(np.sum(cluster_labels == i+1)) 
                                   for i in range(n_clusters)}
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nComprehensive analysis completed! Results saved to {output_dir}")
    return results

def analyze_teacher_student_pair(teacher_symbols, test_id, excluded_class, 
                                 extract_layers=True, visualize=True):
    """
    Analyze a specific teacher-student pair
    
    Args:
        teacher_symbols: All teacher symbols [100, dim]
        test_id: Student ID (0-99)
        excluded_class: Class excluded from student training
        extract_layers: Whether to extract intermediate layer features
        visualize: Whether to generate visualizations
    
    Returns:
        Dictionary containing analysis results
    """
    print(f"\nAnalyzing Teacher-Student pair {test_id} (excluded class: {excluded_class})")
    
    # Load translation model for this student
    try:
        translation_model = load_translation_model(test_id)
    except FileNotFoundError:
        print(f"Model for test_id {test_id} not found, skipping...")
        return None
    
    # Get teacher symbols for the 99 classes this student learned
    student_class_indices = [i for i in range(100) if i != excluded_class]
    teacher_symbols_99 = teacher_symbols[student_class_indices]  # [99, dim]
    
    # Translate teacher symbols to student space
    with torch.no_grad():
        student_symbols_99 = translation_model(torch.FloatTensor(teacher_symbols_99)).numpy()
    
    # Calculate RSA between teacher and student symbols (99 classes)
    teacher_rdm = 1 - np.corrcoef(teacher_symbols_99)
    student_rdm = 1 - np.corrcoef(student_symbols_99)
    
    # Flatten upper triangular matrices for correlation
    mask = np.triu(np.ones_like(teacher_rdm, dtype=bool), k=1)
    teacher_rdm_flat = teacher_rdm[mask]
    student_rdm_flat = student_rdm[mask]
    
    # Calculate Spearman correlation
    rsa_corr, rsa_pvalue = spearmanr(teacher_rdm_flat, student_rdm_flat)
    
    results = {
        'test_id': test_id,
        'excluded_class': excluded_class,
        'excluded_class_name': CIFAR100_CLASSES[excluded_class],
        'rsa_correlation': rsa_corr,
        'rsa_pvalue': rsa_pvalue,
        'n_classes': 99
    }
    
    # Extract layer-wise features if requested
    if extract_layers:
        layer_features = extract_translation_features(translation_model, teacher_symbols_99)
        results['layer_features'] = layer_features
    
    # Generate visualizations if requested
    if visualize:
        visualize_pair(teacher_symbols_99, student_symbols_99, 
                      student_class_indices, test_id, excluded_class)
    
    return results

def visualize_pair(teacher_symbols, student_symbols, class_indices, test_id, excluded_class):
    """
    Create improved t-SNE visualization for a teacher-student pair
    """
    # Create output directory
    output_dir = f'./semantic_analysis_results/pair_{test_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for t-SNE
    combined_symbols = np.vstack([teacher_symbols, student_symbols])
    labels = ['Teacher'] * len(teacher_symbols) + ['Student'] * len(student_symbols)
    class_names = [CIFAR100_CLASSES[i] for i in class_indices]
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_symbols)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Define colors for different classes (using a colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_indices)))
    
    # Plot teacher symbols
    teacher_tsne = tsne_results[:len(teacher_symbols)]
    for i, (x, y) in enumerate(teacher_tsne):
        plt.scatter(x, y, c=[colors[i]], marker='o', s=100, alpha=0.7, 
                   label=f'T-{class_names[i]}' if i < 10 else "")
    
    # Plot student symbols
    student_tsne = tsne_results[len(teacher_symbols):]
    for i, (x, y) in enumerate(student_tsne):
        plt.scatter(x, y, c=[colors[i]], marker='^', s=100, alpha=0.7,
                   label=f'S-{class_names[i]}' if i < 10 else "")
    
    plt.title(f'Teacher-Student Pair {test_id}\n'
              f'Excluded Class: {CIFAR100_CLASSES[excluded_class]}\n'
              f'○ = Teacher Symbols, △ = Student Symbols', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add legend (only for first 10 classes to avoid clutter)
    if len(class_indices) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/tsne_visualization.png")

def global_analysis(all_results):
    """
    Perform global analysis across all teacher-student pairs
    """
    print("\nPerforming global analysis...")
    
    # Extract RSA correlations
    rsa_correlations = [r['rsa_correlation'] for r in all_results if r is not None]
    rsa_pvalues = [r['rsa_pvalue'] for r in all_results if r is not None]
    
    # Calculate statistics
    mean_rsa = np.mean(rsa_correlations)
    std_rsa = np.std(rsa_correlations)
    median_rsa = np.median(rsa_correlations)
    
    # Count significant correlations
    significant_pairs = sum(1 for p in rsa_pvalues if p < 0.05)
    
    print(f"Global RSA Analysis Results:")
    print(f"  Mean RSA correlation: {mean_rsa:.4f} ± {std_rsa:.4f}")
    print(f"  Median RSA correlation: {median_rsa:.4f}")
    print(f"  Significant pairs (p<0.05): {significant_pairs}/{len(rsa_correlations)}")
    
    # Create summary visualization
    plt.figure(figsize=(12, 8))
    
    # Histogram of RSA correlations
    plt.subplot(2, 2, 1)
    plt.hist(rsa_correlations, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(mean_rsa, color='red', linestyle='--', label=f'Mean: {mean_rsa:.3f}')
    plt.axvline(median_rsa, color='orange', linestyle='--', label=f'Median: {median_rsa:.3f}')
    plt.xlabel('RSA Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of RSA Correlations')
    plt.legend()
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(rsa_correlations)
    plt.ylabel('RSA Correlation')
    plt.title('RSA Correlation Distribution')
    
    # Correlation vs Test ID
    plt.subplot(2, 2, 3)
    test_ids = [r['test_id'] for r in all_results if r is not None]
    plt.scatter(test_ids, rsa_correlations, alpha=0.6)
    plt.xlabel('Test ID (Student)')
    plt.ylabel('RSA Correlation')
    plt.title('RSA Correlation by Student')
    
    # P-value distribution
    plt.subplot(2, 2, 4)
    plt.hist(rsa_pvalues, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='--', label='p=0.05')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./semantic_analysis_results/global_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    df = pd.DataFrame([r for r in all_results if r is not None])
    df.to_csv('./semantic_analysis_results/detailed_results.csv', index=False)
    
    return {
        'mean_rsa': mean_rsa,
        'std_rsa': std_rsa,
        'median_rsa': median_rsa,
        'significant_pairs': significant_pairs,
        'total_pairs': len(rsa_correlations)
    }

def main():
    """
    python C03_semantic_analysis_corrected.py --pair_id 10 --comprehensive --n_clusters 5 --extract_layers --visualize
    """
    parser = argparse.ArgumentParser(description='Comprehensive Semantic Analysis for Translation Module')
    parser.add_argument('--pair_id', type=int, default=None, 
                       help='Specific teacher-student pair to analyze (0-99)')
    parser.add_argument('--extract_layers', action='store_true', default=True,
                       help='Extract intermediate layer features')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')
    parser.add_argument('--global_only', action='store_true', default=False,
                       help='Only perform global analysis')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Perform comprehensive semantic analysis with clustering and RDM')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters for hierarchical clustering (default: 5)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('./semantic_analysis_results', exist_ok=True)
    
    # Load teacher symbols
    teacher_symbols = load_teacher_symbols()
    
    # Print analysis mode
    if args.comprehensive:
        print(f"Running in COMPREHENSIVE mode with {args.n_clusters} clusters")
    else:
        print("Running in STANDARD mode")
    
    all_results = []
    
    if args.pair_id is not None:
        # Analyze specific pair
        excluded_class = args.pair_id  # In leave-one-out, test_id == excluded_class
        
        if args.comprehensive:
            # Perform comprehensive semantic analysis
            result = comprehensive_semantic_analysis(
                teacher_symbols, args.pair_id, excluded_class,
                n_clusters=args.n_clusters, 
                extract_layers=args.extract_layers, 
                visualize=args.visualize
            )
            if result:
                print(f"\nComprehensive analysis completed for pair {args.pair_id}")
                print(f"  Results saved to: {result['output_dir']}")
        else:
            # Standard analysis
            result = analyze_teacher_student_pair(
                teacher_symbols, args.pair_id, excluded_class,
                args.extract_layers, args.visualize
            )
            if result:
                all_results.append(result)
                print(f"\nResults for pair {args.pair_id}:")
                print(f"  RSA Correlation: {result['rsa_correlation']:.4f}")
                print(f"  P-value: {result['rsa_pvalue']:.6f}")
    else:
        if args.comprehensive:
            # Perform comprehensive analysis on all pairs
            print("Performing comprehensive analysis on all teacher-student pairs...")
            for test_id in tqdm(range(100), desc="Processing pairs"):
                excluded_class = test_id  # In leave-one-out setup
                result = comprehensive_semantic_analysis(
                    teacher_symbols, test_id, excluded_class,
                    n_clusters=args.n_clusters,
                    extract_layers=args.extract_layers and not args.global_only,
                    visualize=args.visualize and not args.global_only
                )
                if result:
                    print(f"Completed comprehensive analysis for pair {test_id}")
        else:
            # Analyze all pairs or perform global analysis
            print("Analyzing all teacher-student pairs...")
            for test_id in tqdm(range(100), desc="Processing pairs"):
                excluded_class = test_id  # In leave-one-out setup
                result = analyze_teacher_student_pair(
                    teacher_symbols, test_id, excluded_class,
                    args.extract_layers and not args.global_only, 
                    args.visualize and not args.global_only
                )
                if result:
                    all_results.append(result)
            
            # Perform global analysis
            if all_results:
                global_stats = global_analysis(all_results)
                print(f"\nGlobal Analysis Complete:")
                print(f"  Processed {global_stats['total_pairs']} pairs")
                print(f"  Mean RSA: {global_stats['mean_rsa']:.4f} ± {global_stats['std_rsa']:.4f}")
                print(f"  Significant pairs: {global_stats['significant_pairs']}/{global_stats['total_pairs']}")

if __name__ == "__main__":
    main()