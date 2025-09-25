#!/usr/bin/env python3
"""
RDM Correlation Matrix Aggregation and Analysis

This script reads all rdm_correlation_matrix.csv files from semantic analysis results,
computes average RDM matrices, and performs t-tests for statistical significance.
Visualization follows the same style as the original semantic analysis code.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import seaborn as sns
from pathlib import Path
import sys

# Add dependencies path (same as in the original test script)
script_dir = Path(__file__).parent
project_root = script_dir
while not (project_root / 'Deps').exists() and project_root.parent != project_root:
    project_root = project_root.parent

# Add project root to the Python path if it's not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def find_rdm_csv_files(base_dir=os.path.join(project_root, "Results", "communication_semantic_analysis_results")):
    """
    Find all rdm_correlation_matrix.csv files in the workspace
    
    Args:
        base_dir: Base directory to search for CSV files
        
    Returns:
        List of file paths to rdm_correlation_matrix.csv files
    """
    search_pattern = os.path.join(base_dir, "**", "rdm_correlation_matrix.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} RDM correlation matrix CSV files:")
    for file_path in csv_files:
        print(f"  {file_path}")
    
    return csv_files


def load_rdm_matrices(csv_files):
    """
    Load all RDM correlation matrices from CSV files
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        tuple: (list of matrices, layer names)
    """
    matrices = []
    layer_names = None
    
    for csv_file in csv_files:
        try:
            # Read CSV with first column as index
            df = pd.read_csv(csv_file, index_col=0)
            
            # Store layer names from first file
            if layer_names is None:
                layer_names = df.columns.tolist()
            
            # Convert to numpy array and store
            matrix = df.values
            matrices.append(matrix)
            
            print(f"Loaded matrix from {csv_file}: shape {matrix.shape}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return matrices, layer_names


def compute_average_rdm(matrices):
    """
    Compute average RDM across all matrices
    
    Args:
        matrices: List of RDM correlation matrices
        
    Returns:
        Average RDM matrix
    """
    if not matrices:
        raise ValueError("No matrices provided for averaging")
    
    # Stack matrices and compute mean
    stacked_matrices = np.stack(matrices, axis=0)
    average_matrix = np.mean(stacked_matrices, axis=0)
    
    print(f"Computed average RDM from {len(matrices)} matrices")
    print(f"Average matrix shape: {average_matrix.shape}")
    
    return average_matrix


def compute_ttest_matrix(matrices):
    """
    Perform one-sample t-test for each position in the RDM matrices
    Tests against null hypothesis that correlation = 0
    
    Args:
        matrices: List of RDM correlation matrices
        
    Returns:
        tuple: (t-statistics matrix, p-values matrix)
    """
    if len(matrices) < 2:
        raise ValueError("Need at least 2 matrices for t-test")
    
    # Stack matrices
    stacked_matrices = np.stack(matrices, axis=0)
    n_matrices, n_layers, _ = stacked_matrices.shape
    
    # Initialize result matrices
    t_stats = np.zeros((n_layers, n_layers))
    p_values = np.zeros((n_layers, n_layers))
    
    # Perform t-test for each position
    for i in range(n_layers):
        for j in range(n_layers):
            values = stacked_matrices[:, i, j]
            t_stat, p_val = stats.ttest_1samp(values, 0.0)
            t_stats[i, j] = t_stat
            p_values[i, j] = p_val
    
    print(f"Computed t-test statistics from {n_matrices} matrices")
    print(f"T-statistics range: [{np.min(t_stats):.3f}, {np.max(t_stats):.3f}]")
    print(f"P-values range: [{np.min(p_values):.6f}, {np.max(p_values):.6f}]")
    
    return t_stats, p_values


def visualize_average_rdm(average_matrix, layer_names, output_dir):
    """
    Visualize average RDM correlation matrix
    
    Args:
        average_matrix: Average RDM correlation matrix
        layer_names: List of layer names
        output_dir: Output directory for saving plots
    """
    n_layers = len(layer_names)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot correlation matrix
    im = plt.imshow(average_matrix, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(im, label='Average Spearman Correlation')
    
    # Add correlation values as text
    for i in range(n_layers):
        for j in range(n_layers):
            text = plt.text(j, i, f'{average_matrix[i, j]:.2f}',
                           ha="center", va="center", 
                           color="black" if abs(average_matrix[i, j]) < 0.5 else "white",
                           fontsize=8)
    
    # Set labels and title
    plt.xticks(range(n_layers), layer_names, rotation=45, ha='right')
    plt.yticks(range(n_layers), layer_names)
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title('Average RDM Correlation Matrix\n(Mean Spearman correlation across all experiments)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_rdm_correlation_matrix.png'), 
               dpi=300, format="png", bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'average_rdm_correlation_matrix.eps'), 
               dpi=300, format="eps", bbox_inches='tight')
    plt.close()
    
    print(f"Average RDM visualization saved to {output_dir}")


def visualize_ttest_results(t_stats, p_values, layer_names, output_dir, alpha=0.05):
    """
    Visualize t-test results for RDM correlation matrices
    
    Args:
        t_stats: T-statistics matrix
        p_values: P-values matrix
        layer_names: List of layer names
        output_dir: Output directory for saving plots
        alpha: Significance level for highlighting
    """
    n_layers = len(layer_names)
    
    # Create t-statistics visualization
    plt.figure(figsize=(10, 8))
    
    # # Plot 1: T-statistics
    # plt.subplot(1, 2, 1)
    im1 = plt.imshow(t_stats, cmap='RdBu_r', aspect='equal', vmin=-2000, vmax=2000)
    plt.colorbar(im1, label='T-statistic')
    
    # Add t-statistic values as text
    for i in range(n_layers):
        for j in range(n_layers):
            color = "white" if abs(t_stats[i, j]) > 1000 else "black"
            plt.text(j, i, f'{t_stats[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=8)
    
    plt.xticks(range(n_layers), layer_names, rotation=45, ha='right')
    plt.yticks(range(n_layers), layer_names)
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title('T-statistics\n(One-sample t-test vs. 0)')
    
    # # Plot 2: Significance map
    # plt.subplot(1, 2, 2)
    
    # Create significance matrix (1 for significant, 0 for non-significant)
    sig_matrix = (p_values < alpha).astype(int)
    
    # im2 = plt.imshow(sig_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    # plt.colorbar(im2, label='Significant (p < 0.05)', ticks=[0, 1])
    
    # # Add p-values as text
    # for i in range(n_layers):
    #     for j in range(n_layers):
    #         p_val = p_values[i, j]
    #         if p_val < 0.001:
    #             text = "***"
    #         elif p_val < 0.01:
    #             text = "**"
    #         elif p_val < 0.05:
    #             text = "*"
    #         else:
    #             text = f'{p_val:.3f}'
            
    #         color = "white" if sig_matrix[i, j] else "black"
    #         plt.text(j, i, text, ha="center", va="center", color=color, fontsize=6)
    
    # plt.xticks(range(n_layers), layer_names, rotation=45, ha='right')
    # plt.yticks(range(n_layers), layer_names)
    # plt.xlabel('Layer')
    # plt.ylabel('Layer')
    # plt.title(f'Statistical Significance\n(p < {alpha})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rdm_ttest_results.png'), 
               dpi=300, format="png", bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'rdm_ttest_results.eps'), 
               dpi=300, format="eps", bbox_inches='tight')
    plt.close()
    
    print(f"T-test results visualization saved to {output_dir}")
    
    # Print summary statistics
    n_significant = np.sum(sig_matrix)
    total_comparisons = n_layers * n_layers
    print(f"Significant correlations: {n_significant}/{total_comparisons} ({100*n_significant/total_comparisons:.1f}%)")


def save_results_to_csv(average_matrix, t_stats, p_values, layer_names, output_dir):
    """
    Save analysis results to CSV files
    
    Args:
        average_matrix: Average RDM correlation matrix
        t_stats: T-statistics matrix
        p_values: P-values matrix
        layer_names: List of layer names
        output_dir: Output directory for saving CSV files
    """
    # Save average matrix
    avg_df = pd.DataFrame(average_matrix, index=layer_names, columns=layer_names)
    avg_df.to_csv(os.path.join(output_dir, 'average_rdm_correlation_matrix.csv'))
    
    # Save t-statistics
    t_df = pd.DataFrame(t_stats, index=layer_names, columns=layer_names)
    t_df.to_csv(os.path.join(output_dir, 'rdm_ttest_statistics.csv'))
    
    # Save p-values
    p_df = pd.DataFrame(p_values, index=layer_names, columns=layer_names)
    p_df.to_csv(os.path.join(output_dir, 'rdm_ttest_pvalues.csv'))
    
    print(f"Results saved to CSV files in {output_dir}")


def main():
    """
    Main function to perform RDM aggregation analysis
    """
    print("=== RDM Correlation Matrix Aggregation Analysis ===\n")
    
    # Set up output directory
    output_dir = os.path.join(project_root, "Results", "communication_semantic_analysis_results", "rdm_aggregation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all RDM CSV files
    csv_files = find_rdm_csv_files(os.path.join(project_root, "Results", "communication_semantic_analysis_results"))
    
    if not csv_files:
        print("No RDM correlation matrix CSV files found!")
        print("Please ensure that semantic analysis has been run and CSV files exist.")
        return
    
    # Load matrices
    matrices, layer_names = load_rdm_matrices(csv_files)
    
    if not matrices:
        print("No valid matrices could be loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(matrices)} RDM correlation matrices")
    print(f"Layer names: {layer_names}")
    
    # Compute average RDM
    print("\n--- Computing Average RDM ---")
    average_matrix = compute_average_rdm(matrices)
    
    # Perform t-tests
    if len(matrices) >= 2:
        print("\n--- Performing T-tests ---")
        t_stats, p_values = compute_ttest_matrix(matrices)
    else:
        print("\n--- Skipping T-tests (need at least 2 matrices) ---")
        t_stats = p_values = None
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    visualize_average_rdm(average_matrix, layer_names, output_dir)
    
    if t_stats is not None:
        visualize_ttest_results(t_stats, p_values, layer_names, output_dir)
    
    # Save results to CSV
    print("\n--- Saving Results ---")
    if t_stats is not None:
        save_results_to_csv(average_matrix, t_stats, p_values, layer_names, output_dir)
    else:
        # Save only average matrix
        avg_df = pd.DataFrame(average_matrix, index=layer_names, columns=layer_names)
        avg_df.to_csv(os.path.join(output_dir, 'average_rdm_correlation_matrix.csv'))
        print(f"Average RDM matrix saved to {output_dir}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"- average_rdm_correlation_matrix.png: Average RDM visualization")
    if t_stats is not None:
        print(f"- rdm_ttest_results.png: T-test results visualization")
        print(f"- CSV files: average matrix, t-statistics, and p-values")


if __name__ == "__main__":
    main()