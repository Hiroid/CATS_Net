import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import matplotlib as mpl

# PDF font settings: Use TrueType fonts to prevent font conversion issues
mpl.rcParams['pdf.fonttype'] = 42

# Font configuration - Ensure proper font embedding in PDF
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 14

# Ensure complete font embedding in PDF
mpl.rcParams['pdf.use14corefonts'] = False
mpl.rcParams['pdf.compression'] = 6

def load_feature_names(filename='./lib/THINGS_features_name.xlsx'):
    """
    Load THINGS feature names from Excel file
    
    Args:
        filename: Path to Excel file containing feature names
        
    Returns:
        feature_names: Dictionary mapping feature ID to name {id: name}
    """
    print(f"Loading feature names from: {filename}")
    
    try:
        df = pd.read_excel(filename)
        print(f"Successfully loaded file: {filename}")
        print(f"Excel file shape: {df.shape}")
        
        # Create ID to name mapping dictionary
        feature_names = {}
        for _, row in df.iterrows():
            feature_names[int(row['ID'])] = str(row['name'])
        
        print(f"Loaded {len(feature_names)} feature names")
        return feature_names
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Using default feature names...")
        # Return default names if loading fails
        return {i: f"Feature {i}" for i in range(49)}

def load_correlation_results(filename='./things_cats_correlations_maxfeatures_full.json'):
    """
    Load correlation analysis results from JSON file
    
    Args:
        filename: Path to JSON results file
        
    Returns:
        results: Parsed results dictionary
    """
    print(f"Loading correlation results from: {filename}")
    
    # Try different file paths
    possible_paths = [
        filename,
        os.path.join('..', os.path.basename(filename)),
        os.path.join('..', 'lib', os.path.basename(filename))
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Successfully loaded file: {path}")
                return results
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
            
    raise FileNotFoundError(f"Cannot find file: {filename}")

def convert_results_keys(results):
    """
    Convert string keys in results dictionary to integer keys
    
    Args:
        results: Results dictionary with string keys
        
    Returns:
        converted_results: Results dictionary with integer keys
    """
    # Convert string keys to integers
    converted_results = {}
    for things_idx_str in results:
        things_idx = int(things_idx_str)
        converted_results[things_idx] = {}
        for x_idx_str in results[things_idx_str]:
            x_idx = int(x_idx_str)
            converted_results[things_idx][x_idx] = results[things_idx_str][x_idx_str]
    
    n_things_features = len(converted_results)
    n_x_positions = len(list(converted_results.values())[0]) if converted_results else 0
    
    print(f"Data dimensions: {n_things_features} THINGS features × {n_x_positions} models")
    print(f"THINGS feature index range: {min(converted_results.keys())} - {max(converted_results.keys())}")
    
    return converted_results

def create_rose_chart(things_feature_idx, feature_data, threshold=0.107, save_path=None, feature_name=None):
    """
    Create a rose chart for a single THINGS feature
    
    Args:
        things_feature_idx: THINGS feature index
        feature_data: Data for 30 models corresponding to this feature
        threshold: Threshold value, default 0.107
        save_path: Path to save figure, if None, figure will not be saved
        feature_name: Feature name, if None, index will be used
        
    Returns:
        fig, ax: matplotlib figure objects
    """
    # Validate input data
    if not feature_data:
        raise ValueError(f"Data for feature {things_feature_idx} is empty")
    
    # Extract rho values for 30 models
    n_models = len(feature_data)
    try:
        rho_values = [feature_data[x_idx]['rho'] for x_idx in range(n_models)]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Data format error for feature {things_feature_idx}: {e}")
    
    # Validate rho values
    if not rho_values or any(not isinstance(r, (int, float)) for r in rho_values):
        raise ValueError(f"Feature {things_feature_idx} contains invalid rho values")
    
    # Set angles: 30 models evenly distributed in 360 degrees
    angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    
    # Create polar plot with larger figure size
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    # Plot bars (petals)
    bars = ax.bar(angles, rho_values, width=2*np.pi/n_models*0.8, 
                  bottom=0, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    
    # Add gradient color effect to each petal
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add threshold ring at y=threshold
    ring_width = threshold * 0.02  # Ring width: 2% of threshold value
    
    # Add ring fill area
    theta_fill = np.linspace(0, 2*np.pi, 100)
    r_inner = np.full_like(theta_fill, threshold - ring_width/2)
    r_outer = np.full_like(theta_fill, threshold + ring_width/2)
    ax.fill_between(theta_fill, r_inner, r_outer, color='red', alpha=0.2, label='Threshold')
    
    # Set angle labels (model numbers)
    ax.set_thetagrids(np.degrees(angles), [f'M{i}' for i in range(n_models)])
    ax.tick_params(axis='x', labelsize=10, pad=0)
    
    # Set radial axis range
    max_rho = max(rho_values)
    ax.set_ylim(0, max(max_rho * 1.0, threshold * 1.0))
    
    # Set radial grid without labels
    radial_ticks = np.linspace(0, max(max_rho * 0.8, threshold * 1.0), 4)
    ax.yaxis.set_ticklabels([])
    ax.set_rgrids(radial_ticks[1:], alpha=0.6)
    
    # Set polar plot position
    ax.set_position([0, 0, 0.3, 0.3])  # [left, bottom, width, height]
    
    # Set title
    if feature_name:
        title = f'Feature {things_feature_idx}: {feature_name}\n'
    else:
        title = f'Feature {things_feature_idx}\n'
    
    ax.set_title(title, fontsize=18, linespacing=1)
    
    # Add statistics text
    mean_rho = np.mean(rho_values)
    max_rho_idx = np.argmax(rho_values)
    max_rho_val = rho_values[max_rho_idx]
    above_threshold_count = sum(1 for r in rho_values if r > threshold)
    
    stats_text = f'Mean correlation: {mean_rho:.3f}\n Above threshold: {above_threshold_count} models'
    ax.text(0.5, -0.1, stats_text, transform=ax.transAxes, fontsize=16,
            horizontalalignment='center', verticalalignment='top', 
            bbox=dict(facecolor='#FFFACD', alpha=0.6,
            edgecolor='gray', boxstyle='round,pad=0.2'),
            linespacing=1)
    
    # Save figure
    if save_path:
        try:
            # Ensure save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save as PDF with proper font embedding
            plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                       dpi=300, pad_inches=0.2,
                       metadata={'Creator': 'matplotlib', 'Producer': 'matplotlib'})
            print(f"Saved rose chart: {save_path}")
        except Exception as e:
            print(f"Failed to save figure {save_path}: {e}")
    
    return fig, ax

def plot_all_rose_charts(results, feature_names=None, threshold=0.107, output_dir='rose_charts'):
    """
    Plot rose charts for all THINGS features
    
    Args:
        results: Correlation analysis results dictionary
        feature_names: Feature names dictionary
        threshold: Threshold value
        output_dir: Output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    n_things_features = len(results)
    print(f"Starting to plot {n_things_features} THINGS feature rose charts...")
    
    # Get actual feature indices list
    feature_indices = sorted(results.keys())
    
    # Plot rose chart for each THINGS feature
    for i, things_idx in enumerate(feature_indices):
        print(f"Drawing THINGS feature {things_idx} ({i + 1}/{n_things_features})")
        
        # Get feature data
        feature_data = results[things_idx]
        
        # Get feature name
        feature_name = feature_names.get(things_idx) if feature_names else None
        
        # Create save path (PDF format)
        save_path = os.path.join(output_dir, f'things_feature_{things_idx:02d}_rose_chart.pdf')
        
        # Plot rose chart
        fig, ax = create_rose_chart(things_idx, feature_data, threshold, save_path, feature_name)
        
        # Always close figure to save memory
        plt.close(fig)
    
    print(f"All rose charts saved to {output_dir} directory")


def main():
    """
    Main function
    """
    print("=== THINGS Feature Rose Chart Generator ===")
    print("Function: Generate rose charts for each THINGS feature showing correlation coefficients of 30 models")
    print()
    
    # Set parameters
    threshold = 0.107  # Threshold value
    output_dir = 'rose_charts'  # Output directory
    
    try:
        # Load feature names
        feature_names = load_feature_names()
        
        # Load correlation analysis results
        results = load_correlation_results()
        
        # Convert key format
        results = convert_results_keys(results)
        
        # Plot all rose charts
        plot_all_rose_charts(results, feature_names, threshold, output_dir)
        
        print("\n=== Drawing Complete ===")
        print(f"All rose charts saved to {output_dir} directory")
        print("Each THINGS feature corresponds to one rose chart")
        print("Petals = 30 models, Petal height = correlation coefficient rho")
        print(f"Red ring area = threshold value {threshold}")
        
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
