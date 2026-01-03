import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import json
import time
import os

def load_data():
    """
    Load CATS and THINGS vector data (full version)
    """
    print("Loading data (full version)...")
    
    # Use full version CSV files
    shared_vec_path = 'SharedVec'
    cats_file = os.path.join(shared_vec_path, 'CATS_vectors_full.csv')
    things_file = os.path.join(shared_vec_path, 'THINGS_vectors_full.csv')
    
    # Check if files exist
    if not os.path.exists(cats_file):
        raise FileNotFoundError(f"CATS full version file not found: {cats_file}")
    if not os.path.exists(things_file):
        raise FileNotFoundError(f"THINGS full version file not found: {things_file}")
    
    # Read CSV files
    df_cats = pd.read_csv(cats_file)
    df_things = pd.read_csv(things_file)
    
    print(f"CATS shape: {df_cats.shape}, THINGS shape: {df_things.shape}")
    
    # Extract feature data (skip first 3 columns: wordnet_id, things_index, imagenet_index)
    cats_features = df_cats.iloc[:, 3:].values  # shape: (n_matches, n_trails * vector_dim)
    things_features = df_things.iloc[:, 3:].values  # shape: (n_matches, things_vector_dim)
    
    # Reshape CATS data from 2D to 3D
    # CSV columns: trail21_dim0, trail21_dim1, ..., trail50_dim19
    # 30 trails (21-50), each trail has 20 dimensions
    n_trails = 30
    vector_dim = 20
    n_matches = cats_features.shape[0]
    
    # Reshape: (n_matches, n_trails * vector_dim) -> (n_matches, n_trails, vector_dim)
    cats_3d = cats_features.reshape(n_matches, n_trails, vector_dim)
    
    # Transpose to (n_trails, vector_dim, n_matches) to match expected format
    cats_vectors = cats_3d.transpose(1, 2, 0)
    
    # Keep THINGS data in (n_matches, features) format for correlation computation
    things_vectors = things_features
    
    print(f"Reshaped data - CATS: {cats_vectors.shape}, THINGS: {things_vectors.shape}")
    
    return cats_vectors, things_vectors

def compute_things_features_correlations(cats_vectors, things_vectors):
    """
    Compute correlations between each THINGS feature and all CATS (x,y) positions.
    For each THINGS feature, find the y position with maximum correlation in each x dimension.
    
    Args:
        cats_vectors: CATS data with shape (n_x, n_y, n_data)
        things_vectors: THINGS data with shape (n_data, n_features)
        
    Returns:
        Dictionary with structure: result[things_feature_idx][x_idx] = {'y': best_y, 'rho': correlation, 'p': p_value}
    """
    n_x, n_y, n_data = cats_vectors.shape
    n_things_data, n_things_features = things_vectors.shape
    
    print(f"Computing correlations...")
    print(f"CATS: {n_x} trails × {n_y} dims × {n_data} samples")
    print(f"THINGS: {n_things_data} samples × {n_things_features} features")
    print(f"Total correlations to compute: {n_things_features * n_x * n_y}")
    
    # Check dimension match
    if n_data != n_things_data:
        raise ValueError(f"Data size mismatch: CATS ({n_data}) vs THINGS ({n_things_data})")
    
    results = {}
    start_time = time.time()
    
    # Iterate through each THINGS feature
    for things_feature_idx in range(n_things_features):
        if (things_feature_idx + 1) % 10 == 0:
            print(f"Processing THINGS feature {things_feature_idx + 1}/{n_things_features}")
        
        things_feature_data = things_vectors[:, things_feature_idx]
        results[things_feature_idx] = {}
        
        # Iterate through each x dimension (trail)
        for x_idx in range(n_x):
            max_correlation = -2  # Initialize below valid range [-1, 1]
            best_y = -1
            best_p_value = 1.0
            
            # Find best y position in current x dimension
            for y_idx in range(n_y):
                cats_position_data = cats_vectors[x_idx, y_idx, :]
                
                try:
                    correlation, p_value = pearsonr(cats_position_data, things_feature_data)
                    
                    if not np.isnan(correlation) and correlation > max_correlation:
                        max_correlation = correlation
                        best_y = y_idx
                        best_p_value = p_value
                        
                except Exception:
                    continue
            
            # Store best result for current THINGS feature and x position
            results[things_feature_idx][x_idx] = {
                'y': best_y,
                'rho': max_correlation,
                'p': best_p_value
            }
    
    elapsed_time = time.time() - start_time
    print(f"Computation completed in {elapsed_time:.2f} seconds")
    
    return results

def save_results(results):
    """
    Save results to JSON file
    """
    output_file = 'things_cats_correlations_maxfeatures_full.json'
    print(f"\nSaving results to {output_file}")
    print(f"Result dimensions: {len(results)} THINGS features × {len(results[0])} x positions")
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for things_idx in results:
        json_results[str(things_idx)] = {}
        for x_idx in results[things_idx]:
            json_results[str(things_idx)][str(x_idx)] = {
                'y': int(results[things_idx][x_idx]['y']),
                'rho': float(results[things_idx][x_idx]['rho']),
                'p': float(results[things_idx][x_idx]['p'])
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    # Compute and print statistics
    all_correlations = []
    all_p_values = []
    
    for things_idx in results:
        for x_idx in results[things_idx]:
            all_correlations.append(results[things_idx][x_idx]['rho'])
            all_p_values.append(results[things_idx][x_idx]['p'])
    
    print("\nStatistics:")
    print(f"Total correlations: {len(all_correlations)}")
    print(f"Max correlation: {np.max(all_correlations):.4f}")
    print(f"Min correlation: {np.min(all_correlations):.4f}")
    print(f"Mean correlation: {np.mean(all_correlations):.4f}")
    print(f"Std correlation: {np.std(all_correlations):.4f}")
    print(f"Mean p-value: {np.mean(all_p_values):.4f}")
    
    # Find position of maximum correlation
    max_corr = np.max(all_correlations)
    for things_idx in results:
        for x_idx in results[things_idx]:
            if results[things_idx][x_idx]['rho'] == max_corr:
                print(f"\nMax correlation location:")
                print(f"  THINGS feature: {things_idx}, x position: {x_idx}, y position: {results[things_idx][x_idx]['y']}")
                print(f"  rho: {results[things_idx][x_idx]['rho']:.4f}, p: {results[things_idx][x_idx]['p']:.4f}")
                break
        else:
            continue
        break
    
    return results

def main():
    """
    Main function to compute THINGS-CATS feature correlations
    """
    print("=== THINGS-CATS Feature Correlation Analysis (Full Version) ===")
    print("Data files: CATS_vectors_full.csv and THINGS_vectors_full.csv")
    print("Goal: For each THINGS feature, find y position with max correlation in each x dimension\n")
    
    # Load data
    cats_vectors, things_vectors = load_data()
    
    # Compute correlations
    results = compute_things_features_correlations(cats_vectors, things_vectors)
    
    # Save results
    save_results(results)
    
    print("\n=== Analysis Complete ===")
    print("Results saved to: things_cats_correlations_maxfeatures_full.json")
    print("Result structure: result[things_feature_idx][x_idx] = {'y': best_y, 'rho': correlation, 'p': p_value}")

if __name__ == "__main__":
    main()
