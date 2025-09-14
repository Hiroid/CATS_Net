#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract symbol_set from trail21-50 model files and save them to concept_vectors folder.

This script reads PyTorch model files from /workspace/Results/param/ directory,
extracts the 'symbol_set' tensor from each file, and saves them individually
to /workspace/Results/concept_vectors/ directory.
"""

import torch
import os
from pathlib import Path

# Get script directory and project root
script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

def extract_symbol_sets():
    """Extract symbol_set from trail21-50 model files."""
    
    # Define paths
    param_dir = os.path.join(project_root, 'Results', 'param')
    output_dir = os.path.join(project_root, 'Results', 'concept_vectors')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process trail21 to trail50
    extracted_count = 0
    failed_files = []
    
    for trail_num in range(21, 51):
        input_file = os.path.join(param_dir, f'imagenet1k_ss20_fixfe_trail{trail_num}.pt')
        output_file = os.path.join(output_dir, f'symbol_set_trail{trail_num}.pt')
        
        try:
            # Load the model file
            print(f"Processing trail{trail_num}...")
            model_data = torch.load(input_file, map_location='cpu')
            
            # Check if it's a dictionary and contains symbol_set
            if not isinstance(model_data, dict):
                print(f"Warning: {input_file} is not a dictionary")
                failed_files.append(f"trail{trail_num}")
                continue
                
            if 'symbol_set' not in model_data:
                print(f"Warning: {input_file} does not contain 'symbol_set'")
                failed_files.append(f"trail{trail_num}")
                continue
            
            # Extract symbol_set
            symbol_set = model_data['symbol_set']
            print(f"  Symbol_set shape: {symbol_set.shape}")
            print(f"  Symbol_set type: {type(symbol_set)}")
            
            # Save symbol_set
            torch.save(symbol_set, output_file)
            print(f"  Saved to: {output_file}")
            extracted_count += 1
            
        except FileNotFoundError:
            print(f"Error: File {input_file} not found")
            failed_files.append(f"trail{trail_num}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            failed_files.append(f"trail{trail_num}")
    
    # Summary
    print(f"\n=== Extraction Summary ===")
    print(f"Successfully extracted: {extracted_count}/30 files")
    print(f"Output directory: {output_dir}")
    
    if failed_files:
        print(f"Failed files: {', '.join(failed_files)}")
    else:
        print("All files processed successfully!")
    
    return extracted_count, failed_files

def verify_extracted_files():
    """Verify the extracted symbol_set files."""
    
    output_dir = os.path.join(project_root, 'Results', 'concept_vectors')
    
    print(f"\n=== Verification ===")
    print(f"Checking files in: {output_dir}")
    
    for trail_num in range(21, 51):
        output_file = os.path.join(output_dir, f'symbol_set_trail{trail_num}.pt')
        
        if os.path.exists(output_file):
            try:
                symbol_set = torch.load(output_file, map_location='cpu')
                print(f"trail{trail_num}: ✓ Shape: {symbol_set.shape}, Type: {type(symbol_set)}")
            except Exception as e:
                print(f"trail{trail_num}: ✗ Error loading: {e}")
        else:
            print(f"trail{trail_num}: ✗ File not found")

def main():
    """Main function."""
    print("Starting symbol_set extraction from trail21-50...")
    print(f"Project root: {project_root}")
    
    # Extract symbol_sets
    extracted_count, failed_files = extract_symbol_sets()
    
    # Verify extracted files
    verify_extracted_files()
    
    print(f"\nExtraction completed!")
    return extracted_count, failed_files

if __name__ == "__main__":
    main()