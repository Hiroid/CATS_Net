#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract matching WordNet IDs between ImageNet classes and THINGS concepts.

This script compares WordNet IDs from two files:
1. /workspace/Deps/CustomFuctions/classes.py - ImageNet classes with WordNet IDs as keys
2. /data/THINGS/things_concepts.tsv - THINGS concepts with WordNet IDs in column 20

Returns a dictionary mapping WordNet IDs to their index positions in the ImageNet classes.
"""

import sys
import os
from collections import OrderedDict
from pathlib import Path

# Get script directory and project root
script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

# Add the path to import the classes module
sys.path.append(os.path.join(project_root, 'Deps', 'CustomFuctions'))

def load_imagenet_classes():
    """Load ImageNet classes from classes.py file."""
    try:
        from classes import IMAGENET2012_CLASSES
        return IMAGENET2012_CLASSES
    except ImportError as e:
        print(f"Error importing classes: {e}")
        return None

def load_things_wordnet_ids(tsv_path):
    """Load WordNet IDs from THINGS concepts TSV file."""
    wordnet_ids = set()
    
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            
            for line in f:
                columns = line.strip().split('\t')
                if len(columns) >= 20:  # WordNet ID is in column 20 (index 19)
                    wordnet_id = columns[19].strip()
                    if wordnet_id and wordnet_id != '#N/A':
                        wordnet_ids.add(wordnet_id)
    
    except FileNotFoundError:
        print(f"Error: File {tsv_path} not found")
        return set()
    except Exception as e:
        print(f"Error reading {tsv_path}: {e}")
        return set()
    
    return wordnet_ids

def find_matching_wordnet_ids():
    """Find matching WordNet IDs and return their positions in ImageNet classes."""
    # Load ImageNet classes
    imagenet_classes = load_imagenet_classes()
    if imagenet_classes is None:
        return {}
    
    # Load THINGS WordNet IDs
    things_wordnet_ids = load_things_wordnet_ids('/data/THINGS/things_concepts.tsv')
    if not things_wordnet_ids:
        return {}
    
    # Find matches and their positions
    matches = {}
    imagenet_wordnet_ids = set(imagenet_classes.keys())
    
    # Find intersection
    common_ids = imagenet_wordnet_ids.intersection(things_wordnet_ids)
    
    # Get positions (0-based index) in ImageNet classes
    imagenet_keys = list(imagenet_classes.keys())
    matching_pairs = []
    for wordnet_id in common_ids:
        index = imagenet_keys.index(wordnet_id)
        matching_pairs.append((wordnet_id, index))
    
    # Sort by index value to ensure ascending order
    matching_pairs.sort(key=lambda x: x[1])
    
    # Create ordered dictionary
    matches = OrderedDict()
    for wordnet_id, index in matching_pairs:
        matches[wordnet_id] = index
    
    return matches

def main():
    """Main function to execute the matching process."""
    print("Extracting matching WordNet IDs between ImageNet and THINGS datasets...")
    
    matches = find_matching_wordnet_ids()
    
    if matches:
        print(f"\nFound {len(matches)} matching WordNet IDs:")
        print("\nWordNet ID -> ImageNet Index:")
        print("-" * 40)
        
        # Sort by index for better readability
        sorted_matches = sorted(matches.items(), key=lambda x: x[1])
        
        for wordnet_id, index in sorted_matches:
            print(f"{wordnet_id} -> {index}")
        
        print(f"\nTotal matches: {len(matches)}")
        
        # Also return the dictionary for programmatic use
        return matches
    else:
        print("No matching WordNet IDs found.")
        return {}

if __name__ == "__main__":
    result = main()
    
    # Save results to a file for later use
    import json
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, 'Results', 'overlapping')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create final output with metadata
    output_data = {
        "metadata": {
            "description": "Matching WordNet IDs between ImageNet classes and THINGS concepts",
            "index_start": 0,
            "note": "Indices are 0-based from ImageNet classes dictionary, sorted in ascending order",
            "total_matches": len(result)
        },
        "matches": result
    }
    
    output_file = os.path.join(output_dir, 'wordnet_matches.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")