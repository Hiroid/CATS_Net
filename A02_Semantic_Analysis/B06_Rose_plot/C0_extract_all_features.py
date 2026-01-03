"""
Unified Feature Extraction Script - For Research Reproducibility

This script extracts feature vectors from three different data sources:
1. Binder Features: Extract 65-dimensional features for matched ImageNet classes from CSV
2. Concept Vectors: Extract concept vectors from multiple .pt files (30 trails, 20-dim vectors for 1854 concepts each)
3. THINGS Vectors: Extract feature vectors from THINGS_Features.txt file

"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# ==================== Configuration Section ====================

class Config:
    """Configuration class: centralized management of all file paths and parameters"""
    
    # Input file paths
    WORDNET_MATCHES_JSON = 'lib/wordnet_matches.json'
    BINDER_CSV_FILE = 'lib/imgnet1000_Binder65.csv'
    THINGS_FEATURES_TXT = 'lib/THINGS_Features.txt'
    CONCEPT_VEC_DIR = 'lib/Concept_Vec'
    
    # Concept vector file range
    CONCEPT_TRAIL_START = 21
    CONCEPT_TRAIL_END = 50
    
    # Output file paths
    OUTPUT_DIR = 'SharedVec'
    OUTPUT_BINDER_CSV = 'Binder65_vectors.csv'
    OUTPUT_CONCEPT_CSV = 'CATS_vectors.csv'
    OUTPUT_THINGS_CSV = 'THINGS_vectors.csv'
    OUTPUT_CONCEPT_FULL_CSV = 'CATS_vectors_full.csv'  # CATS full version (CATS+THINGS match only)
    OUTPUT_THINGS_FULL_CSV = 'THINGS_vectors_full.csv'  # THINGS full version (CATS+THINGS match only)
    OUTPUT_INDEX_JSON = 'extraction_index_info.json'
    OUTPUT_INDEX_FULL_JSON = 'extraction_index_info_full.json'  # Index info for full version
    
    # Logging level
    LOG_LEVEL = logging.INFO

# ==================== Logging Setup ====================

def setup_logging(level=logging.INFO):
    """Configure logging output format"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging(Config.LOG_LEVEL)

# ==================== WordNet Matching Data Loading ====================

def load_wordnet_matches(json_file: str) -> Dict[str, Any]:
    """
    Load WordNet matching data
    
    Args:
        json_file: Path to wordnet_matches.json file
        
    Returns:
        dict: Complete data structure containing matches and metadata
    """
    logger.info(f"Loading WordNet matching data: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matches = data['matches']
    metadata = data.get('metadata', {})
    
    logger.info(f"Successfully loaded {len(matches)} matches")
    logger.info(f"Index type: {metadata.get('note', 'N/A')}")
    
    return data

def get_sorted_match_items(matches: Dict) -> List[Tuple[str, Dict]]:
    """
    Get match items in the original order from JSON file
    
    Args:
        matches: WordNet matching dictionary
        
    Returns:
        list: List of (wordnet_id, info) tuples in sorted order
    """
    return list(matches.items())

# ==================== Binder65 Feature Extraction ====================

def extract_binder_features(csv_file: str, matches: Dict) -> Tuple[np.ndarray, List[str], List[int], List[int], List[str]]:
    """
    Extract Binder65 features for common WordNet IDs between CSV and matches
    
    Args:
        csv_file: Path to Binder65 features CSV file (contains class_no column and 65-dim feature columns)
        matches: WordNet matching data dictionary
        
    Returns:
        tuple: (feature_matrix, wordnet_ids_list, things_indices_list, imagenet_indices_list, common_wordnet_ids)
            - feature_matrix shape: (N_common, 65)
            - All lists have the same length, maintaining original order from JSON file
            - common_wordnet_ids: List of common wordnet IDs between CSV and matches
    """
    logger.info("=" * 60)
    logger.info("Starting Binder65 Feature Extraction")
    logger.info("=" * 60)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    logger.info(f"CSV file shape: {df.shape}")
    logger.info(f"Feature dimensions: {len(df.columns) - 1} (excluding class_no column)")
    
    # Get all wordnet IDs from CSV
    csv_wordnet_ids = set(df['class_no'].values)
    logger.info(f"Total wordnet IDs in CSV: {len(csv_wordnet_ids)}")
    
    # Get all wordnet IDs from matches
    matches_wordnet_ids = set(matches.keys())
    logger.info(f"Total wordnet IDs in matches: {len(matches_wordnet_ids)}")
    
    # Find common wordnet IDs (intersection)
    common_wordnet_ids_set = csv_wordnet_ids & matches_wordnet_ids
    logger.info(f"Common wordnet IDs (intersection): {len(common_wordnet_ids_set)}")
    
    # Report non-overlapping IDs
    csv_only = csv_wordnet_ids - matches_wordnet_ids
    matches_only = matches_wordnet_ids - csv_wordnet_ids
    
    if csv_only:
        logger.info(f"WordNet IDs only in CSV: {len(csv_only)} (showing first 5)")
        for wid in list(csv_only)[:5]:
            logger.info(f"  - {wid}")
    
    if matches_only:
        logger.info(f"WordNet IDs only in matches: {len(matches_only)} (showing first 5)")
        for wid in list(matches_only)[:5]:
            logger.info(f"  - {wid}")
    
    # Create mapping from class_no (wordnet_id) to row index
    class_to_row = {class_no: idx for idx, class_no in enumerate(df['class_no'])}
    
    # Prepare storage for extracted data
    extracted_features = []
    wordnet_ids = []
    things_indices = []
    imagenet_indices = []
    
    # Iterate in original JSON file order, but only extract common IDs
    sorted_matches = get_sorted_match_items(matches)
    
    logger.info(f"\nExtracting features for common wordnet IDs...")
    logger.info(f"First 3 examples:")
    
    example_count = 0
    for wordnet_id, info in sorted_matches:
        # Only process if wordnet_id is in common set
        if wordnet_id not in common_wordnet_ids_set:
            continue
            
        imagenet_idx = info['imagenet_index']
        things_idx = info['things_index']
        
        # Extract features from CSV
        row_idx = class_to_row[wordnet_id]
        features = df.iloc[row_idx, 1:].values.astype(np.float32)
        
        extracted_features.append(features)
        wordnet_ids.append(wordnet_id)
        things_indices.append(things_idx)
        imagenet_indices.append(imagenet_idx)
        
        # Display first 3 examples
        if example_count < 3:
            logger.info(f"  #{example_count+1}: WordNet={wordnet_id}, ImageNet_index={imagenet_idx}, THINGS_index={things_idx}")
            example_count += 1
    
    # Convert to numpy array
    features_matrix = np.array(extracted_features, dtype=np.float32)
    
    logger.info(f"\nBinder65 Feature Extraction Completed:")
    logger.info(f"  - Successfully extracted: {len(extracted_features)}")
    logger.info(f"  - Feature matrix shape: {features_matrix.shape}")
    if len(things_indices) > 0:
        logger.info(f"  - THINGS index range: {min(things_indices)} - {max(things_indices)}")
    logger.info(f"  - Data type: {features_matrix.dtype}")
    
    return features_matrix, wordnet_ids, things_indices, imagenet_indices, wordnet_ids

# ==================== Concept Vector Extraction ====================

def extract_vectors_from_pt(pt_file: Path, indices: List[int]) -> torch.Tensor:
    """
    Extract vectors at specified indices from a single .pt file
    
    Args:
        pt_file: Path to .pt file
        indices: List of indices to extract
        
    Returns:
        torch.Tensor: Extracted vector matrix
    """
    # Load .pt file
    data = torch.load(pt_file, map_location='cpu')
    
    # Handle different data formats
    if isinstance(data, dict):
        if 'data' in data:
            vectors = data['data']
        else:
            keys = list(data.keys())
            logger.debug(f"Keys in file {pt_file.name}: {keys}")
            vectors = data[keys[0]]
    else:
        vectors = data
    
    # Ensure tensor format
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.tensor(vectors)
    
    logger.debug(f"Data shape in file {pt_file.name}: {vectors.shape}")
    
    # Extract vectors at specified indices
    extracted_vectors = vectors[indices]
    
    return extracted_vectors

def extract_concept_vectors(matches: Dict, concept_vec_dir: str, 
                           trail_start: int, trail_end: int,
                           valid_wordnet_ids: List[str] = None) -> np.ndarray:
    """
    Extract concept vectors from multiple .pt files
    
    Args:
        matches: WordNet matching data dictionary
        concept_vec_dir: Path to Concept_Vec directory
        trail_start: Starting trail number
        trail_end: Ending trail number (inclusive)
        valid_wordnet_ids: Optional list of valid wordnet IDs to filter by
        
    Returns:
        np.ndarray: Concept vector matrix with shape (n_trails, n_matches, vector_dim)
    """
    logger.info("=" * 60)
    logger.info("Starting Concept Vector Extraction")
    logger.info("=" * 60)
    
    # Extract ImageNet index list (in JSON order)
    sorted_matches = get_sorted_match_items(matches)
    
    # Filter by valid wordnet IDs if provided
    if valid_wordnet_ids is not None:
        valid_set = set(valid_wordnet_ids)
        sorted_matches = [(wid, info) for wid, info in sorted_matches if wid in valid_set]
        logger.info(f"Filtering by {len(valid_wordnet_ids)} valid wordnet IDs")
    
    imagenet_indices = [info['imagenet_index'] for _, info in sorted_matches]
    
    logger.info(f"Will extract vectors from {len(imagenet_indices)} indices")
    if len(imagenet_indices) > 0:
        logger.info(f"ImageNet index range: {min(imagenet_indices)} - {max(imagenet_indices)}")
    logger.info(f"Processing {trail_end - trail_start + 1} trail files (trail{trail_start} to trail{trail_end})")
    
    concept_dir = Path(concept_vec_dir)
    all_vectors = []
    
    # Iterate through all trail files
    for i in range(trail_start, trail_end + 1):
        pt_file = concept_dir / f'symbol_set_trail{i}.pt'
        
        if not pt_file.exists():
            logger.warning(f"File not found: {pt_file}")
            continue
            
        logger.info(f"Processing trail{i}...")
        
        try:
            vectors = extract_vectors_from_pt(pt_file, imagenet_indices)
            all_vectors.append(vectors)
            logger.info(f"  - Successfully extracted, shape: {vectors.shape}")
        except Exception as e:
            logger.error(f"  - Processing failed: {e}")
            raise
    
    # Stack all vectors
    if not all_vectors:
        raise ValueError("No vectors were successfully extracted")
    
    final_matrix = torch.stack(all_vectors, dim=0)
    logger.info(f"\nConcept Vector Extraction Completed:")
    logger.info(f"  - Matrix shape: {final_matrix.shape}")
    logger.info(f"  - Expected shape: ({trail_end - trail_start + 1}, {len(imagenet_indices)}, vector_dim)")
    
    # Convert to numpy array
    return final_matrix.numpy()

# ==================== THINGS Vector Extraction ====================

def extract_things_vectors(matches: Dict, features_file: str, 
                          valid_wordnet_ids: List[str] = None) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extract feature vectors from THINGS_Features.txt file
    
    Args:
        matches: WordNet matching data dictionary
        features_file: Path to THINGS_Features.txt file
        valid_wordnet_ids: Optional list of valid wordnet IDs to filter by
        
    Returns:
        tuple: (vector_matrix, wordnet_ids_list, things_indices_list)
    """
    logger.info("=" * 60)
    logger.info("Starting THINGS Vector Extraction")
    logger.info("=" * 60)
    
    # Get matching items in JSON order
    sorted_matches = get_sorted_match_items(matches)
    
    # Filter by valid wordnet IDs if provided
    if valid_wordnet_ids is not None:
        valid_set = set(valid_wordnet_ids)
        sorted_matches = [(wid, info) for wid, info in sorted_matches if wid in valid_set]
        logger.info(f"Filtering by {len(valid_wordnet_ids)} valid wordnet IDs")
    
    # Read THINGS features file
    logger.info(f"Reading file: {features_file}")
    with open(features_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger.info(f"Total lines in file: {len(lines)}")
    
    # Extract vectors at corresponding indices
    extracted_vectors = []
    extracted_wordnet_ids = []
    extracted_things_indices = []
    
    logger.info(f"First 3 matching items examples:")
    for i, (wordnet_id, info) in enumerate(sorted_matches):
        things_idx = info['things_index']
        
        # Check if index is valid
        if things_idx >= len(lines):
            logger.warning(f"THINGS index {things_idx} exceeds file range, skipping")
            continue
        
        # Parse vector
        vector_line = lines[things_idx].strip()
        vector_values = [float(x) for x in vector_line.split()]
        
        extracted_vectors.append(vector_values)
        extracted_wordnet_ids.append(wordnet_id)
        extracted_things_indices.append(things_idx)
        
        # Display first 3 examples
        if i < 3:
            logger.info(f"  #{i+1}: WordNet={wordnet_id}, THINGS_index={things_idx}, vector_dim={len(vector_values)}")
    
    # Convert to numpy array
    vectors_array = np.array(extracted_vectors, dtype=np.float32)
    
    logger.info(f"\nTHINGS Vector Extraction Completed:")
    logger.info(f"  - Successfully extracted: {len(extracted_vectors)}")
    logger.info(f"  - Vector matrix shape: {vectors_array.shape}")
    logger.info(f"  - Data type: {vectors_array.dtype}")
    
    return vectors_array, extracted_wordnet_ids, extracted_things_indices

# ==================== Save Results ====================

def save_results(binder_features: np.ndarray,
                concept_vectors: np.ndarray,
                things_vectors: np.ndarray,
                wordnet_ids: List[str],
                things_indices: List[int],
                imagenet_indices: List[int],
                concept_vectors_full: np.ndarray,
                things_vectors_full: np.ndarray,
                wordnet_ids_full: List[str],
                things_indices_full: List[int],
                imagenet_indices_full: List[int],
                output_dir: str,
                config: Config) -> None:
    """
    Save all extracted features and index information
    
    Args:
        binder_features: Binder feature matrix (matched with Binder)
        concept_vectors: Concept vector matrix (matched with Binder)
        things_vectors: THINGS vector matrix (matched with Binder)
        wordnet_ids: List of WordNet IDs (matched with Binder)
        things_indices: List of THINGS indices (matched with Binder)
        imagenet_indices: List of ImageNet indices (matched with Binder)
        concept_vectors_full: Concept vector matrix (CATS+THINGS only)
        things_vectors_full: THINGS vector matrix (CATS+THINGS only)
        wordnet_ids_full: List of WordNet IDs (CATS+THINGS only)
        things_indices_full: List of THINGS indices (CATS+THINGS only)
        imagenet_indices_full: List of ImageNet indices (CATS+THINGS only)
        output_dir: Output directory path
        config: Configuration object
    """
    logger.info("=" * 60)
    logger.info("Saving Extraction Results")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save ImageNet features as CSV (with labels)
    feature_columns = [f'dim_{i}' for i in range(binder_features.shape[1])]
    df_binder = pd.DataFrame(binder_features, columns=feature_columns)
    df_binder.insert(0, 'wordnet_id', wordnet_ids)
    df_binder.insert(1, 'things_index', things_indices)
    df_binder.insert(2, 'imagenet_index', imagenet_indices)
    
    binder_csv_path = output_path / config.OUTPUT_BINDER_CSV
    df_binder.to_csv(binder_csv_path, index=False)
    logger.info(f"✓ Binder features (CSV): {binder_csv_path}")
    
    # 2. Save Concept vectors as CSV
    # Reshape from (n_trails, n_matches, vector_dim) to 2D for CSV
    # Each row: wordnet_id, things_index, imagenet_index, trail1_dim0, trail1_dim1, ..., trail30_dimN
    n_trails, n_matches, vector_dim = concept_vectors.shape
    
    # Create column names for all trails and dimensions
    concept_columns = []
    for trail in range(n_trails):
        for dim in range(vector_dim):
            concept_columns.append(f'trail{trail+config.CONCEPT_TRAIL_START}_dim{dim}')
    
    # Reshape to 2D: (n_matches, n_trails * vector_dim)
    concept_2d = concept_vectors.transpose(1, 0, 2).reshape(n_matches, -1)
    
    df_concept = pd.DataFrame(concept_2d, columns=concept_columns)
    df_concept.insert(0, 'wordnet_id', wordnet_ids)
    df_concept.insert(1, 'things_index', things_indices)
    df_concept.insert(2, 'imagenet_index', imagenet_indices)
    
    concept_csv_path = output_path / config.OUTPUT_CONCEPT_CSV
    df_concept.to_csv(concept_csv_path, index=False)
    logger.info(f"✓ Concept vectors (CSV): {concept_csv_path}")
    
    # 3. Save THINGS vectors as CSV
    things_columns = [f'dim_{i}' for i in range(things_vectors.shape[1])]
    df_things = pd.DataFrame(things_vectors, columns=things_columns)
    df_things.insert(0, 'wordnet_id', wordnet_ids)
    df_things.insert(1, 'things_index', things_indices)
    df_things.insert(2, 'imagenet_index', imagenet_indices)
    
    things_csv_path = output_path / config.OUTPUT_THINGS_CSV
    df_things.to_csv(things_csv_path, index=False)
    logger.info(f"✓ THINGS vectors (CSV): {things_csv_path}")
    
    # 4. Save index information
    index_info = {
        'metadata': {
            'description': 'Feature extraction index information',
            'total_matches': len(wordnet_ids),
            'extraction_date': '2025',
            'note': 'All indices maintain original order from wordnet_matches.json file'
        },
        'shapes': {
            'binder_features': list(binder_features.shape),
            'concept_vectors': list(concept_vectors.shape),
            'things_vectors': list(things_vectors.shape)
        },
        'indices': {
            'wordnet_ids': wordnet_ids,
            'things_indices': things_indices,
            'imagenet_indices': imagenet_indices
        }
    }
    
    index_json_path = output_path / config.OUTPUT_INDEX_JSON
    with open(index_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_info, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Index information (JSON): {index_json_path}")
    
    # 5. Save full version CATS vectors (CATS+THINGS match only)
    logger.info("\n" + "=" * 60)
    logger.info("Saving Full Version (CATS+THINGS only)")
    logger.info("=" * 60)
    
    n_trails_full, n_matches_full, vector_dim_full = concept_vectors_full.shape
    
    # Create column names for all trails and dimensions
    concept_columns_full = []
    for trail in range(n_trails_full):
        for dim in range(vector_dim_full):
            concept_columns_full.append(f'trail{trail+config.CONCEPT_TRAIL_START}_dim{dim}')
    
    # Reshape to 2D: (n_matches, n_trails * vector_dim)
    concept_2d_full = concept_vectors_full.transpose(1, 0, 2).reshape(n_matches_full, -1)
    
    df_concept_full = pd.DataFrame(concept_2d_full, columns=concept_columns_full)
    df_concept_full.insert(0, 'wordnet_id', wordnet_ids_full)
    df_concept_full.insert(1, 'things_index', things_indices_full)
    df_concept_full.insert(2, 'imagenet_index', imagenet_indices_full)
    
    concept_full_csv_path = output_path / config.OUTPUT_CONCEPT_FULL_CSV
    df_concept_full.to_csv(concept_full_csv_path, index=False)
    logger.info(f"✓ CATS vectors full version (CSV): {concept_full_csv_path}")
    
    # 6. Save full version THINGS vectors
    things_columns_full = [f'dim_{i}' for i in range(things_vectors_full.shape[1])]
    df_things_full = pd.DataFrame(things_vectors_full, columns=things_columns_full)
    df_things_full.insert(0, 'wordnet_id', wordnet_ids_full)
    df_things_full.insert(1, 'things_index', things_indices_full)
    df_things_full.insert(2, 'imagenet_index', imagenet_indices_full)
    
    things_full_csv_path = output_path / config.OUTPUT_THINGS_FULL_CSV
    df_things_full.to_csv(things_full_csv_path, index=False)
    logger.info(f"✓ THINGS vectors full version (CSV): {things_full_csv_path}")
    
    # 7. Save index information for full version
    index_info_full = {
        'metadata': {
            'description': 'Full version feature extraction (CATS+THINGS match only, not matched with Binder)',
            'total_matches': len(wordnet_ids_full),
            'extraction_date': '2025',
            'note': 'All indices maintain original order from wordnet_matches.json file'
        },
        'shapes': {
            'concept_vectors': list(concept_vectors_full.shape),
            'things_vectors': list(things_vectors_full.shape)
        },
        'indices': {
            'wordnet_ids': wordnet_ids_full,
            'things_indices': things_indices_full,
            'imagenet_indices': imagenet_indices_full
        }
    }
    
    index_full_json_path = output_path / config.OUTPUT_INDEX_FULL_JSON
    with open(index_full_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_info_full, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Full version index information (JSON): {index_full_json_path}")
    
    # Print summary information
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)
    logger.info("Binder-matched version:")
    logger.info(f"  - Total matches: {len(wordnet_ids)}")
    logger.info(f"  - Binder features shape: {binder_features.shape}")
    logger.info(f"  - Concept vectors shape: {concept_vectors.shape}")
    logger.info(f"  - THINGS vectors shape: {things_vectors.shape}")
    logger.info("\nFull version (CATS+THINGS only):")
    logger.info(f"  - Total matches: {len(wordnet_ids_full)}")
    logger.info(f"  - Concept vectors shape: {concept_vectors_full.shape}")
    logger.info(f"  - THINGS vectors shape: {things_vectors_full.shape}")
    logger.info(f"\nAll files saved to: {output_path}")

# ==================== Main Function ====================

def main():
    """
    Main function: coordinates the entire feature extraction workflow
    """
    logger.info("\n" + "=" * 60)
    logger.info("Feature Extraction Script Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - WordNet matches file: {Config.WORDNET_MATCHES_JSON}")
    logger.info(f"  - Binder CSV: {Config.BINDER_CSV_FILE}")
    logger.info(f"  - THINGS features file: {Config.THINGS_FEATURES_TXT}")
    logger.info(f"  - Concept vectors directory: {Config.CONCEPT_VEC_DIR}")
    logger.info(f"  - Trail range: {Config.CONCEPT_TRAIL_START} - {Config.CONCEPT_TRAIL_END}")
    logger.info(f"  - Output directory: {Config.OUTPUT_DIR}")
    
    try:
        # 1. Load WordNet matching data
        wordnet_data = load_wordnet_matches(Config.WORDNET_MATCHES_JSON)
        matches = wordnet_data['matches']
        
        # 2. First, extract full version (CATS+THINGS only, without Binder constraint)
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Extracting Full Version (CATS+THINGS match only)")
        logger.info("=" * 60)
        
        # Extract full Concept vectors (no filtering)
        concept_vectors_full = extract_concept_vectors(
            matches, 
            Config.CONCEPT_VEC_DIR,
            Config.CONCEPT_TRAIL_START,
            Config.CONCEPT_TRAIL_END,
            valid_wordnet_ids=None  # No filtering
        )
        
        # Extract full THINGS vectors (no filtering)
        things_vectors_full, wordnet_ids_full, things_indices_full = \
            extract_things_vectors(matches, Config.THINGS_FEATURES_TXT,
                                 valid_wordnet_ids=None)  # No filtering
        
        # Get ImageNet indices for full version
        sorted_matches_full = get_sorted_match_items(matches)
        imagenet_indices_full = [info['imagenet_index'] for _, info in sorted_matches_full]
        
        # Verify full version consistency
        logger.info("\n" + "=" * 60)
        logger.info("Verifying full version data consistency...")
        logger.info("=" * 60)
        
        if concept_vectors_full.shape[1] != things_vectors_full.shape[0]:
            logger.error(f"Full version shape mismatch: Concept={concept_vectors_full.shape[1]}, THINGS={things_vectors_full.shape[0]}")
            raise ValueError("Full version feature count mismatch")
        
        logger.info("✓ Full version data synchronized!")
        logger.info(f"  - Concept vectors: {concept_vectors_full.shape}")
        logger.info(f"  - THINGS vectors: {things_vectors_full.shape}")
        logger.info(f"  - Total matches: {len(wordnet_ids_full)}")
        
        # 3. Extract Binder65 features (this will determine valid wordnet IDs)
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Extracting Binder-matched Version")
        logger.info("=" * 60)
        
        binder_features, wordnet_ids, things_indices, imagenet_indices, valid_wordnet_ids = \
            extract_binder_features(Config.BINDER_CSV_FILE, matches)
        
        logger.info(f"\nValid WordNet IDs for Binder synchronization: {len(valid_wordnet_ids)}")
        
        # 4. Extract Concept vectors (filtered by valid wordnet IDs)
        concept_vectors = extract_concept_vectors(
            matches, 
            Config.CONCEPT_VEC_DIR,
            Config.CONCEPT_TRAIL_START,
            Config.CONCEPT_TRAIL_END,
            valid_wordnet_ids=valid_wordnet_ids
        )
        
        # 5. Extract THINGS vectors (filtered by valid wordnet IDs)
        things_vectors, things_wordnet_ids, things_things_indices = \
            extract_things_vectors(matches, Config.THINGS_FEATURES_TXT,
                                 valid_wordnet_ids=valid_wordnet_ids)
        
        # Verify Binder-matched version consistency
        logger.info("\n" + "=" * 60)
        logger.info("Verifying Binder-matched version data consistency...")
        logger.info("=" * 60)
        
        if wordnet_ids != things_wordnet_ids:
            logger.error("WordNet IDs do not match between Binder and THINGS!")
            logger.error(f"Binder count: {len(wordnet_ids)}, THINGS count: {len(things_wordnet_ids)}")
            raise ValueError("WordNet IDs mismatch")
            
        if things_indices != things_things_indices:
            logger.error("THINGS indices do not match!")
            raise ValueError("THINGS indices mismatch")
        
        # Verify shapes match
        if binder_features.shape[0] != things_vectors.shape[0]:
            logger.error(f"Shape mismatch: Binder={binder_features.shape[0]}, THINGS={things_vectors.shape[0]}")
            raise ValueError("Feature count mismatch between Binder and THINGS")
            
        if concept_vectors.shape[1] != binder_features.shape[0]:
            logger.error(f"Shape mismatch: Concept={concept_vectors.shape[1]}, Binder={binder_features.shape[0]}")
            raise ValueError("Feature count mismatch between Concept and Binder")
        
        logger.info("✓ Binder-matched version data synchronized!")
        logger.info(f"  - Binder features: {binder_features.shape}")
        logger.info(f"  - Concept vectors: {concept_vectors.shape}")
        logger.info(f"  - THINGS vectors: {things_vectors.shape}")
        
        # 6. Save all results (both versions)
        save_results(
            binder_features,
            concept_vectors,
            things_vectors,
            wordnet_ids,
            things_indices,
            imagenet_indices,
            concept_vectors_full,
            things_vectors_full,
            wordnet_ids_full,
            things_indices_full,
            imagenet_indices_full,
            Config.OUTPUT_DIR,
            Config
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("All Extraction Tasks Completed Successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

