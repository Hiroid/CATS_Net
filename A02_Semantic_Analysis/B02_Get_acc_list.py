import torch
import os
import sys
import re
import glob
from pathlib import Path
import argparse
from collections import defaultdict

# Add the parent directory to sys.path
script_dir = Path(__file__).parent
project_root = script_dir
while not (project_root / 'Deps').exists() and project_root.parent != project_root:
    project_root = project_root.parent

# Add project root to the Python path if it's not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from A01_ImageNet import utils, model, data
from torch.utils.data import DataLoader, SubsetRandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_experiment_settings(exp_prefix):
    """Parse experiment settings from exp_prefix string
    
    Args:
        exp_prefix (str): Experiment prefix string
        
    Returns:
        dict: Dictionary containing parsed experiment settings
    """
    settings = {
        'dataset': 'imagenet1k',
        'symbol_size': 20,
        'fix_fe': True,
        'model_name': 'resnet50',
        'mlp_layers': 3,
        'hidden_dim': 100,
        'fix_ts': False,
        'fix_ts_ca': False,
        'fix_symbol_set': False,
        'joint_training': False,
        'symbol_init_type': 'random',
        'num_classes': 1000
    }
    
    # Parse dataset
    if 'imagenet1k' in exp_prefix:
        settings['dataset'] = 'imagenet1k'
        settings['num_classes'] = 1000
    elif 'imagenet100' in exp_prefix:
        settings['dataset'] = 'imagenet100'
        settings['num_classes'] = 100
    elif 'cifar10' in exp_prefix:
        settings['dataset'] = 'cifar10'
        settings['num_classes'] = 10
    elif 'cifar100' in exp_prefix:
        settings['dataset'] = 'cifar100'
        settings['num_classes'] = 100
    
    # Parse symbol size
    ss_match = re.search(r'ss(\d+)', exp_prefix)
    if ss_match:
        settings['symbol_size'] = int(ss_match.group(1))
    
    # Parse model name
    if 'resnet50' in exp_prefix:
        settings['model_name'] = 'resnet50'
    elif 'resnet18' in exp_prefix:
        settings['model_name'] = 'resnet18'
    elif 'vit_b_16' in exp_prefix:
        settings['model_name'] = 'vit_b_16'
    
    # Parse MLP layers
    mlp_match = re.search(r'mlp(\d+)', exp_prefix)
    if mlp_match:
        settings['mlp_layers'] = int(mlp_match.group(1))
    
    # Parse hidden dimension
    hidden_match = re.search(r'hidden(\d+)', exp_prefix)
    if hidden_match:
        settings['hidden_dim'] = int(hidden_match.group(1))
    
    # Parse fix flags
    settings['fix_fe'] = 'fixfe' in exp_prefix
    settings['fix_ts'] = 'fixtsTrue' in exp_prefix
    settings['fix_ts_ca'] = 'fixtscaTrue' in exp_prefix
    settings['fix_symbol_set'] = 'fixsymbolTrue' in exp_prefix
    settings['joint_training'] = 'jointTrue' in exp_prefix
    
    # Parse initialization type
    if 'initrandom' in exp_prefix:
        settings['symbol_init_type'] = 'random'
    elif 'initone_hot' in exp_prefix:
        settings['symbol_init_type'] = 'one_hot'
    elif 'initword2vec' in exp_prefix:
        settings['symbol_init_type'] = 'word2vec'
    
    return settings

def find_experiment_files(exp_prefix, log_dir, param_dir):
    """Find all experiment files matching the given prefix
    
    Args:
        exp_prefix (str): Experiment prefix
        log_dir (str): Log directory path
        param_dir (str): Parameter directory path
        
    Returns:
        tuple: (log_files, param_files) - lists of matching files
    """
    log_files = []
    param_files = []
    
    # Search for log files
    log_pattern = os.path.join(log_dir, f"{exp_prefix}*.log")
    log_files = glob.glob(log_pattern)
    
    # Search for parameter files
    param_pattern = os.path.join(param_dir, f"{exp_prefix}*.pt")
    param_files = glob.glob(param_pattern)
    
    return log_files, param_files

def extract_trial_info(filename, exp_prefix):
    """Extract trial information from filename
    
    Args:
        filename (str): Full filename
        exp_prefix (str): Experiment prefix
        
    Returns:
        str: Trial identifier (e.g., 'trail21', '20250911080447')
    """
    basename = os.path.basename(filename)
    # Remove exp_prefix and file extension
    suffix = basename.replace(exp_prefix, '').replace('.log', '').replace('.pt', '')
    
    # Remove leading underscore if present
    if suffix.startswith('_'):
        suffix = suffix[1:]
    
    return suffix if suffix else 'default'

def get_feature_dataset_path(settings):
    """Get the appropriate feature dataset path based on model type
    
    Args:
        settings (dict): Experiment settings
        
    Returns:
        tuple: (embeddings_path, indices_path)
    """
    model_name = settings['model_name']
    dataset = settings['dataset']
    
    embeddings_file = f"{dataset}_val_{model_name}_embeddings.pt"
    indices_file = f"{dataset}_val_{model_name}_indices.pt"
    
    embeddings_path = os.path.join(project_root, "Results", "FeatureData", embeddings_file)
    indices_path = os.path.join(project_root, "Results", "FeatureData", indices_file)
    
    return embeddings_path, indices_path

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Evaluate CATSNet accuracy with optional random symbol set.')
parser.add_argument('--random_symbol_set', action='store_true',
                    help='Initialize symbol_set with random values.')
parser.add_argument('--exp_prefix', type=str, default='imagenet1k_ss10_fixfe_resnet50_mlp3_hidden100_initrandom',
                    help='Prefix for experiment log.')
parser.add_argument('--max_trials', type=int, default=None,
                    help='Maximum number of trials to process (None for all available).')
parser.add_argument('--verbose', action='store_true',
                    help='Print detailed information during processing.')
parser.add_argument('--per_class_accuracy', action='store_true',
                    help='Calculate per-class accuracy for each trial (time-consuming).')

args = parser.parse_args()
# --- End Argument Parsing ---

def main():
    print(f"Processing experiments with prefix: {args.exp_prefix}")
    
    # Parse experiment settings from prefix
    settings = parse_experiment_settings(args.exp_prefix)
    if args.verbose:
        print("Detected experiment settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # Find all experiment files
    log_dir = os.path.join(project_root, "Results", "log")
    param_dir = os.path.join(project_root, "Results", "param")
    
    log_files, param_files = find_experiment_files(args.exp_prefix, log_dir, param_dir)
    
    if args.verbose:
        print(f"Found {len(log_files)} log files and {len(param_files)} parameter files")
    
    # Group files by trial
    trials = defaultdict(dict)
    
    # Process log files
    for log_file in log_files:
        trial_id = extract_trial_info(log_file, args.exp_prefix)
        trials[trial_id]['log'] = log_file
    
    # Process parameter files
    for param_file in param_files:
        trial_id = extract_trial_info(param_file, args.exp_prefix)
        trials[trial_id]['param'] = param_file
    
    # Filter trials that have both log and param files
    valid_trials = {k: v for k, v in trials.items() if 'param' in v}
    
    # Sort trials by numeric suffix to ensure consistent processing order
    def extract_numeric_suffix(trial_id):
        """Extract numeric suffix from trial_id for sorting"""
        import re
        match = re.search(r'(\d+)$', trial_id)
        return int(match.group(1)) if match else 0
    
    # Sort trial keys by numeric suffix
    sorted_trial_keys = sorted(valid_trials.keys(), key=extract_numeric_suffix)
    valid_trials = {k: valid_trials[k] for k in sorted_trial_keys}
    
    if not valid_trials:
        print("No valid trials found (trials with parameter files)")
        return
    
    print(f"Found {len(valid_trials)} valid trials: {list(valid_trials.keys())}")
    print(f"Trials are now sorted by numeric suffix: {list(valid_trials.keys())}")
    
    # Limit trials if specified
    if args.max_trials:
        trial_keys = list(valid_trials.keys())[:args.max_trials]
        valid_trials = {k: valid_trials[k] for k in trial_keys}
        print(f"Processing first {len(valid_trials)} trials")
    
    # Get feature dataset paths
    embeddings_path, indices_path = get_feature_dataset_path(settings)
    
    # Check if feature files exist
    if not os.path.exists(embeddings_path) or not os.path.exists(indices_path):
        print(f"Warning: Feature files not found for model '{settings['model_name']}':")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Indices: {indices_path}")
        
        # Suggest how to generate the missing files
        print(f"\nTo generate the required feature files, run:")
        print(f"cd /workspace/A02_Semantic_Analysis")
        print(f"python B01_Get_FeatureData.py --model_name {settings['model_name']} --split val --data_root /path/to/your/imagenet")
        print(f"\nSkipping analysis for this model. Please generate the feature files first.")
        return
    
    # Load test dataset
    testset_imagenet = data.FeatureDataset(embeddings_path, indices_path)
    
    all_catsnet_acc_list = []
    
    for trial_idx, (trial_id, trial_files) in enumerate(valid_trials.items()):
        print(f"########## Processing trial {trial_id} ({trial_idx + 1}/{len(valid_trials)}) ##########")
        
        single_catsnet_acc_list = []
        
        # Create network with detected settings
        net = model.cats_net(
            symbol_size=settings['symbol_size'],
            mlp_layers=settings['mlp_layers'],
            hidden_dim=settings['hidden_dim'],
            num_classes=settings['num_classes'],
            fix_fe=settings['fix_fe'],
            fe_type=settings['model_name'],
            pretrain=True,
        )
        
        # Load model parameters
        try:
            net.load_state_dict(torch.load(trial_files['param']))
            print(f"Loaded parameters from: {trial_files['param']}")
        except Exception as e:
            print(f"Error loading parameters for trial {trial_id}: {e}")
            continue
        
        net.to('cuda')
        
        # Conditionally randomize symbol set
        if args.random_symbol_set:
            print("Initializing symbol_set with random values.")
            net.symbol_set.data = torch.rand(net.symbol_set.shape).to('cuda')
        
        # Evaluate overall accuracy
        neti_test_acc = utils.evaluate_accuracy(testset_imagenet, net, use_feature=True)
        all_catsnet_acc_list.append(neti_test_acc)
        print(f"Overall accuracy: {neti_test_acc:.4f}")
        
        # Save overall accuracies
        all_acc_filename_suffix = "random_" if args.random_symbol_set else ""
        all_acc_filename = os.path.join(
            project_root, "Results", "accuracy_list", 
            f"acc_list_{all_acc_filename_suffix}{args.exp_prefix}_alltrails.pt"
        )
        torch.save(all_catsnet_acc_list, all_acc_filename)
        if args.verbose:
            print(f"Saved all trial accuracies to: {all_acc_filename}")
        
        # Evaluate per-class accuracy (only if requested)
        if args.per_class_accuracy:
            print(f"Computing per-class accuracy for trial {trial_id}...")
            for idx in range(settings['num_classes']):
                if args.verbose and idx % 100 == 0:
                    print(f"---------- Processing class {idx}/{settings['num_classes']} ----------")
                
                removed_idx_test = utils.get_indices(testset_imagenet, [idx])
                removed_sampler_test = SubsetRandomSampler(removed_idx_test)
                removed_testloader_imagenet = DataLoader(
                    testset_imagenet,
                    batch_size=128,
                    num_workers=8,
                    sampler=removed_sampler_test
                )
                
                classi_test_acc = utils.evaluate_accuracy(
                    removed_testloader_imagenet, net, use_feature=True, use_iter=True
                )
                single_catsnet_acc_list.append(classi_test_acc)
            
            # Save single trial accuracies
            single_acc_filename_suffix = "random_" if args.random_symbol_set else ""
            single_acc_filename = os.path.join(
                project_root, "Results", "accuracy_list",
                f"acc_list_{single_acc_filename_suffix}{args.exp_prefix}_{trial_id}.pt"
            )
            torch.save(single_catsnet_acc_list, single_acc_filename)
            print(f"Completed trial {trial_id} with per-class accuracy. Saved to: {single_acc_filename}")
        else:
            print(f"Completed trial {trial_id} (skipped per-class accuracy calculation)")

if __name__ == "__main__":
    main()
        