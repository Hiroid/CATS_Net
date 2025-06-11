import torch
import os
import sys
from pathlib import Path
import argparse

# Add the parent directory to sys.path
file_dir = Path().absolute()
workspace_dir = os.path.dirname(file_dir)
sys.path.append(workspace_dir)

from A01_ImageNet import utils, model, data
from torch.utils.data import DataLoader, SubsetRandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Evaluate SeaNet accuracy with optional random symbol set.')
parser.add_argument('--random_symbol_set', action='store_true',
                    help='Initialize symbol_set with random values.')
args = parser.parse_args()
# --- End Argument Parsing ---

all_seanet_acc_list = []
for net_idx in range(30):
    print(f"########## {net_idx} ##########")
    single_seanet_acc_list = []

    testset_imagenet = data.FeatureDataset("../Results/FeatureData/ImageNet1k_test_embeddings.pt", "../Results/FeatureData/ImageNet1k_test_indices.pt")

    net = model.sea_net(
        symbol_size = 20, 
        num_classes = 1000, 
        fix_fe = True, 
        fe_type = "resnet50",
        pretrain = True,
    )
    net.load_state_dict(torch.load(f"../Results/param/imagenet1k_ss20_fixfe_trail{21+net_idx}.pt"))
    net.to('cuda')
    
    # Conditionally randomize symbol set
    if args.random_symbol_set:
        print("Initializing symbol_set with random values.")
        net.symbol_set.data = torch.rand(net.symbol_set.shape).to('cuda')
    
    neti_test_acc = utils.evaluate_accuracy(testset_imagenet, net, use_feature = True)
    all_seanet_acc_list.append(neti_test_acc)
    
    # Determine filename based on random_symbol_set flag
    all_acc_filename_suffix = "random_" if args.random_symbol_set else ""
    all_acc_filename = f"../Results/accuracy_list/acc_list_{all_acc_filename_suffix}imagenet1k_ss20_fixfe_alltrails.pt"
    print(f"Saving all trail accuracies to: {all_acc_filename}")
    torch.save(all_seanet_acc_list, all_acc_filename)

    for idx in range(1000):
        print(f"---------- {idx} ----------")
        removed_idx_test = utils.get_indices(testset_imagenet, [idx])
        removed_sampler_test = SubsetRandomSampler(removed_idx_test)
        removed_testloader_imagenet = DataLoader(
            testset_imagenet, 
            batch_size = 512, 
            num_workers = 8, 
            sampler = removed_sampler_test
        )

        classi_test_acc = utils.evaluate_accuracy(removed_testloader_imagenet, net, use_feature = True, use_iter = True)
        single_seanet_acc_list.append(classi_test_acc)

        # Determine filename based on random_symbol_set flag
        single_acc_filename_suffix = "random_" if args.random_symbol_set else ""
        single_acc_filename = f"../Results/accuracy_list/acc_list_{single_acc_filename_suffix}imagenet1k_ss20_fixfe_trail{21+net_idx}.pt"
        print(f"Saving single trail accuracies to: {single_acc_filename}")
        torch.save(single_seanet_acc_list, single_acc_filename)
        