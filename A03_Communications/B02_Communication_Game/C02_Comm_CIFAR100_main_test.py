# add the path of custom functions
import sys
from pathlib import Path
import os
script_dir = Path(__file__).parent
project_root = script_dir
while not (project_root / 'Deps').exists() and project_root.parent != project_root:
    project_root = project_root.parent

# Add project root to the Python path if it's not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

sys.path.append(os.path.join(project_root, "Deps", "CustomFuctions"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
import random
import os
import argparse
import scipy.io as io
from models import *
import scipy.stats as stats

import Translators, AccracyTest, CATSnet,  Utiliz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameters
parser = argparse.ArgumentParser(description='parameters setting')
parser.add_argument('--device', type=str, default='cuda', help='device type: cuda or cpu')
parser.add_argument('--worker', type=int, default=1, help='worker for data loader')
parser.add_argument('--ct_batch_size_train', type=int, default=96, help='batch size of symbols in D_99 when training TI module')
parser.add_argument('--batch_size_test', type=int, default=100, help='batch size for testing dataset')
parser.add_argument('--num_class', type=int, default=100, help='number of total classes')
# parser.add_argument('--class_id_unaligned', type=int, nargs='+', default=list(range(100)), help='the IDs of unaligned classes during the TI training')
parser.add_argument('--class_id_unaligned', type=list, default=list(range(100)), help='the IDs of unaligned classes during the TI training')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--end_epoch', type=int, default=200, help='eppch number to end with')
parser.add_argument('--context_dim', type=int, default=20, help='context dimension')
parser.add_argument('--path_symbol_speaker_train', type=str, default='./Symbols_of_Speaker/',help='diretory for the symbols of the speaker agent')
parser.add_argument('--path_symbol_speaker_test', type=str, default='./Testing_Symbols_of_Speaker/',help='directory for the testing sysmbols ("command") of D1 from the speaker agent')
parser.add_argument('--train_round_list', type=list, default=list(range(96)), help='round IDs to train the TI module')
parser.add_argument('--test_round_list', type=list, default=[95], help='round IDs to test the TI module')
parser.add_argument('--listener_symbol_path', type=str, default='./Symbol_and_Model_of_Listener/contexts',help='file path to store symbols of listener agent')
parser.add_argument('--listener_model_path', type=str, default='./Symbol_and_Model_of_Listener/checkpoint',help='file path to store the model of listener agent')
parser.add_argument('--listener_symbols_saveTimePoint', type=int, default=1999, help='the contexts at which time point saved are used as the targets')
parser.add_argument('--TInet_hidden_layers', type=int, default=10, help='number of hidden layers in the TI module')
parser.add_argument('--TInet_hidden_neurons', type=int, default=500, help='number of neurons contained in each layer in the TI module')
parser.add_argument('--TInet_p', type=float, default=0.3, help='dropout probility for the TI module')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for TI module')
parser.add_argument('--lr_sche_gamma', type=float, default=0.5, help='the decaying factor of the learning rate')
parser.add_argument('--lr_sche_steps', type=int, default=10, help='the decaying period of the learning rate')
args = parser.parse_args()

def symbol_translation_test(TInet_loaded, test_id_single):
    """Symbol translation function for testing phase"""
    TInet_loaded.eval()
    outputs = []
    with torch.no_grad():
        for sp_order in args.test_round_list:
            inputs = io.loadmat(args.path_symbol_speaker_test + 'context_ep_1999.mat')
            inputs = torch.from_numpy(inputs['context_0_1999'][test_id_single]).to(args.device)
            inputs = inputs.unsqueeze(0)
            output = TInet_loaded(inputs)
            outputs.append(output)
    return outputs

path = '.' + '/2025-06-02'
if not os.path.isdir(path):
    os.makedirs(path)
save_path_cp = path + '/checkpoint'
save_path_ct = path + '/contexts'

# load the pretrained models. pretrained_cdp_cnn was not used in the task.
pretrained_classifier_cnn = models.resnet18(weights=None)
pretrained_classifier_cnn.load_state_dict(
    torch.load(
        os.path.join(project_root, "Deps", "pretrained_fe", "resnet18-f37072fd.pth")
    )
)
pretrained_classifier_cnn.fc = nn.Identity()


print("\n" + "="*80)
print("TESTING PHASE: Loading trained TInet models and evaluating accuracy")
print("="*80)

# Initialize evaluation log
eval_log = open(path + '/evaluation_results.log', mode='w', encoding='utf-8')
print('Testing Phase Results', file=eval_log)
print('test_class_id, round, loss_pos, acc_pos, loss_neg, acc_neg, acc_all', file=eval_log)
eval_log.close()

# Store all results for final summary
all_test_results = []
overall_stats = {'acc_pos_sum': 0, 'acc_neg_sum': 0, 'count': 0}
final_acc_list = []

for test_id in args.class_id_unaligned:
    print(f"Testing model for class ID: {test_id}")
    
    # Check if trained model exists
    model_path = save_path_cp + '/TInet_testid_%d.pth' % (test_id)
    if not os.path.exists(model_path):
        print(f"Warning: No trained model found for test_id {test_id} at {model_path}")
        continue
    
    # Load the trained TInet model
    TInet_test = Translators.TImodule(context_dim=args.context_dim, num_hidden_layer=args.TInet_hidden_layers,
                                    num_hidden_neuron=args.TInet_hidden_neurons, dropout_p=args.TInet_p).to(args.device)
    
    # Load saved state
    saved_state = torch.load(model_path, map_location=args.device)
    TInet_test.load_state_dict(saved_state['net'])
    
    # Load corresponding CATSnet model
    sea_net_test = CATSnet.Net2(
        my_pretrained_classifier=pretrained_classifier_cnn,
        context_dim=args.context_dim
    ).to(args.device)
    model_ckpt = torch.load(args.listener_model_path + '/ckpt_dim_%d_id_%d.pth' % (args.context_dim, test_id))
    sea_net_test.load_state_dict(model_ckpt['net'])
    
    # Load listener symbols
    D99_id_test = Utiliz.generate_train_id(args.num_class, [test_id])
    listener_symbols_test = io.loadmat(args.listener_symbol_path + '/context_id_%d_e_%d.mat' % (test_id, args.listener_symbols_saveTimePoint))
    
    # Perform symbol translation using the loaded TInet
    outputs = symbol_translation_test(TInet_test, test_id)
    
    # Test the translated symbols on the listener agent (same logic as training phase)
    eval_log = open(path + '/evaluation_results.log', mode='a', encoding='utf-8')
    for round, output in enumerate(outputs):
        symbol_listener_test = torch.from_numpy(listener_symbols_test['context_%d_%d' % (test_id, args.listener_symbols_saveTimePoint)]).to(args.device)
        symbol_listener_test[test_id, :] = output
        results_pos, results_neg = AccracyTest.Acc2([symbol_listener_test], sea_net_test, test_id, [test_id], D99_id_test, 50)
        
        for i, idx in enumerate([test_id]):
            acc_pos = results_pos[1][i] / results_pos[2][i] if results_pos[2][i] > 0 else 0
            acc_neg = results_neg[1][i] / results_neg[2][i] if results_neg[2][i] > 0 else 0
            
            print('%d, %d, %.3f, %.3f, %.3f, %.3f, %.3f' % (test_id, round, results_pos[0][i], acc_pos, 
                                                     results_neg[0][i], acc_neg, (acc_pos + acc_neg) * .5), file=eval_log)
            final_acc_list.append((acc_pos + acc_neg) * .5)

            # Update overall statistics
            overall_stats['acc_pos_sum'] += acc_pos
            overall_stats['acc_neg_sum'] += acc_neg
            overall_stats['count'] += 1
    eval_log.close()

# Calculate and print overall statistics
if overall_stats['count'] > 0:
    avg_acc_pos = overall_stats['acc_pos_sum'] / overall_stats['count']
    avg_acc_neg = overall_stats['acc_neg_sum'] / overall_stats['count']
    
    print(f"\nOverall Evaluation Results:")
    print(f"Total test cases: {overall_stats['count']}")
    print(f"Average positive accuracy: {avg_acc_pos:.3f}")
    print(f"Average negative accuracy: {avg_acc_neg:.3f}")
    print(f"Average all samples accuracy: {(avg_acc_pos + avg_acc_neg) * .5:.3f}")
    
    # Save summary
    eval_log = open(path + '/evaluation_results.log', mode='a', encoding='utf-8')
    print('\nSummary:', file=eval_log)
    print('Total test cases: %d' % overall_stats['count'], file=eval_log)
    print('Average positive accuracy: %.3f' % avg_acc_pos, file=eval_log)
    print('Average negative accuracy: %.3f' % avg_acc_neg, file=eval_log)
    print(f'Average all samples accuracy: {(avg_acc_pos + avg_acc_neg) * .5:.3f}', file=eval_log)
    eval_log.close()

    sample1 = final_acc_list
    # --- Test 1: Check if the mean of each sample is significantly greater than zero ---
    # We use a one-sample t-test.
    # H0: The mean of the sample is equal to threshold.
    # H1: The mean of the sample is greater than threshold.
    print("--- One-sample t-tests (mean > 0) ---")
    alpha = 0.05 # Significance level
    threshold = 0.7
    # For sample 1
    # The 'alternative' parameter is set to 'greater' for a one-sided test.
    t_statistic_s1, p_value_s1 = stats.ttest_1samp(sample1, threshold, alternative='greater')
    print(f"Sample 1: t-statistic = {t_statistic_s1:.4f}, p-value = {p_value_s1:.4f}")
    print(f'mean {np.mean(sample1):.4f}, std {np.std(sample1):.4f}')
    if p_value_s1 < alpha:
        print(f"  The mean of Sample 1 is significantly greater than {threshold:.4f}")
    else:
        print(f"  There is not enough evidence to say the mean of Sample 1 is significantly greater than {threshold:.4f}")
    print("-" * 40)


else:
    print("No test results found - please check if trained models exist")

print("Evaluation completed. Results saved to evaluation_results.log")

