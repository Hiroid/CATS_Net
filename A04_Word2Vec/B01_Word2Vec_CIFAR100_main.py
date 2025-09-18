import sys
import os
from pathlib import Path
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
import numpy as np
from datetime import datetime
import torchvision.models as models
from models import *
import scipy.io as io
import argparse
import scipy.stats as stats

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import MixDataLoader, SeparatedDataLoader
import CATSnet,  AccracyTest

# parameters
parser = argparse.ArgumentParser(description='set parameters')
parser.add_argument('--pretraind_model_path', type=str, default='/data/home/scv2590/run/common data/torch_models/', help='path to pretrained model store file')
parser.add_argument('--device', type=str, default='cuda', help='cuda if torch.cuda.is_available() else cpu')
parser.add_argument('--worker', type=int, default=0, help='gpu worker id')
parser.add_argument('--batch_size_train', type=int, default=100, help='batch size for training dataset')
parser.add_argument('--batch_size_test', type=int, default=100, help='batch size for testing dataset')
parser.add_argument('--num_class', type=int, default=100, help='number of total classes')
# parser.add_argument('--test_id', type=int, nargs='+', default=[0], help='the ids of the classes used to test')
parser.add_argument('--test_id', type=list, default=list(range(100)), help='the ids of the classes used to test')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--end_epoch', type=int, default=1000, help='epoch number to end with')
parser.add_argument('--context_dim', type=int, default=20, help='context dimension')
parser.add_argument('--noise_intensity', type=float, nargs='+', default=[0.05], help='intensity of noises added to the contexts')
parser.add_argument('--train_mode', type=bool, default=False, help='set TRUE to train the network, FALSE to test the network')
parser.add_argument('--drop_probility', type=float, default=0.2, help='drop probility for CDP network 3 to avoid over-fitting')
parser.add_argument('--num_process', type=int, default=10, help='how many processes to run in parallel')
parser.add_argument('--wordvec_path', type=str, default='./wordvector/embedding_dim_20.mat', help='path of mat file to store the reduced word vector')
# parser.add_argument('--test_round', type=int, default=2, help='round number to run, determine how many contexts with be generated')
# parser.add_argument('--test_times', type=int, default=10, help='times of tests to search best contexts')
# parser.add_argument('--pretraind_model_path', type=str, default='/share/home/ychen/project/common data/torch_models', help='path to pretrained model store file')
# parser.add_argument('--common_data', type=str, default='/share/home/ychen/project/common data/data', help='path to common data store file')
args = parser.parse_args()


# import data
print('==> Preparing data..')
dataloader_cifar100 = MixDataLoader.DLTrain_part(list(range(100)), args.batch_size_train, args.worker)
name_vecs = torch.from_numpy(io.loadmat(args.wordvec_path)['wv_d_%d' % (args.context_dim)]) 

# neural network model
pretrained_classifier_cnn = models.resnet18()
pretrained_classifier_cnn.load_state_dict(
    torch.load(
        os.path.join(project_root, "Deps", "pretrained_fe", "resnet18-f37072fd.pth")
    )
)
pretrained_classifier_cnn.fc = nn.Identity()

def train(epoch, ni, trainloader_cifar100, my_extended_model, optimizer_net, contexts, criterion, target_extend, train_id, test_id, id_location):
    print('\nEpoch: %d' % epoch)
    # global best_acc
    my_extended_model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader_cifar100):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            p = torch.rand(1)
            neg_mask = torch.rand(args.batch_size_train).ge(p[0])
            binary_labels = torch.ones(args.batch_size_train, dtype=torch.long)
            binary_labels[neg_mask] = 0
            rand_num = torch.ceil(torch.rand(args.batch_size_train-binary_labels.sum())*len(train_id)).long().to(args.device)
            context = contexts[0][targets, :]
            context[neg_mask] = contexts[0][target_extend[id_location[targets[neg_mask]]+rand_num], :]
            context = context + (torch.rand(context.size(), device=args.device)*2.0 - 1.0)*ni
            context = context.to(args.device)
            binary_labels = binary_labels.to(args.device)

            output = my_extended_model(inputs, context)
            loss = criterion(output, binary_labels)
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
            # optimizer_input.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(binary_labels).sum().item()

    acc = 100. * correct / total
    mylog = open('./'+path+'/recode_id_%d.log' % (test_id), mode='a', encoding='utf-8')
    if epoch==args.start_epoch:
        print('Epoch, Loss, Acc', file=mylog)
    print('%d, %.3f,  %.3f%%' % (epoch, train_loss / (batch_idx+1), acc), file=mylog)
    mylog.close()


def test(epoch, testloader_cifar100, my_extended_model, contexts, criterion, whole_target_extend, test_id, id_location):
    my_extended_model.eval()

    test_loss_pos = 0
    predicted_pos = 0
    total_pos = 0
    correct_pos = 0

    test_loss_neg = 0
    predicted_neg = 0
    total_neg = 0
    correct_neg = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_cifar100):

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            rand_num = torch.ceil(torch.rand(args.batch_size_test) * args.num_class).long().to(args.device)

            context_pos = contexts[0][targets, :]
            context_neg = contexts[0][whole_target_extend[id_location[targets]+rand_num], :]
            binary_labels_pos = torch.ones(targets.size(0), dtype=torch.long).to(args.device)
            binary_labels_neg = torch.zeros(targets.size(0), dtype=torch.long).to(args.device)

            outputs_pos = my_extended_model(inputs, context_pos)
            outputs_neg = my_extended_model(inputs, context_neg)
            loss_pos = criterion(outputs_pos, binary_labels_pos)
            loss_neg = criterion(outputs_neg, binary_labels_neg)

            test_loss_pos += loss_pos.item()
            _, predicted_pos = outputs_pos.max(1)
            total_pos += targets.size(0)
            correct_pos += predicted_pos.eq(binary_labels_pos).sum().item()

            test_loss_neg += loss_neg.item()
            _, predicted_neg = outputs_neg.max(1)
            total_neg += targets.size(0)
            correct_neg += predicted_neg.eq(binary_labels_neg).sum().item()

    # testlog = open('./' + path + '/test_id_%d_acc.log' % (test_id), mode='a', encoding='utf-8')
    # if epoch==args.start_epoch:
    #     print('Epoch, Loss_pos, Acc_pos, Loss_neg, Acc_neg', file=testlog)
    # print('%d, %.3f, %.3f%%, %.3f, %.3f%%' % (epoch, test_loss_pos / (batch_idx+1), 100. * correct_pos / total_pos, test_loss_neg / (batch_idx+1), 100. * correct_neg / total_neg), file=testlog)
    # testlog.close()

    return (100. * correct_pos / total_pos + 100. * correct_neg / total_neg) * .5


run_time = datetime.now().date()

def model_training(test_id, ni):

    train_id = [x for x in range(args.num_class) if x not in [test_id]]
    all_id = train_id + [test_id]
    id_location = torch.tensor([all_id.index(x) for x in range(args.num_class)], dtype=torch.long).to(args.device)
    best_acc = 0

    ## import data
    # CIFAR 100
    print('==> Preparing data..')
    trainloader_cifar100 = MixDataLoader.DLTrain_part(train_id, args.batch_size_train, args.worker)
    testloader_cifar100 = MixDataLoader.DLTest_part([test_id], args.batch_size_test, args.worker)

    # context initialization
    contexts = [torch.zeros(args.num_class, args.context_dim).to(args.device)]
    for id, name_vec in enumerate(name_vecs):
        contexts[0][id, :] = torch.tensor(name_vec, dtype=torch.float).to(args.device) * 10
    target_extend = torch.tensor(train_id * 2).to(args.device)
    whole_target_extend = torch.tensor(all_id*2).to(args.device)
    io.savemat('./' + path + '/contexts/context_initial_id_%d.mat' % (test_id),
               {'contexts': np.array(contexts[0].cpu().detach()),
                'all_idx': np.array(all_id),
                'trained_idx': np.array(train_id),
                'tested_idx': np.array(test_id)})

    # structure and parameter setting
    my_extended_model = CATSnet.Net2(
        my_pretrained_classifier=pretrained_classifier_cnn,
        context_dim=args.context_dim
    ).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_net = optim.Adam(my_extended_model.parameters(), lr=0.0001)

    for epoch in range(args.start_epoch, args.end_epoch):
        train(epoch, ni, trainloader_cifar100, my_extended_model, optimizer_net, contexts, criterion, target_extend, train_id, test_id, id_location)
        if epoch % 2 == 0:
            current_acc = test(epoch, testloader_cifar100, my_extended_model, contexts, criterion, whole_target_extend, test_id, id_location)

            if best_acc < current_acc:
                best_acc = current_acc
                print(f"best epoch now {epoch}")
                state = {
                    'net': my_extended_model.state_dict(),
                    'context': contexts,
                    'class_order': train_id,
                    'class_index': id_location,
                    'num_class': args.num_class,
                    'opt_net': optimizer_net.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, './' + path + '/checkpoint/ckpt_dim_%d_id_%d.pth' % (args.context_dim, test_id))

def wordvector_test():
    """
    Load trained models and perform testing for each test_id
    """
    print('==> Starting wordvector testing..')
    
    for noise_intensity in args.noise_intensity:
        path = './' + 'ni=%.2e' % (noise_intensity)
        print(f'Testing with noise intensity: {noise_intensity}')
        final_acc_list = []
        for test_id in args.test_id:
            print(f'Testing class ID: {test_id}')
            
            # Prepare checkpoint path
            checkpoint_path = path + '/checkpoint/ckpt_dim_%d_id_%d.pth' % (args.context_dim, test_id)
            
            if not os.path.exists(checkpoint_path):
                print(f'Checkpoint not found: {checkpoint_path}')
                continue
                
            # Load checkpoint
            print(f'Loading checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            
            # Reconstruct model
            my_extended_model = CATSnet.Net2(
                my_pretrained_classifier=pretrained_classifier_cnn,
                context_dim=args.context_dim
            ).to(args.device)
            my_extended_model.load_state_dict(checkpoint['net'])
            
            # Load contexts and class information
            contexts = checkpoint['context']
            train_id = checkpoint['class_order']
            id_location = checkpoint['class_index']
            all_id = train_id + [test_id]
            whole_target_extend = torch.tensor(all_id*2).to(args.device)
            
            # Prepare test data
            testloader_cifar100 = MixDataLoader.DLTest_part([test_id], args.batch_size_test, args.worker)
            criterion = nn.CrossEntropyLoss()
            
            # Perform testing
            print(f'Testing model for class {test_id}...')
            final_acc = test(checkpoint['epoch'], testloader_cifar100, my_extended_model, 
                           contexts, criterion, whole_target_extend, test_id, id_location)
            final_acc_list.append(final_acc)
            print(f'Final accuracy for class {test_id}: {final_acc:.3f}%')
            
            # Save test results
            result_log = open(path + '/final_test_results.log', mode='a', encoding='utf-8')
            print(f'Class_ID: {test_id}, Final_Accuracy: {final_acc:.3f}%, Epoch: {checkpoint["epoch"]}', file=result_log)
            result_log.close()
        
        result_log = open(path + '/final_test_results.log', mode='a', encoding='utf-8')
        
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

        result_log.close()

    print('==> Wordvector testing completed!')

for noise_intensity in args.noise_intensity:
    path = './' + 'ni=%.2e' % (noise_intensity)
    if not os.path.isdir('./' + path + '/contexts'):
        os.makedirs('./' + path + '/contexts')
    if not os.path.isdir('./' + path + '/checkpoint'):
        os.makedirs('./' + path + '/checkpoint')
    
    # for test_class_id in args.test_id:
    #     model_training(test_class_id, noise_intensity)

    if __name__ == '__main__':
        if args.train_mode:
            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=args.num_process)
            for test_class_id in args.test_id:
                pool.apply_async(model_training, (test_class_id, noise_intensity, ))
            pool.close()
            pool.join()
        else:
            wordvector_test()