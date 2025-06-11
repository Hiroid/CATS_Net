# add the path of custom functions
import sys
sys.path.append("../../Deps/CustomFuctions")

# add necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import torchvision.models as models
from models import *
import scipy.io as io
import os
import argparse
import multiprocessing
import MixDataLoader, SeparatedDataLoader, SEAnet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# parameters setting
parser = argparse.ArgumentParser(description='parameters setting')
parser.add_argument('--device', type=str, default='cuda', help='device type: cuda or cpu')
parser.add_argument('--worker', type=int, default=1, help='worker for data loader')
parser.add_argument('--batch_size_train', type=int, default=200, help='batch size for training dataset')
parser.add_argument('--batch_size_test', type=int, default=100, help='batch size for testing dataset')
parser.add_argument('--num_class', type=int, default=100, help='number of total classes')
parser.add_argument('--start_epoch', type=int, default=0, help='id of the begining epoch')
parser.add_argument('--end_epoch', type=int, default=2000 , help='id of the ending epoch')
parser.add_argument('--context_dim', type=int, default=20, help='context dimension')
parser.add_argument('--noise_intensity', type=list, default=[1e-1], help='intensity of noises injected into the contexts')
parser.add_argument('--epoch_node', type=list, default=[200, 500, 800, 1000, 1500, 1999], help='the epoch to record results')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode: train or test')
args = parser.parse_args()

# load data
# CIFAR 100
print('==> Preparing data..')
class_idxs = list(range(args.num_class))
trainloader_cifar100 = MixDataLoader.DLTrain_whole(args.batch_size_train, args.worker)
testloader_mix_cifar100 = MixDataLoader.DLTest_whole(args.batch_size_test, args.worker)
testloader_cifar100 = SeparatedDataLoader.DLTest(class_idxs, args.batch_size_test, args.worker)

# load pretrained module
pretrained_classifier_cnn = models.resnet18(weights=None)
pretrained_classifier_cnn.load_state_dict(torch.load('../../Deps/pretrained_fe/resnet18-f37072fd.pth'))
pretrained_classifier_cnn.fc = nn.Identity()
pretrained_cdp_cnn = VGG('VGG11')

def train(round, epoch, ni, target_extend, sea_net, optimizer_net, optimizer_symbol, criterion, contexts, path):
    sea_net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader_cifar100):
            p = torch.rand(1)
            neg_mask = torch.rand(args.batch_size_train).ge(p[0])
            binary_labels = torch.ones(args.batch_size_train, dtype=torch.long)
            binary_labels[neg_mask] = 0
            rand_num = torch.ceil(torch.rand(neg_mask.sum())*args.num_class).long()
            context = contexts[0][targets, :]
            context[neg_mask] = contexts[0][target_extend[targets[neg_mask]+rand_num-1], :]
            context = context + (torch.rand(context.size(), device=args.device)*2.0 - 1.0)*ni
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            context = context.to(args.device)
            binary_labels = binary_labels.to(args.device)

            output = sea_net(inputs, context)
            loss = criterion(output, binary_labels)
            optimizer_symbol.zero_grad()
            optimizer_net.zero_grad()
            loss.backward()
            if epoch%2 == 0:
                optimizer_net.step()
            else:
                optimizer_symbol.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(binary_labels).sum().item()

    DataLog = open(path +'/Train_results.log', mode='a', encoding='utf-8')
    if epoch == args.start_epoch:
        print('Round, Epoch, Loss, Acc')
    print('%d, %d, %.3f, %.3f' % (round, epoch, train_loss / batch_idx, correct / total), file=DataLog)
    DataLog.close()

def test(round, epoch, target_extend, sea_net, criterion, contexts, path):
    sea_net.eval()

    test_loss_pos = 0
    predicted_pos = 0
    total_pos = 0
    correct_pos = 0

    test_loss_neg = 0
    predicted_neg = 0
    total_neg = 0
    correct_neg = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_mix_cifar100):

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            rand_num = torch.ceil(torch.rand(args.batch_size_test) * args.num_class).long().to(args.device)

            context_pos = contexts[0][targets, :]
            context_neg = contexts[0][target_extend[targets + rand_num - 1], :]
            binary_labels_pos = torch.ones(targets.size(0), dtype=torch.long).to(args.device)
            binary_labels_neg = torch.zeros(targets.size(0), dtype=torch.long).to(args.device)

            outputs_pos = sea_net(inputs, context_pos)
            outputs_neg = sea_net(inputs, context_neg)
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

    # Only log test results when in training mode
    if args.mode == 'train':
        testlog = open(path + '/Test_Results.log', mode='a', encoding='utf-8')
        if epoch == args.start_epoch:
            print('Round, Epoch, Loss_positive, Acc_positive, Loss_negative, Acc_negative', file=testlog)
        print('%d, %d, %.3f, %.3f, %.3f, %.3f' % (round, epoch, test_loss_pos / batch_idx, correct_pos / total_pos, test_loss_neg / batch_idx, correct_neg / total_neg), file=testlog)
        testlog.close()

    print('Round,\tEpoch,\tLoss_pos,\tAcc_pos,\tLoss_neg,\tAcc_neg')
    print('%d,\t%d,\t%.3f,\t%.3f,\t%.3f,\t%.3f' % (round, epoch, test_loss_pos / batch_idx, correct_pos / total_pos, test_loss_neg / batch_idx, correct_neg / total_neg))

def context_search(round, ni):

    # make directory to store results
    path = f_path + '/' + 'ni=%.2e_r=%d' % (ni, round)
    if not os.path.isdir(path + '/contexts'):
        os.makedirs(path + '/contexts')
    if not os.path.isdir(path + '/checkpoint'):
        os.makedirs(path + '/checkpoint')

    # build and initialize the symbole vectors in the tensor variable "contexts"
    contexts = [torch.rand(args.num_class, args.context_dim).to(args.device)]
    contexts[0].requires_grad = True
    target_extend = torch.arange(args.num_class, device=args.device).repeat(2)
    io.savemat(path + '/contexts/context_initial_r_%d.mat' % (round),
               {'context_ini_%d' % (round): np.array(contexts[0].cpu().detach())})


    # structure and parameter setting
    sea_net = SEAnet.Net2(my_pretrained_classifier=pretrained_classifier_cnn,
                                         my_pretrained_cdp=pretrained_cdp_cnn, context_dim=args.context_dim).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_net = optim.Adam(sea_net.parameters(), lr=0.0001)
    optimizer_symbol = optim.Adam(contexts, lr=0.0001)

    for epoch in range(args.start_epoch, args.end_epoch):
        train(round, epoch, ni, target_extend, sea_net, optimizer_net, optimizer_symbol, criterion, contexts, path)
        test(round, epoch, target_extend, sea_net, criterion, contexts, path)
        if epoch in args.epoch_node:
            if not os.path.isdir(path + '/contexts'):
                os.makedirs(path + '/contexts')
            io.savemat(path + '/contexts/context_r_%d_e_%d.mat' % (round, epoch),
                       {'context_%d_%d' % (round, epoch): np.array(contexts[0].cpu().detach())})
        if epoch == args.end_epoch-1:
            state = {
                    'net': sea_net.state_dict(),
                    'context': contexts,
                    'opt_context': optimizer_symbol.state_dict(),
                    'opt_net': optimizer_net.state_dict(),
                    'epoch': epoch,
                }
            torch.save(state, './' + path + '/checkpoint/ckpt_dim_%d.pth' % (args.context_dim))
            torch.save(sea_net, './' + path + '/checkpoint/ckpt_dim_%d_whole.pth' % (args.context_dim))

def context_test(round, ni):

    # make directory to store results
    path = f_path + '/' + 'ni=%.2e_r=%d' % (ni, round)

    # Determine the latest epoch to load context from
    latest_epoch_to_load = args.epoch_node[-1] # Assumes epoch_node is sorted

    # Define target_extend
    target_extend = torch.arange(args.num_class, device=args.device).repeat(2)

    # Load context
    context_file_path = path + '/contexts/context_r_%d_e_%d.mat' % (round, latest_epoch_to_load)
    if not os.path.exists(context_file_path):
        print(f"Context file not found: {context_file_path}")
        return
    mat_data = io.loadmat(context_file_path)
    contexts = [torch.tensor(mat_data['context_%d_%d' % (round, latest_epoch_to_load)], device=args.device)]
    # contexts[0].requires_grad = True # Not needed for testing

    # Load pre-trained sea_net model
    model_path = './' + path + '/checkpoint/ckpt_dim_%d_whole.pth' % (args.context_dim)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    sea_net = torch.load(model_path, map_location=args.device)
    sea_net.eval() # Set model to evaluation mode

    criterion = nn.CrossEntropyLoss()

    # Run test for a single epoch (or a representative pass)
    # The original loop was for training epochs, which is not needed here.
    # We'll call test once, assuming epoch 0 for logging purposes, or use latest_epoch_to_load
    print(f"Testing with context from epoch {latest_epoch_to_load}")
    test(round, latest_epoch_to_load, target_extend, sea_net, criterion, contexts, path)

f_path = '.'
for ni_idx, ni in enumerate(args.noise_intensity):
    if args.mode == 'train':
        context_search(ni_idx, ni)
    else:  # test mode
        context_test(ni_idx, ni)




