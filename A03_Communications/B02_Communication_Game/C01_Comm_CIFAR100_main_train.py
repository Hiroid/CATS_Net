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

import Translators, AccracyTest, SEAnet,  Utiliz
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameters
parser = argparse.ArgumentParser(description='parameters setting')
parser.add_argument('--device', type=str, default='cuda', help='device type: cuda or cpu')
parser.add_argument('--worker', type=int, default=1, help='worker for data loader')
parser.add_argument('--ct_batch_size_train', type=int, default=96, help='batch size of symbols in D_99 when training TI module')
parser.add_argument('--batch_size_test', type=int, default=100, help='batch size for testing dataset')
parser.add_argument('--num_class', type=int, default=100, help='number of total classes')
parser.add_argument('--class_id_unaligned', type=list, default=list(range(1,100)), help='the IDs of unaligned classes during the TI training')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--end_epoch', type=int, default=102, help='epoch number to end with')
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

def train(epoch):
    # global best_acc
    TInet.train()
    sample_orders = random.sample(args.train_round_list, args.ct_batch_size_train)
    loss = 0
    for sp_order in sample_orders:
        # load the one of 96 symbol sets of speaker agent.
        inputs = io.loadmat(args.path_symbol_speaker_train + '/context_r_%d_e_999.mat' % (sp_order))
        inputs = torch.from_numpy(inputs['context_%d_%d' % (sp_order, 999)][D99_id])
        targets = listener_symbols_D99
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        output = TInet(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch == args.start_epoch:
        trainlog = open(path + '/train_loss.log', mode='a', encoding='utf-8')
        print('\n\n', file=trainlog)
        print('tested class id: %d' % (test_id), file=trainlog)
        trainlog.close
    if epoch % 20 == 0:
        trainlog = open(path + '/train_loss.log', mode='a', encoding='utf-8')
        print(loss, file=trainlog) 
        trainlog.close

def symbol_translation(epoch):
    TInet.eval()
    outputs = []
    testlog = open(path + '/symbol_translation.log', mode='a', encoding='utf-8')
    if epoch==args.start_epoch:
        print('class id: %d' % (test_id), file=testlog)
    print('epoch: %d' % (epoch), file=testlog)
    with torch.no_grad():
        for sp_order in args.test_round_list:
            inputs = io.loadmat(args.path_symbol_speaker_test + 'context_ep_1999.mat')
            inputs = torch.from_numpy(inputs['context_0_1999'][test_id]).to(args.device)
            inputs = inputs.unsqueeze(0)
            output = TInet(inputs)
            outputs.append(output)
            print('round %d:' % (sp_order), file=testlog)
            print('predicted contexts for net2:', file=testlog)
            print(output, file=testlog)
    testlog.close()
    return outputs

# creat results-saving directory 
run_date = datetime.now().date()
path = './' + str(run_date)
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

symbollog = open(path + '/symbol_translation.log', mode='a', encoding='utf-8')
symbollog.close()
testlog = open(path + '/test_acc.log', mode='a', encoding='utf-8')
print('test_class_id, round, epoch, loss_pos, acc_pos, loss_neg, acc_neg', file=testlog)
testlog.close()
trainlog = open(path + '/train_loss.log', mode='a', encoding='utf-8')
trainlog.close()


for test_id in args.class_id_unaligned:
    best_acc = 0
    # network initialization
    TInet = Translators.TImodule(context_dim=args.context_dim, num_hidden_layer=args.TInet_hidden_layers,
                        num_hidden_neuron=args.TInet_hidden_neurons, dropout_p=args.TInet_p).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(TInet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sche_steps, gamma=args.lr_sche_gamma)
    sea_net = SEAnet.Net2(
        my_pretrained_classifier=pretrained_classifier_cnn,
        context_dim=args.context_dim
    ).to(args.device)
    model_ckpt = torch.load(args.listener_model_path + '/ckpt_dim_%d_id_%d.pth' % (args.context_dim, test_id))                                   
    sea_net.load_state_dict(model_ckpt['net'])

    # load symbols of listener agent
    D99_id = Utiliz.generate_train_id(args.num_class, [test_id])
    listener_symbols = io.loadmat(args.listener_symbol_path + '/context_id_%d_e_%d.mat' % (test_id, args.listener_symbols_saveTimePoint))
    listener_symbols_D99 = torch.from_numpy(listener_symbols['context_%d_%d' % (test_id, args.listener_symbols_saveTimePoint)][D99_id, :])
    translated_symbols = []

    for epoch in range(args.start_epoch, args.end_epoch):

        # test the symbols translated from the speaker agent on the listener agent.
        if epoch % 2 == 0:
            outputs = symbol_translation(epoch)
            translated_symbols.append(outputs[0])
            testlog = open('./' + path + '/test_acc.log', mode='a', encoding='utf-8')
            for round, output in enumerate(outputs):
                symbol_listener_test = torch.from_numpy(listener_symbols['context_%d_%d' % (test_id, args.listener_symbols_saveTimePoint)]).to(args.device)
                symbol_listener_test[test_id, :] = output
                results_pos, results_neg = AccracyTest.Acc2([symbol_listener_test], sea_net, test_id, [test_id], D99_id, 50)
                for i, idx in enumerate([test_id]):
                    print('%d, %d, %d, %.3f, %.3f, %.3f, %.3f' % (test_id, round, epoch, results_pos[0][i], results_pos[1][i] / results_pos[2][i], \
                        results_neg[0][i], results_neg[1][i] / results_neg[2][i]), file=testlog)  
                    current_acc = .5 * (results_pos[1][i] / results_pos[2][i]) + .5 * (results_neg[1][i] / results_neg[2][i])
                    if best_acc < current_acc:
                        best_acc = current_acc
                        symbol_dict = {}
                        for i, output in enumerate(translated_symbols):
                            symbol_dict['check_point_%d' % (i)] = np.array(output.cpu())
                        state = {
                            'net': TInet.state_dict(),
                            'context': listener_symbols,
                            'test_idxs': test_id
                        }
                        if not os.path.isdir(save_path_cp):
                            os.makedirs(save_path_cp)
                        torch.save(state, save_path_cp + '/TInet_testid_%d.pth' % (test_id))
                        torch.save(TInet.state_dict(), save_path_cp + '/TInet_testid_%d_whole.pth' % (test_id))
                        if not os.path.isdir(save_path_ct):
                            os.makedirs(save_path_ct)
                        io.savemat(save_path_ct + '/context_pred_testid_%d.mat' % (test_id),
                                    symbol_dict)
            testlog.close()

        train(epoch)


