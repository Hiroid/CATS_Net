import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import MixDataLoader, SeparatedDataLoader

# import data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size_train = 100
batch_size_test = 100
worker = 0
num_class = 100
target_extend = torch.arange(num_class).repeat(2)
idxs = list(range(num_class))
trainloader_cifar100 = MixDataLoader.DLTrain_whole(batch_size_train, worker)
testloader_mix_cifar100 = MixDataLoader.DLTest_whole(batch_size_test, worker)
criterion = nn.CrossEntropyLoss()
path = '../' + str(datetime.now().date())

def Acc(contexts, my_extended_model, test_idxs, epoch_max):
    testloader_sep_cifar100 = SeparatedDataLoader.DLTest(test_idxs, batch_size_test, worker)
    loss_pos = torch.zeros(len(test_idxs))
    loss_neg = torch.zeros(len(test_idxs))
    totl_neg = torch.zeros(len(test_idxs))
    totl_pos = torch.zeros(len(test_idxs))
    crec_pos = torch.zeros(len(test_idxs))
    crec_neg = torch.zeros(len(test_idxs))
    for epoch in range(epoch_max):
        my_extended_model.eval()
        with torch.no_grad():
            for idx, testloader in enumerate(testloader_sep_cifar100):
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    rand_num = torch.ceil(torch.rand(len(targets)) * num_class).long().to(device)

                    context_pos = contexts[targets, :]
                    context_neg = contexts[target_extend[targets + rand_num], :]
                    binary_labels_pos = torch.ones(targets.size(0), dtype=torch.long).to(device)
                    binary_labels_neg = torch.zeros(targets.size(0), dtype=torch.long).to(device)

                    outputs_pos = my_extended_model(inputs, context_pos)
                    outputs_neg = my_extended_model(inputs, context_neg)
                    l_pos = criterion(outputs_pos, binary_labels_pos)
                    l_neg = criterion(outputs_neg, binary_labels_neg)

                    loss_pos[idx] += l_pos.item()
                    _, predicted_pos = outputs_pos.max(1)
                    totl_pos[idx] += targets.size(0)
                    crec_pos[idx] += predicted_pos.eq(binary_labels_pos).sum().item()

                    loss_neg[idx] += l_neg.item()
                    _, predicted_neg = outputs_neg.max(1)
                    totl_neg[idx] += targets.size(0)
                    crec_neg[idx] += predicted_neg.eq(binary_labels_neg).sum().item()

    return  [loss_pos/epoch_max, crec_pos, totl_pos], [loss_neg/epoch_max, crec_neg, totl_neg]

def Acc2(contexts, net, predicted_id, test_idxs_pos, test_idxs_neg, epoch_max_neg, epoch_max_pos=1):
    net.eval()
    testloader_sep_cifar100 = SeparatedDataLoader.DLTest(test_idxs_pos, batch_size_test, worker)
    # testloader_sep_cifar100 = SeparatedDataLoader.DLTrain(test_idxs_pos, batch_size_test, worker)
    testloader_mix_cifar100 = MixDataLoader.DLTest_part(test_idxs_neg, batch_size_test, worker)
    contexts_ids = torch.ones(batch_size_test, dtype=torch.long, device=device)*predicted_id
    context = contexts[0][contexts_ids, :]
    binary_labels_pos = torch.ones(batch_size_test, dtype=torch.long).to(device)
    binary_labels_neg = torch.zeros(batch_size_test, dtype=torch.long).to(device)
    loss_pos = torch.zeros(1)
    loss_neg = torch.zeros(1)
    totl_neg = torch.zeros(1)
    totl_pos = torch.zeros(1)
    crec_pos = torch.zeros(1)
    crec_neg = torch.zeros(1)
    with torch.no_grad():
        for idx, testloader in enumerate(testloader_sep_cifar100):
            for batch_idx, (inputs_pos, targets_pos) in enumerate(testloader):
                if batch_idx==epoch_max_pos:
                    break
                inputs_pos, targets_pos = inputs_pos.to(device), targets_pos.to(device)
                outputs_pos = net(inputs_pos, context)
                l_pos = criterion(outputs_pos, binary_labels_pos)

                loss_pos += l_pos.item()
                _, predicted_pos = outputs_pos.max(1)
                totl_pos += targets_pos.size(0)
                crec_pos += predicted_pos.eq(binary_labels_pos).sum().item()

        for batch_idx, (inputs_neg, targets_neg) in enumerate(testloader_mix_cifar100):
            if batch_idx==epoch_max_neg:
                break
            inputs_neg, targets_neg = inputs_neg.to(device), targets_neg.to(device)
            outputs_neg = net(inputs_neg, context)
            l_neg = criterion(outputs_neg, binary_labels_neg)

            loss_neg += l_neg.item()
            _, predicted_neg = outputs_neg.max(1)
            totl_neg += targets_neg.size(0)
            crec_neg += predicted_neg.eq(binary_labels_neg).sum().item()

    return  [loss_pos/epoch_max_pos, crec_pos, totl_pos], [loss_neg/epoch_max_neg, crec_neg, totl_neg]



def test_mix(contexts, net):
    loss_pos = 0
    loss_neg = 0
    totl_neg = 0
    totl_pos = 0
    crec_pos = 0
    crec_neg = 0
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader_mix_cifar100):
        inputs, targets = inputs.to(device), targets.to(device)
        rand_num = torch.ceil(torch.rand(len(targets)) * num_class).long().to(device)

        context_pos = contexts[targets, :]
        context_neg = contexts[target_extend[targets + rand_num], :]
        binary_labels_pos = torch.ones(targets.size(0), dtype=torch.long).to(device)
        binary_labels_neg = torch.zeros(targets.size(0), dtype=torch.long).to(device)

        outputs_pos = net(inputs, context_pos)
        outputs_neg = net(inputs, context_neg)
        l_pos = criterion(outputs_pos, binary_labels_pos)
        l_neg = criterion(outputs_neg, binary_labels_neg)

        loss_pos += l_pos.item()
        _, predicted_pos = outputs_pos.max(1)
        totl_pos += targets.size(0)
        crec_pos += predicted_pos.eq(binary_labels_pos).sum().item()

        loss_neg += l_neg.item()
        _, predicted_neg = outputs_neg.max(1)
        totl_neg += targets.size(0)
        crec_neg += predicted_neg.eq(binary_labels_neg).sum().item()

    return  [loss_pos/(batch_idx+1), crec_pos/totl_pos], [loss_neg/(batch_idx+1), crec_neg/totl_neg]


def Acc_logic(contexts1, contexts2, net, predicted_id, test_idxs_pos, test_idxs_neg, epoch_max_neg, epoch_max_pos=1):
    net.eval()
    testloader_sep_cifar100 = SeparatedDataLoader.DLTest(test_idxs_pos, batch_size_test, worker)
    # testloader_sep_cifar100 = SeparatedDataLoader.DLTrain(test_idxs_pos, batch_size_test, worker)
    testloader_mix_cifar100 = MixDataLoader.DLTest_part(test_idxs_neg, batch_size_test, worker)
    contexts_ids = torch.ones(batch_size_test, dtype=torch.long, device=device)*predicted_id
    context1 = contexts1[0][contexts_ids, :]
    context2 = contexts2[0][contexts_ids, :]
    binary_labels_pos = torch.ones(batch_size_test, dtype=torch.long).to(device)
    binary_labels_neg = torch.zeros(batch_size_test, dtype=torch.long).to(device)
    loss_pos = torch.zeros(1)
    loss_neg = torch.zeros(1)
    totl_neg = torch.zeros(1)
    totl_pos = torch.zeros(len(test_idxs_pos))
    crec_pos = torch.zeros(len(test_idxs_pos))
    crec_neg = torch.zeros(1)
    with torch.no_grad():
        for idx, testloader in enumerate(testloader_sep_cifar100):
            for batch_idx, (inputs_pos, targets_pos) in enumerate(testloader):
                if batch_idx==epoch_max_pos:
                    break
                inputs_pos, targets_pos = inputs_pos.to(device), targets_pos.to(device)
                outputs_pos1 = net(inputs_pos, context1)
                outputs_pos2 = net(inputs_pos, context2)
                # l_pos = criterion(outputs_pos, binary_labels_pos)

                # loss_pos += l_pos.item()
                _, predicted_pos1 = outputs_pos1.max(1)
                _, predicted_pos2 = outputs_pos2.max(1)
                totl_pos[idx] += targets_pos.size(0)
                crec_pos[idx] += predicted_pos1.eq(binary_labels_pos).mul(predicted_pos2.eq(binary_labels_pos)).sum().item()

        for batch_idx, (inputs_neg, targets_neg) in enumerate(testloader_mix_cifar100):
            if batch_idx==epoch_max_neg:
                break
            inputs_neg, targets_neg = inputs_neg.to(device), targets_neg.to(device)
            outputs_neg1 = net(inputs_neg, context1)
            outputs_neg2 = net(inputs_neg, context2)
            # l_neg = criterion(outputs_neg, binary_labels_neg)

            # loss_neg += l_neg.item()
            _, predicted_neg1 = outputs_neg1.max(1)
            _, predicted_neg2 = outputs_neg2.max(1)
            totl_neg += targets_neg.size(0)
            crec_neg += predicted_neg1.eq(binary_labels_neg).mul(predicted_neg2.eq(binary_labels_neg)).sum().item()

    return  [loss_pos/epoch_max_pos, crec_pos, totl_pos], [loss_neg/epoch_max_neg, crec_neg, totl_neg]



def Acc_seperate(contexts, net, predicted_id, test_idxs_pos, test_idxs_neg, epoch_max_neg, epoch_max_pos=1):
    net.eval()
    testloader_pos_cifar100 = SeparatedDataLoader.DLTest(test_idxs_pos, batch_size_test, worker)
    # testloader_sep_cifar100 = SeparatedDataLoader.DLTrain(test_idxs_pos, batch_size_test, worker)
    testloader_neg_cifar100 = SeparatedDataLoader.DLTest(test_idxs_neg, batch_size_test, worker)
    contexts_ids = torch.ones(batch_size_test, dtype=torch.long, device=device)*predicted_id
    context = contexts[0][contexts_ids, :]
    binary_labels_pos = torch.ones(batch_size_test, dtype=torch.long).to(device)
    binary_labels_neg = torch.zeros(batch_size_test, dtype=torch.long).to(device)
    loss_pos = torch.zeros(len(test_idxs_pos))
    loss_neg = torch.zeros(len(test_idxs_neg))
    totl_neg = torch.zeros(len(test_idxs_neg))
    totl_pos = torch.zeros(len(test_idxs_pos))
    crec_pos = torch.zeros(len(test_idxs_pos))
    crec_neg = torch.zeros(len(test_idxs_neg))
    with torch.no_grad():
        for idx_pos, testloader_pos in enumerate(testloader_pos_cifar100):
            for batch_idx, (inputs_pos, targets_pos) in enumerate(testloader_pos):
                if batch_idx==epoch_max_pos:
                    break
                inputs_pos, targets_pos = inputs_pos.to(device), targets_pos.to(device)
                outputs_pos1 = net(inputs_pos, context)
                # l_pos = criterion(outputs_pos, binary_labels_pos)

                # loss_pos += l_pos.item()
                _, predicted_pos1 = outputs_pos1.max(1)
                totl_pos[idx_pos] += targets_pos.size(0)
                crec_pos[idx_pos] += predicted_pos1.eq(binary_labels_pos).sum().item()

        for idx_neg, testloader_neg in enumerate(testloader_neg_cifar100):
            for batch_idx, (inputs_neg, targets_neg) in enumerate(testloader_neg):
                if batch_idx==epoch_max_neg:
                    break
                inputs_neg, targets_neg = inputs_neg.to(device), targets_neg.to(device)
                outputs_neg2 = net(inputs_neg, context)
                # l_neg = criterion(outputs_neg, binary_labels_neg)

                # loss_neg += l_neg.item()
                _, predicted_neg2 = outputs_neg2.max(1)
                totl_neg[idx_neg] += targets_neg.size(0)
                crec_neg[idx_neg] += predicted_neg2.eq(binary_labels_neg).sum().item()

    return  [loss_pos, crec_pos, totl_pos], [loss_neg, crec_neg, totl_neg]