import torch
import torchvision
import torchvision.transforms as transforms
import random

datapath = './datafile'
def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_name:
            indices.append(i)
    return indices

def DLTrain_whole(batch_size_train, num_worker, datapath=datapath):
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                             std=[n / 255. for n in [68.2, 65.4, 70.4]]),
    ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=True,
                                                       num_workers=num_worker)
    return trainloader_cifar100


def DLTest_whole(batch_size_test, num_worker, datapath=datapath):
    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    ])
    testset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=False, download=True, transform=transform_test_cifar100)
    testloader_cifar100 = torch.utils.data.DataLoader(
        testset_cifar100, batch_size=batch_size_test, shuffle=False, num_workers=num_worker)
    return testloader_cifar100

def DLTrain_part(idxs, batch_size_train, num_worker, datapath=datapath):
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    idx_train = get_indices(trainset_cifar100, idxs)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
    trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
    return trainloader_cifar100

def DLTrain_part_FixedDataLength(idxs, single_class_data_length, batch_size_train, num_worker, datapath=datapath, shuffle_or_nor=False):
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    idx_train = []
    for id in idxs:
        idx_train_single = get_indices(trainset_cifar100, [id])
        if shuffle_or_nor:
            idx_train_sample = random.sample(idx_train_single, single_class_data_length)
        else:
            idx_train_sample = idx_train_single[0:single_class_data_length]
        idx_train.extend(idx_train_sample)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
    trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
    return trainloader_cifar100

def DLTrain_part_FixedDataLength_mix(id_pos, id_neg, pos_class_data_length, neg_class_data_length, batch_size_train, num_worker, datapath=datapath, shuffle_or_nor=False):
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    idx_train = []
    for id in id_pos:
        idx_train_single = get_indices(trainset_cifar100, [id])
        if shuffle_or_nor:
            idx_train_sample = random.sample(idx_train_single, pos_class_data_length)
        else:
            idx_train_sample = idx_train_single[0:pos_class_data_length]
        idx_train.extend(idx_train_sample)
    for id in id_neg:
        idx_train_single = get_indices(trainset_cifar100, [id])
        if shuffle_or_nor:
            idx_train_sample = random.sample(idx_train_single, neg_class_data_length)
        else:
            idx_train_sample = idx_train_single[0:neg_class_data_length]
        idx_train.extend(idx_train_sample)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
    trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
    return trainloader_cifar100

def DLTest_part(idxs, batch_size_test, num_worker, datapath=datapath):
    TestLoader = []
    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    ])
    testset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=False, download=True, transform=transform_test_cifar100)
    idx_test = get_indices(testset_cifar100, idxs)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(idx_test)
    testloader_cifar100 = torch.utils.data.DataLoader(
        testset_cifar100, batch_size=batch_size_test, shuffle=False, num_workers=num_worker, sampler=sampler_test)
    return testloader_cifar100