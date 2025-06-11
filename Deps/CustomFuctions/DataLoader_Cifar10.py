import torch
import torchvision
import torchvision.transforms as transforms

datapath = 'D:\Projects\Context Dependent Learning\datafile'
def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_name:
            indices.append(i)
    return indices

def DLTrain_whole(batch_size_train, num_worker, datapath=datapath):
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n for n in [0.4914, 0.4822, 0.4465]],
                             std=[n for n in [0.247, 0.243, 0.261]]),
    ])
    trainset_cifar10 = torchvision.datasets.CIFAR10(
        root=datapath, train=True, download=True, transform=transform_train_cifar10)
    trainloader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=batch_size_train, shuffle=True,
                                                       num_workers=num_worker)
    return trainloader_cifar10


def DLTest_whole(batch_size_test, num_worker, datapath=datapath):
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n for n in [0.4914, 0.4822, 0.4465]], std=[n for n in [0.247, 0.243, 0.261]])
    ])
    testset_cifar10 = torchvision.datasets.CIFAR10(
        root=datapath, train=False, download=True, transform=transform_test_cifar10)
    testloader_cifar10 = torch.utils.data.DataLoader(
        testset_cifar10, batch_size=batch_size_test, shuffle=False, num_workers=num_worker)
    return testloader_cifar10

def DLTrain_part(idxs, batch_size_train, num_worker, datapath=datapath):
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n for n in [0.4914, 0.4822, 0.4465]], std=[n for n in [0.247, 0.243, 0.261]]),
        ])
    trainset_cifar10 = torchvision.datasets.CIFAR10(
        root=datapath, train=True, download=True, transform=transform_train_cifar10)
    idx_train = get_indices(trainset_cifar10, idxs)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
    trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
    return trainloader_cifar100


def DLTest_part(idxs, batch_size_test, num_worker, datapath=datapath):
    TestLoader = []
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n for n in [0.4914, 0.4822, 0.4465]], std=[n for n in [0.247, 0.243, 0.261]])
    ])
    testset_cifar10 = torchvision.datasets.CIFAR10(
        root=datapath, train=False, download=True, transform=transform_test_cifar10)
    idx_test = get_indices(testset_cifar10, idxs)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(idx_test)
    testloader_cifar10 = torch.utils.data.DataLoader(
        testset_cifar10, batch_size=batch_size_test, shuffle=False, num_workers=num_worker, sampler=sampler_test)
    return testloader_cifar10