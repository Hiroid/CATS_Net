import torch
import torchvision
import torchvision.transforms as transforms

datapath = './datafile'
def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_name:
            indices.append(i)
    return indices

def DLTrain(idxs, batch_size_train, num_worker, datapath=datapath):
    TrainLoader = []
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    for idx in idxs:
        idx_train = get_indices(trainset_cifar100, [idx])
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
        trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
        TrainLoader.append(trainloader_cifar100)
    return TrainLoader

def DLTrain_FixedDataLength(idxs, single_class_data_length, batch_size_train, num_worker, datapath=datapath):
    TrainLoader = []
    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    trainset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=True, download=True, transform=transform_train_cifar100)
    for idx in idxs:
        idx_train = get_indices(trainset_cifar100, [idx])
        idx_train_part = idx_train[0:single_class_data_length]
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train_part)
        trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=batch_size_train, shuffle=False, num_workers=num_worker, sampler=sampler_train)
        TrainLoader.append(trainloader_cifar100)
    return TrainLoader


def DLTest(idxs, batch_size_test, num_worker, datapath=datapath):
    TestLoader = []
    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    ])
    testset_cifar100 = torchvision.datasets.CIFAR100(
        root=datapath, train=False, download=True, transform=transform_test_cifar100)
    for idx in idxs:
        idx_test = get_indices(testset_cifar100, [idx])
        sampler_test = torch.utils.data.sampler.SubsetRandomSampler(idx_test)
        testloader_cifar100 = torch.utils.data.DataLoader(
            testset_cifar100, batch_size=batch_size_test, shuffle=False, num_workers=num_worker, sampler=sampler_test)
        TestLoader.append(testloader_cifar100)
    return TestLoader