import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import glob
import os
import webdataset as wds

# CIFAR10 train
mean_cifar10_train = [x / 255 for x in [125.30691805, 122.95039414, 113.86538318]]
std_cifar10_train = [x / 255 for x in [62.99321928, 62.08870764, 66.70489964]]
# CIFAR10 test
mean_cifar10_test = [x / 255 for x in [126.02464141, 123.7085042,  114.85431865]]
std_cifar10_test = [x / 255 for x in [62.89639135, 61.93752718, 66.7060564 ]]

# CIFAR100 train
mean_cifar100_train = [x / 255 for x in [129.30416561, 124.0699627,  112.43405006]] 
std_cifar100_train = [x / 255 for x in [68.1702429,  65.39180804, 70.41837019]]
# CIFAR100 test
mean_cifar100_test = [x / 255 for x in [129.74308525, 124.28521836, 112.69526387]] 
std_cifar100_test = [x / 255 for x in [68.40415141, 65.62775279, 70.65942155]]

# ImageNet1k train
mean_imagenet1k_train = [0.485, 0.456, 0.406]
std_imagenet1k_train = [0.229, 0.224, 0.225]

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path, indices_path):
        self.x = torch.load(embeddings_path)
        self.y = torch.load(indices_path)
        self.targets = self.y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def mkdataset(args):
    if args.dataset == 'cifar10':
        data_train = torchvision.datasets.CIFAR10(
            root = args.data_root, train = True, download = True, transform = transform_cifar('cifar10')
        )
        data_test = torchvision.datasets.CIFAR10(
            root = args.data_root, train = False, download = True, transform = transform_cifar('cifar10')
        )
    elif args.dataset == 'cifar100':
        data_train = torchvision.datasets.CIFAR100(
            root = args.data_root, train = True, download = True, transform = transform_cifar('cifar100')
        )
        data_test = torchvision.datasets.CIFAR100(
            root = args.data_root, train = False, download = True, transform = transform_cifar('cifar100')
        )
    elif args.dataset == 'imagenet1k':
        local_data_train_path = "/data/ImageNet-WebDataset/train"
        num_train_shards = len(glob.glob(os.path.join(local_data_train_path, '*.tar')))
        urls_train = f"{local_data_train_path}/imagenet-train-{{000000..{num_train_shards-1:06d}}}.tar"
        data_train = wds.WebDataset(urls_train, shardshuffle=True)
        data_train = data_train.shuffle(1000)
        data_train = data_train.decode("pil")
        data_train = data_train.to_tuple("jpg", "cls")
        data_train = data_train.map_tuple(transform_imagenet1k('train'))

        local_data_test_path = "/data/ImageNet-WebDataset/val"
        num_test_shards = len(glob.glob(os.path.join(local_data_test_path, '*.tar')))
        urls_test = f"{local_data_test_path}/imagenet-val-{{000000..{num_test_shards-1:06d}}}.tar"
        data_test = wds.WebDataset(urls_test)
        data_test = data_test.decode("pil")
        data_test = data_test.to_tuple("jpg", "cls")
        data_test = data_test.map_tuple(transform_imagenet1k('test'))
    return data_train, data_test

def transform_cifar(type):
    if type == 'cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(), # 转为Tensor
                    transforms.Normalize(mean_cifar10_train, std_cifar10_train), # 归一化
                    ])
    else:
        transform = transforms.Compose([
                    transforms.ToTensor(), # 转为Tensor
                    transforms.Normalize(mean_cifar100_train, std_cifar100_train), # 归一化
                    ])

    return transform

def transform_imagenet1k(type):
    if type == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet1k_train, std=std_imagenet1k_train)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet1k_train, std=std_imagenet1k_train)
        ])
    return transform