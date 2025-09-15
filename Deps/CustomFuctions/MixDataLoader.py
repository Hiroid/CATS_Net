import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import Dataset
from PIL import Image
import glob
from tqdm import tqdm

datapath = './datafile'

class SelectiveImageFolder(Dataset):
    """
    Custom dataset that only loads specified class folders, avoiding the need for SubsetRandomSampler.
    Preloads all images into memory for maximum speed.
    """
    def __init__(self, root, class_names, transform=None):
        """
        Args:
            root: path to dataset root directory
            class_names: list of class folder names to include
            transform: optional transform to be applied on a sample
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Create class_to_idx mapping for selected classes only
        for idx, class_name in enumerate(sorted(class_names)):
            self.class_to_idx[class_name] = idx
        
        print(f"Loading images into memory")
        
        # Preload all images and labels for selected classes
        for class_name in tqdm(class_names):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                class_idx = self.class_to_idx[class_name]
                # Get all image files in this class folder
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPEG', '*.JPG']
                class_image_count = 0
                for ext in image_extensions:
                    image_paths = glob.glob(os.path.join(class_path, ext))
                    for img_path in image_paths:
                        # Load image immediately and store in memory
                        try:
                            image = Image.open(img_path).convert('RGB')
                            # Apply transform if provided
                            if self.transform:
                                image = self.transform(image)
                            self.samples.append((image, class_idx))
                            class_image_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to load image {img_path}: {e}")
                            continue
                # print(f"Loaded {class_image_count} images for class '{class_name}'")
        
        print(f"Total images loaded into memory: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        return image, label

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


def load_spose_embedding(embedding_path):
    """
    Load spose embedding vectors from text file
    Args:
        embedding_path: path to the spose_embedding_49d_sorted.txt file
    Returns:
        numpy array of shape (num_classes, embedding_dim)
    """
    embeddings = []
    with open(embedding_path, 'r') as f:
        for line in f:
            # Split by spaces and convert to float
            vec = [float(x) for x in line.strip().split()]
            embeddings.append(vec)
    return np.array(embeddings)


def get_things_class_names(dataset_path):
    """
    Get sorted class names from THINGS dataset directory
    Args:
        dataset_path: path to THINGS object_images directory
    Returns:
        list of class names sorted alphabetically
    """
    class_names = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    return sorted(class_names)


def get_things_indices(dataset, class_indices):
    """
    Get sample indices for specified classes in THINGS dataset
    Args:
        dataset: ImageFolder dataset
        class_indices: list of class indices to include
    Returns:
        list of sample indices
    """
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_indices:
            indices.append(i)
    return indices


def DLTrain_things_part(dataset_path, class_indices, batch_size_train, num_worker):
    """
    Create training dataloader for THINGS dataset with specified classes
    Args:
        dataset_path: path to THINGS object_images directory
        class_indices: list of class indices to include in training
        batch_size_train: batch size for training
        num_worker: number of workers
    Returns:
        DataLoader for training
    """
    transform_train_things = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset_things = ImageFolder(root=dataset_path, transform=transform_train_things)
    idx_train = get_things_indices(trainset_things, class_indices)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(idx_train)
    trainloader_things = torch.utils.data.DataLoader(
        trainset_things, batch_size=batch_size_train, shuffle=False, 
        num_workers=num_worker, sampler=sampler_train
    )
    return trainloader_things


def DLTrain_things_part_optimized(dataset_path, class_indices, batch_size_train, num_worker):
    """
    Optimized training dataloader for THINGS dataset that only loads specified classes
    Args:
        dataset_path: path to THINGS object_images directory
        class_indices: list of class indices to include in training
        batch_size_train: batch size for training
        num_worker: number of workers
    Returns:
        DataLoader for training
    """
    transform_train_things = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all class names and select the ones we need
    all_class_names = get_things_class_names(dataset_path)
    selected_class_names = [all_class_names[i] for i in class_indices]
    
    # Use optimized dataset that only loads selected classes
    trainset_things = SelectiveImageFolder(
        root=dataset_path, 
        class_names=selected_class_names, 
        transform=transform_train_things
    )
    
    trainloader_things = torch.utils.data.DataLoader(
        trainset_things, batch_size=batch_size_train, shuffle=True, 
        num_workers=num_worker
    )
    return trainloader_things


def DLTest_things_part(dataset_path, class_indices, batch_size_test, num_worker):
    """
    Create test dataloader for THINGS dataset with specified classes
    Args:
        dataset_path: path to THINGS object_images directory
        class_indices: list of class indices to include in testing
        batch_size_test: batch size for testing
        num_worker: number of workers
    Returns:
        DataLoader for testing
    """
    transform_test_things = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    testset_things = ImageFolder(root=dataset_path, transform=transform_test_things)
    idx_test = get_things_indices(testset_things, class_indices)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(idx_test)
    testloader_things = torch.utils.data.DataLoader(
        testset_things, batch_size=batch_size_test, shuffle=False,
        num_workers=num_worker, sampler=sampler_test
    )
    return testloader_things


def DLTest_things_part_optimized(dataset_path, class_indices, batch_size_test, num_worker):
    """
    Optimized test dataloader for THINGS dataset that only loads specified classes
    Args:
        dataset_path: path to THINGS object_images directory
        class_indices: list of class indices to include in testing
        batch_size_test: batch size for testing
        num_worker: number of workers
    Returns:
        DataLoader for testing
    """
    transform_test_things = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all class names and select the ones we need
    all_class_names = get_things_class_names(dataset_path)
    selected_class_names = [all_class_names[i] for i in class_indices]
    
    # Use optimized dataset that only loads selected classes
    testset_things = SelectiveImageFolder(
        root=dataset_path, 
        class_names=selected_class_names, 
        transform=transform_test_things
    )
    
    testloader_things = torch.utils.data.DataLoader(
        testset_things, batch_size=batch_size_test, shuffle=False,
        num_workers=num_worker
    )
    return testloader_things