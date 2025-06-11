import torch
import os
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Extract features from ImageNet dataset.')
parser.add_argument('--data_root', type=str, default='/data0/share/datasets',
                    help='Root directory path for the ImageNet dataset.')
args = parser.parse_args()
# --- End Argument Parsing ---

resnet50 = models.resnet50(weights=None)
resnet50.load_state_dict(torch.load('../Deps/pretrained_fe/resnet50-0676ba61.pth'))
resnet50_fe1 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50_fe1.eval()
resnet50_fe1.to('cuda')

# ImageNet1k train
mean_imagenet1k_train = [0.485, 0.456, 0.406]
std_imagenet1k_train = [0.229, 0.224, 0.225]

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_imagenet1k_train, std=std_imagenet1k_train)
])

# Construct the full path using the arguments
dataset_path = os.path.join(args.data_root, 'ImageNet', 'val')
print(f"Loading dataset from: {dataset_path}")

dataset = ImageFolder(
    root = dataset_path,
    transform = transform_val
)

dataloader = DataLoader(
    dataset,
    batch_size = 512,
    num_workers = 8,
    shuffle = False,
    pin_memory = True,
    prefetch_factor = 2,
)

print(f"Dataset size ({args.split}): {len(dataset)}")

ImageNet1k_embeddings = None
for X, y in tqdm(dataloader):
    X = X.to('cuda')
    with torch.no_grad():
        outputs = resnet50_fe1(X)
    outputs = outputs.cpu().reshape(outputs.size(0), -1)
    if ImageNet1k_embeddings == None:
        ImageNet1k_embeddings = outputs
    else:
        ImageNet1k_embeddings = torch.cat([ImageNet1k_embeddings, outputs], dim = 0)

print(ImageNet1k_embeddings.shape)

# Construct output filenames based on the split
output_embeddings_file = f'../Results/FeatureData/ImageNet1k_test_embeddings.pt'
output_indices_file = f'../Results/FeatureData/ImageNet1k_test_indices.pt'

print(f"Saving embeddings to: {output_embeddings_file}")
print(f"Saving indices to: {output_indices_file}")

torch.save(ImageNet1k_embeddings, output_embeddings_file)
torch.save(dataset.targets, output_indices_file)

