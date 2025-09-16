import torch
import os
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
project_root = Path(__file__).resolve().parent.parent

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Extract features from ImageNet dataset.')
parser.add_argument('--data_root', type=str, default='/data', help='Root directory path for the ImageNet dataset.')
parser.add_argument('--model_name', '-mn', type=str, default='resnet50', 
                    choices=['resnet50', 'resnet18', 'vit_b_16'], 
                    help='Backbone model to use for feature extraction.')
parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], 
                    help='Dataset split to process.')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for data loading.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
args = parser.parse_args()
# --- End Argument Parsing ---

def create_feature_extractor(model_name):
    """Create feature extractor based on model name
    
    Args:
        model_name (str): Name of the backbone model
        
    Returns:
        torch.nn.Module: Feature extractor model
        int: Feature dimension
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.load_state_dict(
            torch.load(
                os.path.join(project_root, "Deps", "pretrained_fe", "resnet18-f37072fd.pth")
            )
        )
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_dim = 512
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.load_state_dict(
            torch.load(
                os.path.join(project_root, "Deps", "pretrained_fe", "resnet50-0676ba61.pth")
            )
        )
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
        
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
        model.load_state_dict(
            torch.load(
                os.path.join(project_root, "Deps", "pretrained_fe", "vit_b_16-c867db91.pth")
            )
        )
        # For ViT, we need to extract features before the final classifier
        feature_extractor = ViTFeatureExtractor(model)
        feature_dim = 768
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    feature_extractor.eval()
    return feature_extractor, feature_dim

class ViTFeatureExtractor(torch.nn.Module):
    """Custom feature extractor for Vision Transformer"""
    
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
    
    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

# Create feature extractor based on model name
feature_extractor, feature_dim = create_feature_extractor(args.model_name)
feature_extractor.to('cuda')

# ImageNet1k normalization parameters
mean_imagenet1k = [0.485, 0.456, 0.406]
std_imagenet1k = [0.229, 0.224, 0.225]

def get_transform(split='val'):
    """Get appropriate transform based on dataset split
    
    Args:
        split (str): Dataset split ('train' or 'val')
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet1k, std=std_imagenet1k)
        ])
    else:  # val
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet1k, std=std_imagenet1k)
        ])
    return transform

# Construct the full path using the arguments
dataset_path = os.path.join(args.data_root, 'ImageNet', args.split)
print(f"Loading dataset from: {dataset_path}")
print(f"Using model: {args.model_name}")
print(f"Feature dimension: {feature_dim}")

# Create dataset with appropriate transform
transform = get_transform(args.split)
dataset = ImageFolder(
    root=dataset_path,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=2,
)

print(f"Dataset size ({args.split}): {len(dataset)}")

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "Results", "FeatureData")
os.makedirs(output_dir, exist_ok=True)

# Extract features
print("Extracting features...")
embeddings = None
for X, y in tqdm(dataloader, desc=f"Processing {args.split} set with {args.model_name}"):
    X = X.to('cuda')
    with torch.no_grad():
        outputs = feature_extractor(X)
    
    # Flatten outputs for consistent shape across different models
    outputs = outputs.cpu().reshape(outputs.size(0), -1)
    
    if embeddings is None:
        embeddings = outputs
    else:
        embeddings = torch.cat([embeddings, outputs], dim=0)

print(f"Final embeddings shape: {embeddings.shape}")

# Construct output filenames based on model and split
output_embeddings_file = os.path.join(
    output_dir, 
    f"ImageNet1k_{args.split}_{args.model_name}_embeddings.pt"
)
output_indices_file = os.path.join(
    output_dir, 
    f"ImageNet1k_{args.split}_{args.model_name}_indices.pt"
)

print(f"Saving embeddings to: {output_embeddings_file}")
print(f"Saving indices to: {output_indices_file}")

# Save the extracted features and labels
torch.save(embeddings, output_embeddings_file)
torch.save(dataset.targets, output_indices_file)

print("Feature extraction completed successfully!")

