# CATS-Net Semantic Analysis Tools

This directory contains comprehensive analysis tools for evaluating CATS-Net models trained on ImageNet, focusing on semantic understanding, concept abstraction, and model interpretability.

## Overview

The semantic analysis tools provide multiple perspectives for understanding how CATS-Net processes and categorizes visual information:

- **Feature Extraction**: Extract deep features from ImageNet validation set
- **Accuracy Analysis**: Evaluate model performance across different configurations
- **Grad-CAM Visualization**: Generate class activation maps for interpretability
- **Hypercategory Classification**: Analyze performance on semantic groupings
- **Functional Entropy**: Measure concept abstraction effectiveness

## Terminology Mapping

| Paper Term          | Code Implementation                                      |
| :------------------ | :------------------------------------------------------- |
| CATS-Net            | `cats_net` (referenced from `A01_ImageNet.model`)        |
| CA (Concept Abstraction) Module | Layers prefixed with `cdp_` (e.g., `cdp_fc1`, `cdp_bn1`) |
| TS (Task-Solving) Module | Layers prefixed with `ts_` (e.g., `ts_fc1`, `ts_bn1`)   |
| Concept Vectors     | `symbol_set` (learnable parameter in `cats_net`)         |
| Feature Extractor   | `fe` (attribute within `cats_net`)                       |

## Prerequisites

- Python 3.9.16
- PyTorch 2.0.1
- CUDA 11.7
- Torchvision 0.15.2
- Additional packages: `tqdm`, `matplotlib`, `seaborn`, `pandas`, `scipy`, `nltk`, `PIL`
- Trained CATS-Net models (from `../Results/param/`)
- ImageNet validation dataset

## Directory Structure

-   `B01_Get_FeatureData.py`: Extract ImageNet features using ResNet50 backbone.
-   `B02_Get_acc_list.py`: Evaluate accuracy across multiple trained models and perform class-wise analysis.
-   `B03_Grad-CAM.ipynb`: Generate Grad-CAM visualizations for model interpretability.
-   `B04_ImageNet_hypercategories.ipynb`: Analyze semantic hypercategories using WordNet taxonomy.
-   `B05_functional_entropy.ipynb`: Measure functional entropy of concept abstraction.
-   `classes.py`: ImageNet class definitions and synset mappings.
-   `../Deps/`: Parent directory where pretrained models and sample images are stored.
    -   `../Deps/pretrained_fe/resnet50-0676ba61.pth`: Pretrained ResNet50 feature extractor weights.
    -   `../Deps/CAM_fig/`: Sample images for Grad-CAM visualization.
-   `../Results/`: Parent directory where analysis outputs are saved.
    -   `../Results/FeatureData/`: Feature embeddings and class indices.
    -   `../Results/accuracy_list/`: Model performance metrics and class-wise accuracies.
    -   `../Results/hypercategory/`: Semantic category mappings.
    -   `../Results/entropy_stat/`: Functional entropy analysis results.

## Core Scripts

### 1. Feature Extraction (`B01_Get_FeatureData.py`)

Extracts deep features from ImageNet validation set using a pretrained ResNet50 backbone.

**Usage:**
```bash
python B01_Get_FeatureData.py --data_root /path/to/datasets
```

**Key Arguments:**
- `--data_root`: Root directory path for ImageNet dataset (default: `/data0/share/datasets`, change as needed)

**Outputs:**
- `../Results/FeatureData/ImageNet1k_test_embeddings.pt`: Feature embeddings (N×2048)
- `../Results/FeatureData/ImageNet1k_test_indices.pt`: Corresponding class labels

**Function:**
- Loads pretrained ResNet50 from `../Deps/pretrained_fe/resnet50-0676ba61.pth`
- Applies standard ImageNet preprocessing transformations
- Extracts features from the penultimate layer (before final classification)
- Processes entire validation set with batch size 512

### 2. Accuracy Evaluation (`B02_Get_acc_list.py`)

Systematically evaluates CATS-Net model accuracy across multiple trained models and class-wise performance.

**Usage:**
```bash
python B02_Get_acc_list.py [--random_symbol_set]
```

**Key Arguments:**
- `--random_symbol_set`: Initialize symbol_set with random values for ablation study

**Function:**
- Evaluates 30 consecutive model checkpoints (`trail21.pt` to `trail50.pt`)
- Computes overall test accuracy for each model
- Performs class-wise accuracy analysis by excluding each of 1000 classes
- Supports ablation studies with randomized concept vectors

**Outputs:**
- `../Results/accuracy_list/acc_list_imagenet1k_ss20_fixfe_alltrails.pt`: Overall accuracies
- `../Results/accuracy_list/acc_list_imagenet1k_ss20_fixfe_trail{X}.pt`: Class-wise accuracies
- `../Results/accuracy_list/acc_list_random_imagenet1k_ss20_fixfe_*.pt`: Randomized baselines

## Analysis Notebooks

### 3. Grad-CAM Visualization (`B03_Grad-CAM.ipynb`)

Generates gradient-weighted class activation maps to visualize which image regions contribute most to concept-specific predictions.

**Key Features:**
- Implements Grad-CAM for CATS-Net architecture
- Incorporates concept abstraction gating mechanism
- Visualizes attention maps for specific concept-class pairs
- Supports custom image inputs and target class selection

**Example Usage:**
- Load sample images from `../Deps/CAM_fig/`
- Select target concepts (e.g., Tench, Lifeboat, Sunglasses)
- Generate heatmap overlays showing model attention

### 4. Hypercategory Analysis (`B04_ImageNet_hypercategories.ipynb`)

Analyzes ImageNet classes according to semantic hypercategories using WordNet taxonomy.

**Key Features:**
- Utilizes NLTK WordNet for semantic hierarchy analysis
- Groups 1000 ImageNet classes into meaningful categories:
  - `mammal`: Mammalian animals (218 classes)
  - `others_animal`: Other animals (180 classes)  
  - `instrumentality`: Tools and instruments (350 classes)
  - `others_artifact`: Other human-made objects (172 classes)
  - `others_entity`: Remaining entities (80 classes)

**Outputs:**
- `../Results/hypercategory/imagenet1k_hypercategory_v2.pt`: Class-to-category mapping

### 5. Functional Entropy Analysis (`B05_functional_entropy.ipynb`)

Measures the functional entropy of concept abstraction by comparing trained models with random baselines.

**Key Features:**
- Evaluates concept abstraction effectiveness through entropy metrics
- Compares trained CATS-Net against randomized configurations
- Generates statistical distributions for concept utilization
- Supports ablation studies with random concept vectors and random task-solving modules

**Analysis Types:**
- **Trained Configuration**: Uses learned concept vectors with random assignments
- **Random TS-Net**: Completely randomized task-solving network
- Statistical comparison of concept utilization patterns

## Support Files

### `classes.py`
Contains comprehensive ImageNet class definitions:
- `IMAGENET2012_CLASSES`: Ordered dictionary mapping synset IDs to class names
- Complete mapping for all 1000 ImageNet classes
- Compatible with WordNet semantic analysis

## Workflow Example

1. **Extract Features:**
   ```bash
   python B01_Get_FeatureData.py --data_root /path/to/imagenet
   ```

2. **Evaluate Model Performance:**
   ```bash
   python B02_Get_acc_list.py
   python B02_Get_acc_list.py --random_symbol_set  # Ablation study
   ```

3. **Run Analysis Notebooks:**
   - Execute `B03_Grad-CAM.ipynb` for interpretability analysis
   - Execute `B04_ImageNet_hypercategories.ipynb` for semantic grouping
   - Execute `B05_functional_entropy.ipynb` for concept abstraction analysis

## Prerequisites for Analysis

**Required Model Files:**
- Trained CATS-Net checkpoints: `../Results/param/imagenet1k_ss20_fixfe_trail{21-50}.pt`
- Pretrained ResNet50: `../Deps/pretrained_fe/resnet50-0676ba61.pth`

**Required Data:**
- ImageNet validation set at specified `data_root`
- Sample images for Grad-CAM: `../Deps/CAM_fig/fig1.JPEG`, `../Deps/CAM_fig/fig2.JPEG`

**Output Directories:**
- `../Results/FeatureData/`: Feature embeddings and labels
- `../Results/accuracy_list/`: Model performance metrics  
- `../Results/hypercategory/`: Semantic category mappings
- `../Results/entropy_stat/`: Functional entropy statistics

## GPU Requirements

All scripts are configured to use CUDA-enabled GPUs. GPU device selection is controlled via:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust as needed
```

The analysis tools are optimized for modern GPUs with sufficient memory for batch processing of ImageNet-scale datasets. 