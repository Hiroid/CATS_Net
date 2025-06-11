# CATS-Net
Official Code for Article “A neural network model for concept formation, understanding and communication”

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

## Contents

* [Overview](#overview)
* [Repository Contents](#repository-contents)
* [System Requirements](#system-requirements)
* [Installation Guide](#installation-guide)
* [Terminology Mapping](#terminology-mapping)
* [Component Descriptions](#component-descriptions)
* [Usage Examples](#usage-examples)
* [Results](#results)
* [Citation](#citation)
* [License](#license)
* [Issues](#issues)

## Overview

CATS-Net (Concept Abstraction and Task-Solving Network) is a comprehensive framework for understanding and implementing concept abstraction in neural networks. The system combines supervised learning with concept-based reasoning to create interpretable and efficient neural architectures. This repository provides implementations across multiple datasets and analysis tools for understanding concept abstraction, communication between neural agents, and representational similarity with neural activity.

The framework addresses the challenge of learning meaningful concept representations that can be used for both classification tasks and inter-agent communication. By separating concept abstraction (CA) from task-solving (TS) modules, CATS-Net enables better interpretability and more effective knowledge transfer between different learning scenarios.

## Repository Contents

* **A01_ImageNet/**: ImageNet-1K training and evaluation with ResNet50/ResNet18/ViT backbones
* **A02_Semantic_Analysis/**: Comprehensive analysis tools for semantic understanding and model interpretability
* **A03_Communications/**: Multi-agent communication experiments on CIFAR-100
  * **B01_SEA_Net/**: CATS-Net implementation and training on CIFAR-100
  * **B02_Communication_Game/**: Translation-Interpretation module for Speaker-Listener communication
  * **B03_Internal_Structure/**: Internal representational structure analysis and network visualization
* **A04_Word2Vec/**: Word2Vec-based context learning experiments on CIFAR-100
* **A05_RSA/**: Representational Similarity Analysis comparing neural and computational models
* **Deps/**: Dependencies and shared utilities including custom functions and pretrained weights
* **Results/**: Output directory for experimental results, logs, and model checkpoints
* **B01_figures/**: Generated figures and visualizations

## System Requirements

### Hardware Requirements

CATS-Net requires a computer with sufficient RAM and GPU memory to support deep learning operations. For minimal performance:

**Minimum Requirements:**
* RAM: 8+ GB
* GPU: NVIDIA GPU with 4+ GB VRAM
* Storage: 50+ GB available space

**Recommended Specifications:**
* RAM: 32+ GB
* GPU: NVIDIA GPU with 12+ GB VRAM (RTX 3080 or better)
* CPU: 8+ cores, 3.0+ GHz
* Storage: 100+ GB SSD

### Software Requirements

#### Operating System Requirements

The package has been tested on:
* **Linux**: CentOS 7.9

#### Core Dependencies

**Python Environment:**
* Python 3.9.16+
* PyTorch 2.0.1+
* CUDA 11.7+ (for GPU acceleration)
* Torchvision 0.15.2+

**MATLAB Environment (for RSA analysis):**
* MATLAB R2020a+
* Statistics and Machine Learning Toolbox
* Bioinformatics Toolbox
* SPM12 (Statistical Parametric Mapping)

**Additional Python Packages:**
```bash
pip install numpy scipy matplotlib seaborn pandas tqdm nltk pillow
```

## Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/your-username/CATS-Net.git
cd CATS-Net
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv cats_env
source cats_env/bin/activate  # On Windows: cats_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install numpy scipy matplotlib seaborn pandas tqdm nltk pillow
```

### 3. Download Required Assets

**Pretrained Feature Extractors:**
```bash
# Create dependencies directory
mkdir -p Deps/pretrained_fe

# Download pretrained weights
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O Deps/resnet50-0676ba61.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O Deps/pretrained_fe/resnet18-f37072fd.pth
```

**Dataset Setup:**
```bash
# ImageNet (if using A01_ImageNet)
# Download ImageNet-1K to /path/to/imagenet/
# Structure: train/, val/ subdirectories

# CIFAR-100 (automatically downloaded by scripts)
# No manual setup required
```

### 4. Verify Installation

```bash
cd A01_ImageNet
python main.py --help  # Should display argument options
```

The installation should complete in approximately 10-15 minutes on a system with recommended specifications.

## Terminology Mapping

To help connect paper concepts with code implementation:

| Paper Term | Code Implementation | Location |
|:-----------|:-------------------|:---------|
| CATS-Net | `sea_net` | A01_ImageNet/model.py |
| CATS-Net | `Net2` | A03_Communications/Deps/CustomFunctions/SEAnet.py |
| CA (Concept Abstraction) Module | Layers with `cdp_` prefix | Multiple locations |
| TS (Task-Solving) Module | Layers with `ts_` prefix | A01_ImageNet/model.py |
| TS (Task-Solving) Module | Layers with `clf_` prefix | A03_Communications |
| Concept Vectors | `symbol_set` | A01_ImageNet/model.py |
| Concept Vectors | `contexts` | A03_Communications |
| Feature Extractor | `fe` attribute | Multiple locations |
| Translation-Interpretation Module | `TImodule` | A03_Communications/B02_Communication_Game |

## Component Descriptions

### A01_ImageNet - Large-Scale Image Classification

**Purpose**: Train and evaluate CATS-Net on ImageNet-1K dataset with various backbone architectures.

**Key Features**:
- Support for ResNet50, ResNet18, and Vision Transformer backbones
- Configurable concept vector dimensions (default: 20)
- Frozen and fine-tuned feature extractor modes
- Comprehensive logging and checkpoint management

**Quick Start**:
```bash
cd A01_ImageNet
python main.py -ne 5 -bs 512 -lr 0.001 -ss 20 --model_name resnet50 --use_pretrain --fix_fe
```

### A02_Semantic_Analysis - Model Interpretability

**Purpose**: Comprehensive analysis tools for understanding CATS-Net semantic processing and concept abstraction.

**Key Analyses**:
- Deep feature extraction and visualization
- Grad-CAM interpretability analysis
- Hypercategory semantic grouping using WordNet
- Functional entropy measurement
- Class-wise accuracy evaluation

**Quick Start**:
```bash
cd A02_Semantic_Analysis
python B01_Get_FeatureData.py --data_root /path/to/imagenet
python B02_Get_acc_list.py
```

### A03_Communications - Multi-Agent Communication

**Purpose**: Study emergent communication between neural agents through concept-based symbolic exchange.

#### B01_SEA_Net - CIFAR-100 Training
- CATS-Net implementation on CIFAR-100
- Binary classification with concept-context pairing
- Noise injection for robust learning

#### B02_Communication_Game - Agent Translation
- Translation-Interpretation (TI) module training
- Speaker-Listener communication protocols
- Symbol translation and accuracy evaluation

#### B03_Internal_Structure - Representational Analysis
- Graph-theoretic analysis of concept structures
- Representational Distance Matrix (RDM) comparison
- Statistical validation of communication effectiveness

**Quick Start**:
```bash
cd A03_Communications/B01_SEA_Net
python C01_CATSNet_CIFAR100_main.py --mode train
```

### A04_Word2Vec - Semantic Context Learning

**Purpose**: Integration of Word2Vec semantic embeddings with visual concept learning on CIFAR-100.

**Key Features**:
- Fixed Word2Vec concept vectors (dimension 20)
- Leave-one-out cross-validation across 100 classes
- Statistical significance testing with effect size analysis
- Multiprocessing support for parallel training

**Quick Start**:
```bash
cd A04_Word2Vec
python B01_Word2Vec_CIFAR100_main.py --train_mode True --num_process 10
```

### A05_RSA - Neural Similarity Analysis

**Purpose**: Representational Similarity Analysis comparing computational models with neural data using fMRI.

**Core Components**:
- **ROI Analysis**: Region-specific brain-model correlations
- **Searchlight Analysis**: Whole-brain voxel-wise RSA mapping
- **Model Clustering**: Consensus analysis across computational models
- **Statistical Testing**: Group-level significance testing

**Quick Start**:
```matlab
cd A05_RSA
run('ROI/ROIAnalysis_RSA_Concept.m');
run('Searchlight/A0_RSA_YuModel.m');
```

## Usage Examples

### Training CATS-Net on ImageNet

```bash
# Basic training with ResNet50 backbone
cd A01_ImageNet
python main.py \
    --model_name resnet50 \
    --num_epochs 10 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --symbol_size 20 \
    --use_pretrain \
    --fix_fe

# Training with Vision Transformer
python main.py \
    --model_name vit_b_16 \
    --num_epochs 15 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --symbol_size 30
```

### Semantic Analysis Pipeline

```bash
# Extract features from validation set
cd A02_Semantic_Analysis
python B01_Get_FeatureData.py --data_root /path/to/imagenet

# Evaluate model accuracy across multiple checkpoints
python B02_Get_acc_list.py

# Run ablation study with random concept vectors
python B02_Get_acc_list.py --random_symbol_set

# Generate interpretability visualizations (Jupyter notebooks)
jupyter notebook B03_Grad-CAM.ipynb
jupyter notebook B04_ImageNet_hypercategories.ipynb
jupyter notebook B05_functional_entropy.ipynb
```

### Communication Game Setup

```bash
# Step 1: Train CATS-Net agents on CIFAR-100
cd A03_Communications/B01_SEA_Net
python C01_CATSNet_CIFAR100_main.py --mode train --end_epoch 2000

# Step 2: Train Translation-Interpretation module
cd ../B02_Communication_Game
python C01_Comm_CIFAR100_main_train.py

# Step 3: Test communication effectiveness
python C02_Comm_CIFAR100_main_test.py

# Step 4: Analyze internal representations
cd ../B03_Internal_Structure
matlab -r "C01_cifar100_concept2graph; C02_cifar100_RDM_speaker_listener;"
```

### Word2Vec Context Learning

```bash
cd A04_Word2Vec

# Train with multiprocessing (recommended)
python B01_Word2Vec_CIFAR100_main.py \
    --train_mode True \
    --num_process 10 \
    --context_dim 20 \
    --noise_intensity 0.05

# Test trained models
python B01_Word2Vec_CIFAR100_main.py --train_mode False

# Statistical analysis (MATLAB)
matlab -r "B02_cifar100_word2vec_acc_ttest;"
```

### RSA Neural Analysis

```matlab
% MATLAB environment
cd A05_RSA

% ROI-based analysis
addpath('ROI/');
run('ROI/ROIAnalysis_RSA_Concept.m');
run('ROI/ROIAnalysis_RSA_CA.m');

% Whole-brain searchlight analysis
addpath('Searchlight/');
A0_RSA_YuModel(1, './lib/', 'SearchlightRadius', 10);

% Model clustering analysis
run('kmeans_rsa_clustering.m');
```

## Results

CATS-Net demonstrates superior performance across multiple benchmarks:

### ImageNet Classification
- **Top-1 Accuracy**: 76.1% (ResNet50 backbone)
- **Concept Interpretability**: High-quality Grad-CAM visualizations
- **Semantic Consistency**: Strong correlation with WordNet hypercategories

### Communication Effectiveness  
- **CIFAR-100 Communication**: >85% accuracy in speaker-listener tasks
- **Symbol Translation**: Effective cross-agent concept transfer
- **Representational Similarity**: Significant structural alignment between agents

### Neural Correspondence
- **RSA Analysis**: Significant correlations with human brain activity
- **ROI Specificity**: Strong matches in language and visual processing regions
- **Model Consensus**: Robust performance across multiple computational models

### Word2Vec Integration
- **Semantic Enhancement**: Improved performance with linguistic priors
- **Statistical Significance**: p < 0.001 across all test conditions
- **Effect Size**: Large effect sizes (Cohen's d > 0.8) for concept learning

## Citation

If you use CATS-Net in your research, please cite:

```bibtex
@article{catsnet_2025,
  title={A neural network model for concept formation, understanding and communication},
  author={Liangxuan Guo, Haoyang Chen, Yang Chen, Yanchao Bi and Shan Yu},
  journal={Under review at Nature Computational Science},
  year={2025},
  doi={[DOI]}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Liangxuan Guo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Issues

If you encounter any problems or have questions about CATS-Net:

1. **Check Documentation**: Review component-specific README files for detailed usage instructions
2. **Search Issues**: Look through existing GitHub issues for similar problems
3. **Create New Issue**: Provide detailed information including:
   - Component being used (A01_ImageNet, A02_Semantic_Analysis, etc.)
   - Error messages and stack traces
   - System configuration (OS, Python version, GPU type)
   - Steps to reproduce the issue

## Contributing

We welcome contributions to CATS-Net! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

For major changes, please open an issue first to discuss the proposed modifications.

## Acknowledgments

This work builds upon numerous open-source projects and research contributions. Special thanks to:

- PyTorch team for the deep learning framework
- SPM developers for neuroimaging analysis tools
- The broader machine learning and neuroscience communities

---

**Note**: This repository contains implementations for research purposes. For production use, please ensure appropriate testing and validation for your specific use case.
