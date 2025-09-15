# Word2Vec-based Context Learning on CIFAR-100
This directory contains the code for training and testing Word2Vec-based context learning models on the CIFAR-100 dataset.

## Overview

The Word2Vec Context Learning module implements a system that leverages pre-trained Word2Vec embeddings as semantic context vectors for CIFAR-100 image classification. The approach combines visual features from CNN models with semantic word embeddings to enhance classification performance through contextual learning.

## Terminology Mapping

To help readers connect the concepts from the paper to the code implementation, here is a brief mapping:

| Paper Term              | Code Implementation                                      |
| :---------------------- | :------------------------------------------------------- |
| CATS-Net            | `Net2` in `SEAnet.py`                               |
| CA (Concept Abstraction) Module | Layers prefixed with `cdp_` (e.g., `cdp_fc1`, `cdp_bn1`) in `SEAnet.py` |
| TS (Task-Solving) Module | Layers prefixed with `clf_` (e.g., `clf_fc1`, `clf_fc2`) in `SEAnet.py` |
| Feature Extractor   | `pretrained_classifier_cnn` (ResNet18)                 |
| Word2Vec Context (Concept) Vectors| `name_vecs` loaded from `wordvector/embedding_dim_20.mat`, (keep fixed in training script) |
| Binary Classification   | Distinguishing positive/negative class-context pairs    |
| Noise Injection         | Adding Gaussian noise to context vectors during training |


## Prerequisites

- Python 3.11.11
- PyTorch 2.4.0
- CUDA 11.8
- NumPy 1.26.4
- SciPy 1.14.0
- Matplotlib (for visualization)
- MATLAB (for statistical analysis)
- An environment with CUDA enabled GPUs is recommended. Set the desired GPU using `os.environ["CUDA_VISIBLE_DEVICES"] = "4"` in the code.

## Directory Structure

- `B01_Word2Vec_CIFAR100_main.py`: Main training and testing script.
- `B02_cifar100_word2vec_acc_ttest.m`: MATLAB script for statistical significance testing.
- `wordvector/`: Directory containing Word2Vec embedding files.
  - `embedding_dim_20.mat`: Pre-trained Word2Vec embeddings with dimension 20.
- `datafile/`: Directory for CIFAR-100 dataset files.
- `ni=X.XXe-XX/`: Output directories for different noise intensity configurations.
  - `contexts/`: Saved initial context vectors for each class.
  - `checkpoint/`: Saved model checkpoints.
  - `recode_id_*.log`: Training progress logs.
  - `final_test_results.log`: Final testing results.
- `../../Deps/CustomFuctions/`: Parent directory containing custom modules.
  - `MixDataLoader.py`: Data loader for mixed training data.
  - `SeparatedDataLoader.py`: Data loader for separated test data.
  - `SEAnet.py`: Contains the extended network model definition.
  - `AccracyTest.py`: Accuracy testing utilities.

## Training

To train the Word2Vec context learning model, run the main script:

```bash
python B01_Word2Vec_CIFAR100_main.py [arguments]
```

**Key Arguments:**

- `--device`: Device type (`cuda` or `cpu`, default: `cuda`)
- `--worker`: Number of workers for data loader (default: 0)
- `--batch_size_train`: Training batch size (default: 100)
- `--batch_size_test`: Testing batch size (default: 100)
- `--num_class`: Number of total classes (default: 100)
- `--test_id`: List of class IDs used for testing (default: range(100))
- `--start_epoch`: Starting epoch number (default: 0)
- `--end_epoch`: Ending epoch number (default: 1000)
- `--context_dim`: Context dimension (default: 20)
- `--noise_intensity`: List of noise intensities for context vectors (default: [0.05])
- `--train_mode`: Training mode flag (True for training, False for testing, default: False)
- `--drop_probility`: Dropout probability for CDP network (default: 0.2)
- `--num_process`: Number of parallel processes (default: 10)
- `--wordvec_path`: Path to Word2Vec embedding file (default: `./wordvector/embedding_dim_20.mat`)

**Example Usage:**

```bash
# Training mode with multiprocessing
python B01_Word2Vec_CIFAR100_main.py --train_mode 1 --num_process 10

# Testing mode
python B01_Word2Vec_CIFAR100_main.py --train_mode 0

# Custom training with specific parameters
python B01_Word2Vec_CIFAR100_main.py \
    --train_mode True \
    --context_dim 20 \
    --noise_intensity 0.05 \
    --end_epoch 500 \
    --batch_size_train 128

python B01_THINGS_main.py \
    --train_mode True \
    --noise_intensity 0.1 \
    --end_epoch 100 \
```

## Testing

To test trained models, run the testing script:

```bash
python B01_Word2Vec_CIFAR100_main.py --train_mode False
```

For statistical analysis, use the MATLAB script:

```matlab
run('B02_cifar100_word2vec_acc_ttest.m')
```

## Model Architecture

The Word2Vec Context Learning system consists of:

1. **Feature Extractor**: Pre-trained ResNet18 (with final fully-connected layer replaced by identity)
2. **CA Component**: Fully Connected Layers (NOT VGG11-based, although defined in `B01_Word2Vec_CIFAR100_main.py` Line `54`, `174` and `223` but didn't use) concept vector processing module
3. **TS Component**: Fully Connected Layers
4. **Context Vectors**: Fixed Word2Vec embedding vectors for each class
5. **Binary Classification Head**: Final layer for positive/negative classification

## Training Process

The training process involves:

1. **Context Initialization**: Load Word2Vec embeddings and scale by factor of 10
2. **Leave-One-Out Training**: For each test class, train on remaining 99 classes
3. **Contrastive Learning**: Use positive and negative class-context pairs
4. **Noise Injection**: Add Gaussian noise to context vectors during training
5. **Dynamic Negative Sampling**: Randomly select negative contexts during training
6. **Model Checkpointing**: Save best models based on combined positive/negative accuracy

### Training Loop Details:
- For each class ID, create separate training/testing splits
- Initialize contexts with Word2Vec embeddings multiplied by 10
- Train with binary cross-entropy loss on positive/negative pairs
- Add random noise with specified intensity to context vectors
- Evaluate every 2 epochs and save best performing models
- Support multiprocessing for parallel training across classes

## Testing Process

The testing process includes:

1. **Model Loading**: Load trained checkpoints for each test class
2. **Context Restoration**: Restore saved context vectors and class mappings
3. **Accuracy Evaluation**: Test on both positive and negative samples
4. **Statistical Analysis**: Perform one-sample t-tests against threshold (0.7)
5. **Results Logging**: Save detailed evaluation results and statistics

## Output Files

### Training Phase:
- **Model Checkpoints**: Complete model state including network, contexts, and optimizer
- **Context Vectors**: Initial context vectors saved in MATLAB format
- **Training Logs**: Epoch-wise loss and accuracy progression
- **Directory Structure**: Organized by noise intensity levels

### Testing Phase:
- **Final Test Results**: Comprehensive accuracy analysis for each class
- **Statistical Tests**: One-sample t-test results with effect sizes
- **Summary Statistics**: Mean accuracy and standard deviation across classes

### Accuracy Data for Plot:
- Results compatible with overall analysis pipeline
- Statistical significance testing at α = 0.05 level

## Performance Metrics

The system evaluates performance using:

1. **Positive Accuracy**: Classification accuracy on correct class-context pairs
2. **Negative Accuracy**: Classification accuracy on incorrect class-context pairs  
3. **Combined Accuracy**: Average of positive and negative accuracies
4. **Statistical Significance**: One-sample t-tests against performance threshold (0.7)
5. **Effect Size**: Cohen's d for practical significance assessment

## Notes

- The system requires pre-trained Word2Vec embeddings for CIFAR-100 class names
- Training uses leave-one-out cross-validation across all 100 classes
- GPU usage is controlled via `CUDA_VISIBLE_DEVICES` environment variable (set to "4")
- Multiprocessing support enables parallel training across different test classes
- Context vectors are initialized with Word2Vec embeddings scaled by factor 10
- Statistical analysis includes significance testing with effect size calculations
- Results are automatically organized by noise intensity levels
- The system supports batch processing with configurable noise intensities
- Best models are selected based on combined positive/negative accuracy performance 