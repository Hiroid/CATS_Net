# CATS-Net on CIFAR-100
This directory contains the code for training and evaluating CATS-Net models (named as `sea_net` in the code) on the CIFAR-100 dataset.

## Terminology Mapping

To help readers connect the concepts from the paper to the code implementation, here is a brief mapping:

| Paper Term          | Code Implementation                                      |
| :------------------ | :------------------------------------------------------- |
| CATS-Net            | `Net2` in `CATSnet.py`                               |
| CA (Concept Abstraction) Module | Layers prefixed with `cdp_` (e.g., `cdp_fc1`, `cdp_bn1`) in `CATSnet.py` |
| TS (Task-Solving) Module | Layers prefixed with `clf_` (e.g., `clf_fc1`, `clf_fc2`) in `CATSnet.py` |
| Concept Vectors     | `contexts` (learnable parameter in training script)     |
| Feature Extractor   | `pretrained_classifier_cnn` (ResNet18)                 |

## Prerequisites

- Python 3.11.11
- PyTorch 2.4.0
- CUDA 11.8
- Torchvision 0.19.0
- NumPy 1.26.4
- SciPy 1.14.0
- An environment with CUDA enabled GPUs is recommended. Set the desired GPU using `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` in the code.

## Directory Structure

- `C01_CATSNet_CIFAR100_main.py`: Main script for training and testing.
- `datafile/`: Directory containing CIFAR-100 dataset files.
  - `cifar-100-python/`: CIFAR-100 dataset in Python format.
- `ni=X.XXe-XX_r=X/`: Output directories for different noise intensity and round configurations.
  - `contexts/`: Saved context vectors at different training epochs.
  - `checkpoint/`: Saved model checkpoints.
  - `Train_results.log`: Training loss and accuracy logs.
  - `Test_Results.log`: Testing results logs.
- `../../Deps/CustomFuctions/`: Parent directory containing custom modules.
  - `MixDataLoader.py`: Data loader for mixed training data.
  - `SeparatedDataLoader.py`: Data loader for separated test data.
  - `CATSnet.py`: Contains the CATS-Net model definition.
  - `models/`: Additional model definitions.
- `../../Deps/pretrained_fe/`: Directory for pretrained feature extractor weights.
  - `resnet18-f37072fd.pth`: ResNet18 pretrained weights.

## Training

To train a model, run the main script:

```bash
python C01_CATSNet_CIFAR100_main.py [arguments]
```

**Key Arguments:**

- `--device`: Device type (`cuda` or `cpu`, default: `cuda`)
- `--worker`: Number of workers for data loader (default: 1)
- `--batch_size_train`: Training batch size (default: 200)
- `--batch_size_test`: Testing batch size (default: 100)
- `--num_class`: Number of classes (default: 100)
- `--start_epoch`: Starting epoch ID (default: 0)
- `--end_epoch`: Ending epoch ID (default: 2000)
- `--context_dim`: Context dimension (default: 20)
- `--noise_intensity`: List of noise intensities for context vectors (default: [1e-1])
- `--epoch_node`: Epochs to record results (default: [200, 500, 800, 1000, 1500, 1999])
- `--mode`: Operation mode (`train` or `test`, default: `train`)

**Example Usage:**

```bash
# Training mode
python C01_CATSNet_CIFAR100_main.py --mode train

# Testing mode
python C01_CATSNet_CIFAR100_main.py --mode test
```

## Model Architecture

The CATS-Net model consists of:

1. **Feature Extractor**: Pre-trained ResNet18 (with final fully-connected layer replaced by identity)
2. **CA Component**: Fully Connected Layers (NOT VGG11-based, although defined in `C01_CATSNet_CIFAR100_main.py` Line `48` and `159` but didn't use) concept vector processing module
3. **TS Component**: Fully Connected Layers
4. **Context Vectors**: Learnable embedding vectors for each class
5. **Binary Classification Head**: Final layer for positive/negative classification

## Training Process

The training process involves:

1. **Context Learning**: Learnable context vectors are optimized alongside the network
2. **Alternating Optimization**: Network parameters and context vectors are updated alternately
3. **Noise Injection**: Random noise is added to context vectors during training
4. **Binary Classification**: The model learns to distinguish between correct and incorrect class-context pairs

## Output Files

During training, the script saves:

1. **Context Vectors**: Saved at specified epochs in `.mat` format
2. **Model Checkpoints**: Complete model state saved at the end of training
3. **Training Logs**: Loss and accuracy metrics for each epoch
4. **Test Logs**: Evaluation results on positive and negative samples

## Testing

In test mode, the script:

1. Loads the latest trained context vectors and model checkpoint
2. Evaluates the model on both positive (correct) and negative (incorrect) class-context pairs
3. Reports accuracy for both positive and negative classifications

## Notes

- The model uses ResNet18 as the feature extractor, which should be pre-downloaded
- Training alternates between optimizing network parameters and context vectors
- The system supports multiple noise intensity levels for robust training
- Results are automatically logged and saved for analysis
- GPU usage is controlled via `CUDA_VISIBLE_DEVICES` environment variable 