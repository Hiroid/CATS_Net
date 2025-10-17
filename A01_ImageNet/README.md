# CATS-Net on ImageNet
This directory contains the code for training and evaluating CATS-Net models (named as `cats_net` in the code) on the ImageNet dataset.
## Terminology Mapping

To help readers connect the concepts from the paper to the code implementation, here is a brief mapping:

| Paper Term          | Code Implementation                                      |
| :------------------ | :------------------------------------------------------- |
| CATS-Net            | `cats_net` in `model.py`             |
| CA (Concept Abstraction) Module | Layers prefixed with `cdp_` (e.g., `cdp_fc1`, `cdp_bn1`) in `model.py` |
| TS (Task-Solving) Module | Layers prefixed with `ts_` (e.g., `ts_fc1`, `ts_bn1`) in `model.py`   |
| Concept Vectors     | `symbol_set` (learnable parameter in `cats_net`)      |
| Feature Extractor   | `fe` (attribute within `cats_net`)        |

## Prerequisites

-   Python 3.9.16
-   PyTorch 2.0.1
-   CUDA 11.7
-   Torchvision 0.15.2
-   An environment with CUDA enabled GPUs is recommended. Set the desired GPU using `export CUDA_VISIBLE_DEVICES=X`.

## Directory Structure

-   `main.py`: Main script for training.
-   `model.py`: Contains the model definitions (`cats_net`).
-   `data.py`: Handles data loading and preprocessing for ImageNet.
-   `utils.py`: Contains utility functions for training, optimization, etc.
-   `argument.py`: Defines command-line arguments.
-   `gen_imagenet_concept.sh`: Example script for concept generation (modify as needed).
-   `../Deps/`: Parent directory where pretrained feature extractor weight should be downloaded before training. 
    -   `../Deps/resnet50-0676ba61.pth`: Download from https://download.pytorch.org/models/resnet50-0676ba61.pth
-   `../Results/`: Parent directory where outputs are saved.
    -   `../Results/log/`: Contains log files from training runs.
    -   `../Results/param/`: Contains saved model checkpoints (`.pt` files).

## Training

To train a model, run the `main.py` script.

```bash
python main.py [arguments]
```

**Key Arguments (defined in `argument.py`):**

*   `--data_root`: Path to the ImageNet dataset directory (default: `/data0/share/datasets`).
*   `--model_name` (`-mn`): Feature extractor backbone (e.g., `resnet50`, `resnet18`, `vit_b_16`, default: `resnet50`).
*   `--num_epochs` (`-ne`): Number of training epochs (default: 5).
*   `--batch_size` (`-bs`): Training batch size (default: 512).
*   `--learning_rate` (`-lr`): Learning rate (default: 0.001).
*   `--optimizer_type` (`-ot`): Optimizer (`Adam` or `SGD`, default: `Adam`).
*   `--symbol_size` (`-ss`): Symbol size for the SEA-Net (default: 20).
*   `--num_classes` (`-nc`): Number of classes (default: 1000).
*   `--fix_fe` (`-ff`): Freeze the feature extractor weights (default: True).
*   `--use_pretrain` (`-up`): Use pretrained weights for the feature extractor (default: True).

Refer to `argument.py` for a full list and descriptions of arguments.

**Output Files:**

During training, the script will save:

1.  **Log File:** To `../Results/log/[prefix_]imagenet1k_ss{symbol_size}_fixfe_{timestamp}.log`
2.  **Checkpoint File:** To `../Results/param/[prefix_]imagenet1k_ss{symbol_size}_fixfe_{timestamp}.pt`

Where `{timestamp}` is automatically generated, `{symbol_size}` comes from the argument, and `[prefix_]` is added if `--exp_prefix` is provided.

## Concept Abstraction

The `gen_imagenet_concept.sh` script provides an example of how to run the `main.py` script for concept generation tasks (you might need to adapt it based on the exact functionality in `utils.py` or `main.py` when specific flags like `--gen_concept` are used, which aren't explicitly defined in the provided `argument.py`).

```bash
bash gen_imagenet_concept.sh
# or potentially:
# python main.py -ne 5 -bs 512 -lr 0.001 -nc 1000 -ss 20 --dataset imagenet1k --model_name resnet50 --fix_fe --use_pretrain
```

Check the script and potentially the `main.py` or `utils.py` code for specific arguments related to concept generation if needed.

## Post-Execution: Renaming Outputs

After a training run (or other experiment) is complete, it is recommended to rename the generated output files for better organization.

1.  Go to the `../Results/log/` directory. Find the log file corresponding to your run (e.g., `imagenet1k_ss20_fixfe_20240406175751.log`).
2.  Rename it to `trailXX.log`, where `XX` is a unique identifier for your trial (e.g., `imagenet1k_ss20_fixfe_trail21.log`).
3.  Go to the `../Results/param/` directory. Find the checkpoint file corresponding to your run (e.g., `imagenet1k_ss20_fixfe_20240406175751.pt`).
4.  Rename it to `trailXX.pt`, using the same `XX` identifier as the log file (e.g., `imagenet1k_ss20_fixfe_trail21.pt`).

This helps keep track of different experiments. 