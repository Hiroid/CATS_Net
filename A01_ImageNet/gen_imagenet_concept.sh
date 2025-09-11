#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --nodelist gpu02
#SBATCH --mem=64G
#SBATCH -o ./slurm/slurm%A_%a.out
#SBATCH -e ./slurm/slurm%A_%a.err
#SBATCH --array=0%1

cmdlines=(
    "python -m A01_ImageNet.main -ne 5 -bs 512 -lr 0.001 -nc 1000 -ss 20 --dataset imagenet1k --model_name resnet50 --fix_fe --use_pretrain"

    # "python -m A01_ImageNet.main -ne 5 -bs 512 -lr 0.001 -nc 1000 -ss 20 --dataset imagenet1k --model_name vit_b_16 --fix_fe --use_pretrain"
)
cmd=${cmdlines[SLURM_ARRAY_TASK_ID]}
$cmd

