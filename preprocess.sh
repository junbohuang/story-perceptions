#!/bin/bash
#SBATCH --gpus-per-node=2

huggingface-cli login --token $HFTOKENS
python src/preprocess.py