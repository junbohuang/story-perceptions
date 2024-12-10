#!/bin/bash
#SBATCH --gpus-per-node=i

huggingface-cli login --token $HFTOKENS
python src/preprocess.py
