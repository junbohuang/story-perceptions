#!/bin/bash
#SBATCH --gpus-per-node=1

huggingface-cli login --token $HFTOKENS
python src/preprocess.py
