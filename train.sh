#!/bin/bash
#SBATCH --gpus-per-node=2

huggingface-cli login --token $HFTOKENS
python models/llm2vec-llama31-8b-instruct.py
