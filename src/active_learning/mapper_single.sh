#!/bin/bash

source /etc/profile
module load anaconda/2021b
source activate protein-uq

echo "Split: " $2
echo "Model: " $3
echo "Representation: " $4
echo "Uncertainty: " $5
echo "AL Strategy: " $6
echo "Dropout: " $7


python active_learning.py --split $2 --model $3 --representation $4 --uncertainty $5 --scale --num_folds 3 --al_strategy $6 --al_topk 1 --mean --dropout $7 > $8 2>&1
