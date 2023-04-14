#!/bin/bash

source /etc/profile
module load anaconda/2021b
source activate protein-uq

echo "Split: " $2
echo "Model: " $3
echo "Representation: " $4
echo "Uncertainty: " $5
echo "Dropout: " $6
echo "Cross-validation seed: " $7


python train_all.py --split $2 --model $3 --representation $4 --uncertainty $5 --dropout $6 --scale --seed $7 > $8 2>&1
