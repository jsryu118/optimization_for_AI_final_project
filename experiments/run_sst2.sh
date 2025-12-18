#!/bin/bash

# Run all SST-2 experiments (13 optimizers × 5 epochs)

echo "Starting SST-2 experiments..."
echo "Total: 13 optimizers × 5 epochs each"
echo ""

# Parameter-free optimizers
# echo "Running parameter-free optimizers..."
# for opt in dog ldog tdog prodigy; do
#   echo "→ Running SST-2 with $opt"
#   python src/main.py --task sst2 --optimizer $opt
#   echo ""
# done

# Baseline optimizers
echo "Running baseline optimizers..."
# for opt in adamw_0.001 adamw_0.0001 adamw_0.00001; do
for opt in sgd_1.0 sgd_0.5 adam_0.01 adam_0.005 adamw_0.01 adamw_0.005; do

  echo "→ Running SST-2 with $opt"
  python src/main.py --task sst2 --optimizer $opt
  echo ""
done

echo "SST-2 experiments completed!"
echo "Results saved in ./results/"
