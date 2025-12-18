#!/bin/bash

# Run all CIFAR-10 experiments (13 optimizers × 50 epochs)

echo "Starting CIFAR-10 experiments..."
echo "Total: 13 optimizers × 50 epochs each"
echo ""

# Parameter-free optimizers
# echo "Running parameter-free optimizers..."
# for opt in dog ldog tdog prodigy; do
#   echo "→ Running CIFAR-10 with $opt"
#   python src/main.py --task cifar10 --optimizer $opt
#   echo ""
# done

# Baseline optimizers
echo "Running baseline optimizers..."
# for opt in adamw_0.001 adamw_0.0001 adamw_0.00001; do
for opt in sgd_1.0 sgd_0.5 adam_0.01 adam_0.005 adamw_0.01 adamw_0.005; do

  echo "→ Running CIFAR-10 with $opt"
  python src/main.py --task cifar10 --optimizer $opt
  echo ""
done

echo "CIFAR-10 experiments completed!"
echo "Results saved in ./results/"
