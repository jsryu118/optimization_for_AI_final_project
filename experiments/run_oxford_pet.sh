#!/bin/bash

# Run all Oxford-IIIT Pet experiments (13 optimizers × 10 epochs)

echo "Starting Oxford-IIIT Pet experiments..."
echo "Total: 13 optimizers × 10 epochs each"
echo ""

# Parameter-free optimizers
echo "Running parameter-free optimizers..."
for opt in dog ldog tdog prodigy; do
  echo "→ Running Oxford-Pet with $opt"
  python src/main.py --task oxford_pet --optimizer $opt
  echo ""
done

# Baseline optimizers
echo "Running baseline optimizers..."
# for opt in sgd_0.1 sgd_0.01 sgd_0.001 adam_0.001 adam_0.0001 adam_0.00001 adamw_0.001 adamw_0.0001 adamw_0.00001; do
for opt in sgd_1.0 sgd_0.5 sgd_0.1 sgd_0.01 sgd_0.001 adam_0.01 adam_0.005 adam_0.001 adam_0.0001 adam_0.00001 adamw_0.001 adamw_0.0001 adamw_0.00001 adamw_0.01 adamw_0.005; do
  echo "→ Running Oxford-Pet with $opt"
  python src/main.py --task oxford_pet --optimizer $opt
  echo ""
done

echo "Oxford-IIIT Pet experiments completed!"
echo "Results saved in ./results/"
