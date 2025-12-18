#!/bin/bash

# Run all experiments for all tasks
# This will take a LONG time! Consider running in background or using screen/tmux

echo "Starting all experiments..."
echo "This will run 39 experiments total (3 tasks × 13 optimizers)"
echo ""

# CIFAR-10 (50 epochs each)
echo "========================================"
echo "CIFAR-10 Experiments"
echo "========================================"

echo "Running parameter-free optimizers..."
for opt in dog ldog tdog prodigy; do
  echo "Running CIFAR-10 with $opt"
  python src/main.py --task cifar10 --optimizer $opt
done

echo "Running baseline optimizers..."
for opt in sgd_0.1 sgd_0.01 sgd_0.001 adam_0.001 adam_0.0001 adam_0.00001 adamw_0.001 adamw_0.0001 adamw_0.00001; do
  echo "Running CIFAR-10 with $opt"
  python src/main.py --task cifar10 --optimizer $opt
done

# Oxford-IIIT Pet (10 epochs each)
echo ""
echo "========================================"
echo "Oxford-IIIT Pet Experiments"
echo "========================================"

echo "Running parameter-free optimizers..."
for opt in dog ldog tdog prodigy; do
  echo "Running Oxford-Pet with $opt"
  python src/main.py --task oxford_pet --optimizer $opt
done

echo "Running baseline optimizers..."
for opt in sgd_0.1 sgd_0.01 sgd_0.001 adam_0.001 adam_0.0001 adam_0.00001 adamw_0.001 adamw_0.0001 adamw_0.00001; do
  echo "Running Oxford-Pet with $opt"
  python src/main.py --task oxford_pet --optimizer $opt
done

# SST-2 (5 epochs each)
echo ""
echo "========================================"
echo "SST-2 Experiments"
echo "========================================"

echo "Running parameter-free optimizers..."
for opt in dog ldog tdog prodigy; do
  echo "Running SST-2 with $opt"
  python src/main.py --task sst2 --optimizer $opt
done

echo "Running baseline optimizers..."
for opt in sgd_0.1 sgd_0.01 sgd_0.001 adam_0.001 adam_0.0001 adam_0.00001 adamw_0.001 adamw_0.0001 adamw_0.00001; do
  echo "Running SST-2 with $opt"
  python src/main.py --task sst2 --optimizer $opt
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Results are saved in ./results/"
echo "Run 'python analyze_results.py' to see the summary"
