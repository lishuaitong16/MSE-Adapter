#!/bin/bash

cd "$(dirname "$0")"


echo "Starting regression (simsv2, mosei) on GPU X..."
nohup python run.py --task_type regression --gpu_ids 1 > logs/run_regression.log 2>&1 &
echo "Regression PID: $!"

echo "Starting classification (meld, cherma) on GPU Y..."
nohup python run.py --task_type classification --gpu_ids 1 > logs/run_classification.log 2>&1 &
echo "Classification PID: $!"

echo "Both processes started. Monitor with:"
echo "  tail -f logs/run_regression.log"
echo "  tail -f logs/run_classification.log"
