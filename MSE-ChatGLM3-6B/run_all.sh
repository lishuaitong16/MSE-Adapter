#!/bin/bash

cd "$(dirname "$0")"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate MSE-Adapter

echo "Starting regression (simsv2, mosei) on GPU 3..."
nohup python run.py --task_type regression --modelName cmcm_reg --model_save_dir results/models_reg --res_save_dir results/results_reg --gpu_ids 3 > logs/run_regression.log 2>&1 &
echo "Regression PID: $!"

echo "Starting classification (meld, cherma) on GPU 2..."
nohup python run.py --task_type classification --modelName cmcm_cls --model_save_dir results/models_cls --res_save_dir results/results_cls --gpu_ids 2 > logs/run_classification.log 2>&1 &
echo "Classification PID: $!"

echo "Both processes started. Monitor with:"
echo "  tail -f logs/run_regression.log"
echo "  tail -f logs/run_classification.log"
