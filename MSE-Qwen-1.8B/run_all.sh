#!/bin/bash

cd "$(dirname "$0")"


echo "Starting regression (simsv2, mosei) on GPU X..."
nohup python run.py --task_type regression --modelName cmcm_reg --model_save_dir results/models_reg --res_save_dir results/results_reg --gpu_ids 1 > logs/run_regression.log 2>&1 &
echo "Regression PID: $!"

echo "Starting classification (meld, cherma) on GPU Y..."
nohup python run.py --task_type classification --modelName cmcm_cls --model_save_dir results/models_cls --res_save_dir results/results_cls --gpu_ids 1 > logs/run_classification.log 2>&1 &
echo "Classification PID: $!"

echo "Both processes started. Monitor with:"
echo "  tail -f logs/run_regression.log"
echo "  tail -f logs/run_classification.log"
