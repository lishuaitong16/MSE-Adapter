#!/bin/bash
THRESHOLD=11000  # MB，低于此值认为 GPU 空闲
LOG=/home/lishuaitong/MSE-Adapter/wait_and_launch_chatglm.log

echo "[$(date)] 开始监控 GPU 2 和 GPU 3，显存低于 ${THRESHOLD}MB 时启动 ChatGLM..." | tee $LOG

while true; do
    USED_2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    USED_3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    echo "[$(date)] GPU2 used: ${USED_2}MB  GPU3 used: ${USED_3}MB" | tee -a $LOG

    if [ "$USED_2" -lt "$THRESHOLD" ] && [ "$USED_3" -lt "$THRESHOLD" ]; then
        echo "[$(date)] 两张卡均空闲，启动 ChatGLM..." | tee -a $LOG
        cd /home/lishuaitong/MSE-Adapter/MSE-ChatGLM3-6B
        bash run_all.sh 2>&1 | tee -a $LOG
        break
    fi
    sleep 120
done
