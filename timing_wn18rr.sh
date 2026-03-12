#!/bin/bash
# timing_wn18rr.sh
# Measures warm training throughput and one eval round.
# Logs to logs/wn18rr_timing.txt in real-time (tail -f to watch).
mkdir -p /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs
LOG=/nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs/wn18rr_timing.txt
cd /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max
{
  echo "=== GPU Info ==="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo ""
  echo "=== WN18RR timing: 3 epochs, eval at epoch 2 ==="
  /nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py \
      --cfg configs/Exphormer/wn18rr.yaml \
      wandb.use False \
      optim.max_epoch 3 \
      train.eval_period 2
  echo "=== Exit code: $? ==="
} 2>&1 | tee "$LOG"
