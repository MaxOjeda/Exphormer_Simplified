#!/bin/bash
# run_wn18rr.sh — full-graph (NBFNet-style) training, WN18RR, 100 epochs
# Logs are written in real-time to logs/wn18rr_run.txt (tail -f to watch)
# Estimated: ~8-10h on H100 (200 steps×~1.4s/step train + ~120s eval every 5 epochs)
# Full-graph training eliminates the subgraph structural mismatch
mkdir -p /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs
LOG=/nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs/wn18rr_run.txt
cd /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max
{
  echo "=== GPU Info ==="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "=== Start: $(date) ==="
  /nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py \
      --cfg configs/Exphormer/wn18rr.yaml \
      wandb.use False \
      optim.max_epoch 100 \
      train.auto_resume False
  echo "=== End: $(date) | Exit code: $? ==="
} 2>&1 | tee "$LOG"
