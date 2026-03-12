#!/bin/bash
cd /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "=== WN18RR smoke test: 5 train batches, full eval with eval_batch_size=32 ==="
/nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py \
    --cfg configs/Exphormer/wn18rr.yaml \
    wandb.use False \
    optim.max_epoch 1 \
    train.batch_size 64 \
    kgc.max_nodes 100 \
    train.max_iter 5
echo "=== Exit code: $? ==="
