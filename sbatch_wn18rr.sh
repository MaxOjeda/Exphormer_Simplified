#!/bin/bash
#SBATCH --job-name=exphormer_wn18rr
#SBATCH --partition=compute-gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs/wn18rr_slurm_%j.out
#SBATCH --error=/nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs/wn18rr_slurm_%j.out

mkdir -p /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs
LOG=/nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max/logs/wn18rr_b16_full.txt

cd /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max

CKPT=results_b16/wn18rr/0/ckpt.pt
if [ -f "$CKPT" ]; then
  echo "INFO: checkpoint found at $CKPT — auto_resume=True will resume from it."
else
  echo "INFO: no checkpoint found — training from epoch 0."
fi

{
  echo "=== GPU Info ==="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "=== Start: $(date) ==="

  /nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py \
      --cfg configs/Exphormer/wn18rr.yaml \
      wandb.use False \
      train.auto_resume True \
      out_dir results_b16

  echo "=== End: $(date) | Exit code: $? ==="
} 2>&1 | tee -a "$LOG"
