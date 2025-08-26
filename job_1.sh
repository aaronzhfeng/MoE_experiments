#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH -t 10-10:00:00
#SBATCH -J MoE_Exp
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
# SBATCH --cpus-per-task=8   # uncomment/tune if you want CPU cores

set -euo pipefail

# --- Initialize conda for non-interactive shell ---
source /new-stg/home/aaron/miniconda/etc/profile.d/conda.sh || \
  eval "$(/new-stg/home/aaron/miniconda/bin/conda shell.bash hook)"

conda activate g2s_env

# (optional) sanity check
python - <<'PY'
import sys, torch, os
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/experts runs/tb

python -m hetero_moe.training.train_expert   --expert smiles   --config hetero_moe/configs/smiles_expert.yaml   --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz   --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz   --save_path runs/experts/smiles.pt   --epochs 100 --batch_size 8 --device cuda   --grad_clip 1.0   --loss_spike_factor 8 --loss_spike_warmup 2000 --loss_floor 0.02 --outlier_policy cap_running   --num_workers 2 --pin_memory --persistent_workers   --log_every 100   --metrics_csv runs/experts/smiles.metrics.csv   --batch_metrics_csv runs/experts/smiles.batch_metrics.csv   --gpu_debug --gpu_report_every 1000
