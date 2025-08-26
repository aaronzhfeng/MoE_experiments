#!/bin/bash
#SBATCH -J moe-smiles
#SBATCH -w gpu-1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# ---- Conda in non-interactive shells ----
if [ -f "/new-stg/home/aaron/miniconda/etc/profile.d/conda.sh" ]; then
  source /new-stg/home/aaron/miniconda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)" || true
fi
conda activate g2s_env

# Slurm sets CUDA_VISIBLE_DEVICES for the allocation; don't override it.
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO_DIR="$HOME/MoE_experiments"
cd "$REPO_DIR"

mkdir -p runs/experts runs/tb logs

# Diagnostics
nvidia-smi
python -V
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("current_device:", torch.cuda.current_device())
PY

# ---- Train ----
srun python -u -m hetero_moe.training.train_expert \
  --expert smiles \
  --config hetero_moe/configs/smiles_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --save_path runs/experts/smiles.pt \
  --epochs 3 --batch_size 8 --device cuda \
  --num_workers 4 --pin_memory --persistent_workers \
  --grad_clip 1.0 \
  --loss_spike_factor 3.0 --loss_spike_warmup 200 --outlier_policy skip \
  --log_every 100 \
  --metrics_csv runs/experts/smiles.metrics.csv \
  --batch_metrics_csv runs/experts/smiles.batch_metrics.csv

