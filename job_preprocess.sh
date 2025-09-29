#!/bin/bash
#SBATCH -J moe-preproc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Slurm sets CUDA_VISIBLE_DEVICES for the allocation; don't override it.
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

REPO_DIR="$HOME/MoE_experiments"
cd "$REPO_DIR"

mkdir -p logs runs/moe hetero_moe/data/raw/ord hetero_moe/data/processed/ord/graph2smiles_npz

# ---- Map ORD files to expected names ----
ln -sf "$HOME/ORD/ord-src-train.txt" hetero_moe/data/raw/ord/src-train.txt
ln -sf "$HOME/ORD/ord-tgt-train.txt" hetero_moe/data/raw/ord/tgt-train.txt
ln -sf "$HOME/ORD/ord-src-val.txt"   hetero_moe/data/raw/ord/src-val.txt
ln -sf "$HOME/ORD/ord-tgt-val.txt"   hetero_moe/data/raw/ord/tgt-val.txt
ln -sf "$HOME/ORD/ord-src-test.txt"  hetero_moe/data/raw/ord/src-test.txt
ln -sf "$HOME/ORD/ord-tgt-test.txt"  hetero_moe/data/raw/ord/tgt-test.txt

# ---- Preprocess to NPZ (Graph2SMILES bridge) ----
python -m hetero_moe.preprocess.graph2smiles_bridge \
  --raw_dir hetero_moe/data/raw/ord \
  --out_dir hetero_moe/data/processed/ord/graph2smiles_npz \
  --model g2s --repr smiles --max_src_len 512 --max_tgt_len 512 --workers 4

# ---- Diagnostics ----
nvidia-smi | sed -n '1,20p'
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
