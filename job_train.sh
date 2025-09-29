#!/bin/bash
#SBATCH -J moe-train
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

REPO_DIR="$HOME/MoE_experiments"
cd "$REPO_DIR"

mkdir -p logs runs/moe

echo "=== STARTING TRAINING PHASE ==="
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"

# ---- Diagnostics ----
echo "=== SYSTEM DIAGNOSTICS ==="
nvidia-smi | sed -n '1,20p'
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

# Check if preprocessing files exist
echo "=== CHECKING PREPROCESSING OUTPUT ==="
if [ -f "hetero_moe/data/processed/ord/graph2smiles_npz/train_0.npz" ] && \
   [ -f "hetero_moe/data/processed/ord/graph2smiles_npz/val_0.npz" ] && \
   [ -f "hetero_moe/data/processed/ord/graph2smiles_npz/test_0.npz" ]; then
    echo "✅ All preprocessing files found"
    ls -lh hetero_moe/data/processed/ord/graph2smiles_npz/*.npz
else
    echo "❌ Missing preprocessing files. Please run preprocessing first!"
    exit 1
fi

# ---- Train MoE (smiles + graph) ----
echo "=== STARTING TRAINING ==="
echo "Training started at: $(date)"

srun python -u -m hetero_moe.training.train_moe \
  --config hetero_moe/configs/moe.yaml \
  --train_bin hetero_moe/data/processed/ord/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/ord/graph2smiles_npz/val_0.npz \
  --save_path runs/moe/ord_smiles_graph.pt \
  --epochs 5 --batch_size 32 --device cuda \
  --num_workers 4 --pin_memory --persistent_workers \
  --log_every 100

echo "=== TRAINING COMPLETED ==="
echo "Training finished at: $(date)"

# Check if training output exists
if [ -f "runs/moe/ord_smiles_graph.pt.best" ]; then
    echo "✅ Training model saved successfully"
    ls -lh runs/moe/ord_smiles_graph.pt*
else
    echo "❌ Training model not found!"
    exit 1
fi

# ---- Evaluate on test set ----
echo "=== STARTING EVALUATION ==="
echo "Evaluation started at: $(date)"

python -m hetero_moe.evaluation.eval_moe \
  --test_bin hetero_moe/data/processed/ord/graph2smiles_npz/test_0.npz \
  --load_path runs/moe/ord_smiles_graph.pt.best \
  --vocab_file hetero_moe/data/processed/ord/graph2smiles_npz/vocab_smiles.txt \
  --beam_size 5 --k 5 \
  --out runs/moe/ord_eval_results.json

echo "=== EVALUATION COMPLETED ==="
echo "Evaluation finished at: $(date)"

# Check if evaluation output exists
if [ -f "runs/moe/ord_eval_results.json" ]; then
    echo "✅ Evaluation results saved successfully"
    echo "Results file size: $(ls -lh runs/moe/ord_eval_results.json | awk '{print $5}')"
else
    echo "❌ Evaluation results not found!"
    exit 1
fi

echo "=== FULL PIPELINE COMPLETED ==="
echo "All stages completed successfully at: $(date)"
echo "Final outputs:"
echo "- Model: runs/moe/ord_smiles_graph.pt.best"
echo "- Results: runs/moe/ord_eval_results.json"
