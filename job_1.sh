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

python -m hetero_moe.training.train_moe_ntf \
  --experts graph,smiles \
  --config hetero_moe/configs/smiles_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --epochs 100 --batch_size 8 --device cuda --lr 1e-3 --grad_clip 1.0 \
  --max_steps_per_seq 256 --stop_on_eos \
  --tf_warmup_steps 32 --aux_alpha 0.3 \
  --router_use_gatefeats --router_gate_dim 2048 \
  --log_every 100 --gpu_debug --gpu_report_every 1000 \
  --save_path runs/moe_ntf/smiles_x2.pt \
  --metrics_csv runs/moe_ntf/smiles_x2.metrics.csv \
  --batch_metrics_csv runs/moe_ntf/smiles_x2.batches.csv \
  --valid_eval_em --valid_em_batches 100 \
  --num_workers 4 --pin_memory --persistent_workers \
  --target_key target_ids --inspect_batch

