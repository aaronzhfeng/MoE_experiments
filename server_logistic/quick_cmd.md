tail -f /new-stg/home/aaron/MoE_experiments/logs/moe-preproc-$(squeue -u $USER -o %i -h | head -n1).out

tail -f /new-stg/home/aaron/MoE_experiments/logs/moe-train-$(squeue -u $USER -o %i -h | head -n1).out

tail -f /new-stg/home/aaron/MoE_experiments/logs/moe-ord-$(squeue -u $USER -o %i -h | head -n1).out