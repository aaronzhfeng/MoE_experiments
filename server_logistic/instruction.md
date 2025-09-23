### Cluster quick reference (GPU, Slurm, environment)

Table of contents
- [Quick system checks](#quick-system-checks)
- [Slurm overview](#slurm-overview)
- [Check GPU availability](#check-gpu-availability)
- [Quick GPU status checks (login node)](#quick-gpu-status-checks-login-node)
- [Submit a GPU job (template)](#submit-a-gpu-job-template)
- [Monitor and manage jobs](#monitor-and-manage-jobs)
  - [Job monitoring (watch & logs)](#job-monitoring-watch--logs)
- [Environment: conda and modules](#environment-conda-and-modules)
- [Storage and quotas](#storage-and-quotas)
- [Networking / proxy](#networking--proxy)
- [Troubleshooting](#troubleshooting)
- [Handy aliases (defined in ~/.bashrc)](#handy-aliases-defined-in-bashrc)

This guide captures how to check GPU availability, submit/manage jobs, and handle environment basics on this cluster. It excludes project-specific instructions.

### Quick system checks

```bash
# OS and kernel
cat /etc/os-release | sed -n '1,8p'; uname -a

# CPU and memory
lscpu | sed -n '1,20p'
free -h

# Disk and your home usage
df -hT | sed -n '1,60p'
du -sh "$HOME"
```

### Slurm overview

```bash
# Partitions and generic resources
sinfo -o "%P %a %l %D %c %G %m %f"

# All GPU queue jobs (colored helper if available)
gg            # alias to: squeue -p gpu (with coloring)

# Your jobs
qq            # alias to: squeue -u $USER (with coloring)
```

Notes:
- The `gpu` partition provides access to A6000 and H100 GPUs. Request GPUs explicitly (see templates below).
- Avoid pinning to a specific node (e.g., `-w gpu-2`) unless necessary; let Slurm schedule for you.

### Check GPU availability

Login nodes typically do not expose GPUs. Use Slurm to test on a compute node.

Interactive test (brief session):

```bash
srun -p gpu --gpus=1 --mem=16G -t 00:10:00 --pty bash -l
module purge && module load cuda/12.3
source ~/.bashrc
python - <<'PY'
import torch
print('cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
exit
```

Batch test (submit and read the log):

```bash
cat > gpu_test.sh <<'SH'
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -J gpu_test
#SBATCH -o slurm-%j.out

module purge && module load cuda/12.3
source ~/.bashrc

python - <<'PY'
import subprocess, sys
print('=== nvidia-smi ===')
subprocess.run(['nvidia-smi'])
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch check error:', e, file=sys.stderr)
PY
SH

sbatch gpu_test.sh
```

#### Quick GPU status checks (login node)

```bash
# Partitions and GPU types/counts
sinfo -o "%P %a %l %D %c %G %m"

# Node-level view for the gpu partition
sinfo -N -p gpu -o "%N %G %T %C"

# Inspect each GPU node directly (works on this cluster)
for n in gpu-1 gpu-2 gpu-3; do
  echo "=== $n ==="
  ssh "$n" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true'
done

# Individually
ssh gpu-1 nvidia-smi
ssh gpu-2 nvidia-smi
ssh gpu-3 nvidia-smi

# Live watch of one GPU on a node (press q to quit)
ssh -t gpu-3 'watch -n 1 "nvidia-smi -i 0"'
```

### Submit a GPU job (template)

General A6000/H100-agnostic template (1 GPU):

```bash
cat > job_gpu.sh <<'SH'
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1                  # or: --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -J myjob
#SBATCH -o slurm-%j.out

module purge
module load cuda/12.3             # adjust if needed
source ~/.bashrc                  # ensures conda is initialized
conda activate g2s_env            # choose your env

# Run your workload
python your_script.py --arg1 foo
SH

sbatch job_gpu.sh
```

Request a specific GPU type when needed:

```bash
# A6000 example
#SBATCH --gres=gpu:a6000:1

# H100 example
#SBATCH --gres=gpu:h100:1
```

### Monitor and manage jobs

```bash
# Watch queues
gg                # GPU queue snapshot
qq                # your jobs
sq                # alias to: squeue

# Inspect a job
scontrol show job <JOBID>
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqCPUS,ReqMem,AllocTRES

# Cancel a job
scancel <JOBID>

# Follow logs
tail -f slurm-<JOBID>.out
```

#### Job monitoring (watch & logs)

```bash
# Replace <JOBID> with your job id

# Auto-refresh queue status
watch -n 2 'squeue -j <JOBID>'

# Live follow last 50 lines of the job's log (default path from job_2.sh)
watch -n 1 'tail -n 10 logs/moe-ord-<JOBID>.out'

watch -n 5 'tail -n 10 logs/moe-ord-747367.out'

# Or just tail the full log
tail -n 200 -f logs/moe-ord-<JOBID>.out

# If the log file isn't created yet, wait for it then follow
while [ ! -f "logs/moe-ord-<JOBID>.out" ]; do sleep 1; done; tail -f logs/moe-ord-<JOBID>.out

# Inspect detailed job fields and reason
scontrol show job <JOBID> | egrep 'JobState|Reason|NodeList|StartTime|RunTime|StdOut|Gres'

# Historical accounting (if enabled on the cluster)
sacct -j <JOBID> --format=JobID,State,ExitCode,Start,End,Elapsed,NodeList,ReqTRES,AllocTRES
```

### Environment: conda and modules

```bash
# Conda basics
conda env list
conda activate g2s_env     # or another env in /new-stg/home/aaron/miniconda/envs

# Modules (CUDA, compilers, etc.)
module avail
module purge && module load cuda/12.3
```

Tips:
- Prefer loading a CUDA module that matches your PyTorch/CUDA build. If unsure, start with `cuda/12.3` (default on this system) and use wheels built for that runtime.
- Source `~/.bashrc` inside batch scripts so `conda` is available in non-interactive shells.

### Storage and quotas

```bash
df -hT | sed -n '1,60p'   # mount usage
du -sh "$HOME"            # your home size
quota -s                   # if quotas are enabled
```

Notes:
- `/new-stg/home` is a shared NFS and often near capacity; place large datasets/outputs on shared storage if available.

### Networking / proxy

Proxies are exported in the login environment:

```bash
env | egrep -i '^(http|https|ftp|rsync)_proxy='
```

For `pip` behind a proxy (usually not needed if env vars are present):

```bash
pip install <pkg> --proxy "$http_proxy"
```

### Troubleshooting

- sbatch says you must specify GPU: add `#SBATCH --gpus=1` or `#SBATCH --gres=gpu:<type>:<n>`.
- `nvidia-smi` not found on login node: run it inside an allocated job (`srun` or `sbatch`).
- Conda not found in job: add `source ~/.bashrc` before `conda activate`.
- Long queue wait: shorten `--time`, lower `--mem`, avoid pinning nodes, and request only the GPUs you need.

### Handy aliases (defined in ~/.bashrc)

```bash
sq   # squeue
gg   # squeue -p gpu (color-coded by node)
qq   # squeue -u aaron (your jobs)
```


