#!/bin/bash
#SBATCH --job-name=maltese-t6
#SBATCH --partition=RJMILLER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=bkassaie@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/t6_%j.out
#SBATCH --error=logs/t6_%j.err

# T6 supervised fine-tune of the char-vocab TrOCR decoder, run UNATTENDED on the
# full MLRS/korpus_malti corpus. Stock trocr-base-printed encoder (no Stage-1 T5
# checkpoint). Relies on the already-cached HuggingFace login and the `ocr` conda
# env — NO token is hardcoded here.
#
# Submit:  sbatch scripts/train_t6.sbatch
# Watch:   squeue -u bkassaie ; tail -f logs/t6_<jobid>.out
# Cancel:  scancel <jobid>
# See "## T6 unattended run" in documentation.md.

set -euo pipefail

# --- conda -----------------------------------------------------------------
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ocr

# --- workdir ---------------------------------------------------------------
cd ~/ocr/maltese-ocr
mkdir -p logs

# --- reproducibility banner ------------------------------------------------
echo "=========================================================="
echo "T6 supervised fine-tune"
echo "start time : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "hostname   : $(hostname)"
echo "slurm job  : ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_NODELIST:-<none>}"
echo "git commit : $(git rev-parse HEAD)  ($(git rev-parse --abbrev-ref HEAD))"
echo "python     : $(which python)"
echo "----------------------------------------------------------"
nvidia-smi || echo "WARNING: nvidia-smi failed — no GPU visible?"
echo "=========================================================="

# --- fine-tune -------------------------------------------------------------
# bf16, batch_size=32 (see configs/stage2.yaml). If this OOMs on the 46 GB L40S,
# lower data.batch_size to 16 (then 8) in configs/stage2.yaml and resubmit.
python -m maltese_ocr.train.run --config configs/stage2.yaml

echo "=========================================================="
echo "end time   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=========================================================="
