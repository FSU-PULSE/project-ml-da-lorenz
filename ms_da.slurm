#!/bin/bash
#SBATCH --job-name=l63_ms_da
#SBATCH --output=/home/jv24b/github/debug_da/logs/ms_da_%A_%a.out
#SBATCH --error=/home/jv24b/github/debug_da/logs/ms_da_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jvelasco@coaps.fsu.edu
#SBATCH --partition=class
# Array job: each task is one whole node and runs a full ms_da.py with a distinct
# master seed (seed = MS_DA_SEED_BASE + array task id), producing one results/*.h5.
# The array size == the seed count; override it at submit time, e.g.
# `sbatch --array=0-3 ms_da.slurm` for 4 seeds (see submit_ms_da.sh).
#SBATCH --array=0-3
# GPUs are not registered with SLURM gres on this cluster, so we reserve the
# whole node (--exclusive) to guarantee sole access to its RTX A5000. Exclude
# class101/102 (reachable without SLURM, usually busy) and 103 as a buffer.
#SBATCH --exclude=class101,class102,class103

# Path to my conda manager
source /home/jv24b/miniconda3/etc/profile.d/conda.sh

WORKING_DIR="/home/jv24b/github/debug_da"
cd $WORKING_DIR

# Activate lorenzo env
conda activate lorenzo

# Ensure conda's newer libstdc++ is used (avoids GLIBCXX_* import errors) and
# that numbalsoda finds its shared libs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Each task's master seed; experiment params come from env (set by submit_ms_da.sh)
# with defaults matching the current launch_env.sh run.
SEED=$(( ${MS_DA_SEED_BASE:-100} + SLURM_ARRAY_TASK_ID ))

echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname), seed=${SEED}"

python -u $WORKING_DIR/ms_da.py \
    --seed   "$SEED" \
    --ic     "${MS_DA_N_IC:-250}" \
    --m      "${MS_DA_M:-100}" \
    --ne     "${MS_DA_NE:-100}" \
    --ne-max "${MS_DA_NE_MAX:-1000}" \
    --reg    "${MS_DA_REG:-0.7}"
MS_DA_EXIT=$?
if [ $MS_DA_EXIT -ne 0 ]; then
    echo "ms_da.py failed with exit code $MS_DA_EXIT"
    exit $MS_DA_EXIT
fi

echo "ms_da.py completed at: $(date +"%Y-%m-%d %H:%M:%S")"

echo ""
echo "=============================="
echo "All done : $(date)"
echo "=============================="
