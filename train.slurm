#!/bin/bash
#SBATCH --job-name=l63_training
#SBATCH --output=/home/jv24b/github/lorenz_ml_da/logs/train_%j.out
#SBATCH --error=/home/jv24b/github/lorenz_ml_da/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=16000M
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jvelasco@coaps.fsu.edu
#SBATCH --partition=class
#SBATCH --exclude=class101,class102,class105,class106,class107,class108,class109,class110,class111,class112

# Path to my conda manager
source /home/jv24b/miniconda3/etc/profile.d/conda.sh

WORKING_DIR="/home/jv24b/github/lorenz_ml_da"
cd $WORKING_DIR

# Activate lorenzo env
conda activate lorenzo

# Ensure conda's newer libstdc++ is used (avoids GLIBCXX_* import errors)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

ML_MODEL_CONFIG_FILE="${1:-config.yml}"
echo "ML_MODEL_CONFIG_FILE: $ML_MODEL_CONFIG_FILE"

# Launch training
echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Training started at: $(date +"%Y-%m-%d %H:%M:%S")"

#python /home/jv24b/github/lorenz_ml_da/Main_ML.py "/home/jv24b/github/lorenz_ml_da/$ML_MODEL_CONFIG_FILE"
python -u $WORKING_DIR/Main_ML.py $ML_MODEL_CONFIG_FILE
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "Training completed at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Job completed at: $(date +"%Y-%m-%d %H:%M:%S")"

echo ""
echo "=============================="
echo "All done : $(date)"
echo "=============================="
