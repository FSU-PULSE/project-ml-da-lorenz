#!/bin/bash

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
