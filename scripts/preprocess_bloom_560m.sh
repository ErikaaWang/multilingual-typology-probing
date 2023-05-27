#!/bin/bash

#SBATCH --array=0-15

USER=s2308470
HOME_ROOT_DIRECTORY=/home/${USER}
SCRATCH_ROOT_DIRECTORY=/disk/scratch/${USER}

HOME_DATA_DIR=${HOME_ROOT_DIRECTORY}/multilingual-typology-probing/data/ud/ud-treebanks-v2.1
SCRATCH_DATA_DIR=${SCRATCH_DIRECTORY}/data/ud/ud-treebanks-v2.1

# first copy input data from home dir to scratch dir
# cp ${HOME_DATA_DIR}/* ${SCRATCH_DATA_DIR}

CORPUS=($(cat scripts/languages_bloom.lst))
echo "now processing task id: ${SLURM_ARRAY_TASK_ID}"
echo "CORPUS: ${CORPUS[${SLURM_ARRAY_TASK_ID}]}"

# OUTPUT_FILE=${SCRATCH_DIRECTORY}/output/output_${CORPUS[${SLURM_ARRAY_TASK_ID}]}.txt
source /home/${USER}/.bashrc
source activate multilingual-typology-probing

echo "python preprocess_treebank.py ${CORPUS[${SLURM_ARRAY_TASK_ID}]} --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-560m --use-gpu"
python preprocess_treebank.py ${CORPUS[${SLURM_ARRAY_TASK_ID}]} --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-560m --use-gpu

# after finish, move data from scratch to home
