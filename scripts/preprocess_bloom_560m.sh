#!/bin/bash

#SBATCH --array=1-16

USER=s2308470
HOME_ROOT_DIRECTORY=/home/${USER}
SCRATCH_ROOT_DIRECTORY=/disk/scratch/${USER}

HOME_DATA_DIR=${HOME_ROOT_DIRECTORY}/multilingual-typology-probing/data/ud/ud-treebanks-v2.1
SCRATCH_DATA_DIR=${SCRATCH_DIRECTORY}/data/ud/ud-treebanks-v2.1

# first copy input data from home dir to scratch dir
# cp ${HOME_DATA_DIR}/* ${SCRATCH_DATA_DIR}


# python test.py > output_${SLURM_ARRAY_TASK_ID}.txt

LANGUAGES=($(cat scripts/languages_bloom.lst))
echo "now processing task id: ${SLURM_ARRAY_TASK_ID}"
echo "LANGUAGE: ${LANGUAGES[${SLURM_ARRAY_TASK_ID}-1]}"

# conda activate multilingual-typology-probing

# for CORPUS in $(cat scripts/languages_bloom.lst); do
#   echo "python preprocess_treebank.py $CORPUS --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-560m --use-gpu"
#   python preprocess_treebank.py $CORPUS --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-560m --use-gpu
# done