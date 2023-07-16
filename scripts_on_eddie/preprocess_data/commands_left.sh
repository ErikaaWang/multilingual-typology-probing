#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -cwd  
#$ -q gpu
#$ -l h_vmem=32G
#$ -pe gpu-a100 1

USER=s2308470
HOME_ROOT_DIRECTORY=/home/${USER}
SCRATCH_ROOT_DIRECTORY=/exports/eddie/scratch/${USER}

SCRATCH_DATA_DIR=${SCRATCH_ROOT_DIRECTORY}/multilingual-typology-probing/data/ud/ud-treebanks-v2.1

source ${HOME_ROOT_DIRECTORY}/.bashrc
source activate multilingual-typology-probing

echo "python preprocess_treebank.py UD_English --bloom bloom-1b7 --use-gpu"
python preprocess_treebank.py UD_English --bloom bloom-1b7 --use-gpu
