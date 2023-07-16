#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -cwd  
#$ -l h_vmem=32G
#$ -q gpu
#$ -pe gpu-a100 1

USER=s2308470
COMMAND="`sed \"${SGE_TASK_ID}q;d\" $1`"

HOME_ROOT_DIRECTORY=/home/${USER}
SCRATCH_ROOT_DIRECTORY=/exports/eddie/scratch/${USER}

SCRATCH_DATA_DIR=${SCRATCH_ROOT_DIRECTORY}/multilingual-typology-probing/data/ud/ud-treebanks-v2.1


source ${HOME_ROOT_DIRECTORY}/.bashrc
source activate multilingual-typology-probing

echo "$COMMAND"
eval "$COMMAND"