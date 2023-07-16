#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -cwd  
#$ -l h_vmem=32G
#$ -q gpu
#$ -pe gpu-a100 1
#$ -t 1-14

LAYER=17
CHECKPOINT=$1
USER=s2308470
HOME_ROOT_DIRECTORY=/home/${USER}
SCRATCH_ROOT_DIRECTORY=/exports/eddie/scratch/${USER}

SCRATCH_DATA_DIR=${SCRATCH_ROOT_DIRECTORY}/multilingual-typology-probing/data/ud/ud-treebanks-v2.1

# loop over languages in parallel
CORPUS=($(cat scripts/languages_bloom.lst))
echo "now processing task id: ${SGE_TASK_ID}"
echo "CORPUS: ${CORPUS[${SGE_TASK_ID}-1]}"

source ${HOME_ROOT_DIRECTORY}/.bashrc
source activate multilingual-typology-probing


echo "python preprocess_treebank.py ${CORPUS[${SGE_TASK_ID}-1]} --experiment-name inter-layer-$LAYER --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-1b7 --checkpoint $CHECKPOINT --inter-layer $LAYER --use-gpu --reset-HF-cache-dir"
python preprocess_treebank.py ${CORPUS[${SGE_TASK_ID}-1]} --experiment-name inter-layer-$LAYER --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-1b7 --checkpoint $CHECKPOINT --inter-layer $LAYER --use-gpu --reset-HF-cache-dir
# python preprocess_treebank.py ${CORPUS[${SGE_TASK_ID}-1]} --experiment-name inter-layer-$LAYER --treebanks-root $SCRATCH_DATA_DIR --bloom bloom-1b7 --inter-layer $LAYER --use-gpu