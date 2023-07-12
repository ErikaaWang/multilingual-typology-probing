#!/bin/sh
#PBS -d /exports/eddie/scratch/s2308470/multilingual-typology-probing
#PBS -D /exports/eddie/scratch/s2308470/multilingual-typology-probing
#PBS -e /exports/eddie/scratch/s2308470/multilingual-typology-probing
#PBS -o /exports/eddie/scratch/s2308470/multilingual-typology-probing
USER=s2308470

source /home/${USER}/.bashrc
source activate multilingual-typology-probing

echo "python preprocess_treebank.py UD_English --bloom bloom-1b7 --use-gpu"
python preprocess_treebank.py UD_English --bloom bloom-1b7 --use-gpu
