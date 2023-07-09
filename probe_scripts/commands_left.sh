#!/bin/sh

USER=s2308470

source /home/${USER}/.bashrc
source activate multilingual-typology-probing

echo "python run.py --language UD_Spanish --experiment-name inter-layer-13 --attribute POS --trainer poisson --gpu --embedding bloom-1b1-intermediate-global_step100000 greedy --selection-size 50 --selection-criterion mi"
python run.py --language UD_Spanish --experiment-name inter-layer-13 --attribute POS --trainer poisson --gpu --embedding bloom-1b1-intermediate-global_step100000 greedy --selection-size 50 --selection-criterion mi
