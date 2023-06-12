#!/bin/sh

USER=s2308470

source /home/${USER}/.bashrc
source activate multilingual-typology-probing

echo "python run.py --language UD_Catalan --experiment-name inter-layer-13 --attribute Number --trainer poisson --gpu --embedding bloom-560m-intermediate-global_step1000 greedy --selection-size 50 --selection-criterion mi"
python run.py --language UD_Catalan --experiment-name inter-layer-13 --attribute Number --trainer poisson --gpu --embedding bloom-560m-intermediate-global_step1000 greedy --selection-size 50 --selection-criterion mi
