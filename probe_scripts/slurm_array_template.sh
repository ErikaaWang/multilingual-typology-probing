#!/bin/sh

USER=s2308470
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`"

source /home/${USER}/.bashrc
source activate multilingual-typology-probing

echo "$COMMAND"
eval "$COMMAND"