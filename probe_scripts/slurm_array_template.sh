#!/bin/sh

COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`

eval "$COMMAND"