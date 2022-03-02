#!/bin/bash

# For activating the conda environment at Duke Computing Cluster

srun --pty bash -i # Start an interactive bash session
. /opt/apps/rhel8/miniconda3/etc/profile.d/conda.sh
conda activate dm_control
