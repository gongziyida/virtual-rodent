#!/bin/bash  
#SBATCH --job-name=simple_test_IMPALA
#SBATCH --output=simple_test_IMPALA.out
#SBATCH --error=simple_test_IMPALA.err
#SBATCH --mem=60000 # 60000 MB RAM
#SBATCH -t 0-23:59 # Time limit days-hrs:min:sec
#SBATCH -N 1 # Requested number of notes
#SBATCH -n 50 # Requested number of CPUs
#SBATCH -p scavenger-gpu # Partition
#SBATCH --gres=gpu:2 # gpu:[num requested gpus]
#SBATCH --exclude=dcc-tdunn-gpu-01

source /hpc/home/zg93/virtual-rodent/activate_environment.sh
export SIMULATOR_IMPALA="test" 
python3 simple_test_IMPALA.py
python3 visualize.py
