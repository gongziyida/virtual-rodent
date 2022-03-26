## Installation
1. Install [dm\_control](https://github.com/deepmind/dm_control)
2. In Slurm, memory and cpu-binding must be set for the program to run:
   `srun -p scavenger-gpu -n 1 --pty --mem 60000 --gres=gpu:4 --cpu-bind=no -t 0-01:00 /bin/bash`
