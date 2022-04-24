import os, socket, time
import tqdm
import copy
import numpy as np
import torch

from torch.multiprocessing import Event, Value, Process, set_start_method
import torch.distributed.rpc as rpc

from virtual_rodent.utils import load_checkpoint
from .base import IMPALABase, _N_CPU
from .DistributedAgent import DistributedAgent
from .Learner import Learner
from .ParameterServer import run_parameter_server

#master_addr = re.match('\w*-\w*-.[0-9]*', os.environ['SLURM_JOB_NODELIST']).group()
#''.join(master_addr.split('['))
os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
os.environ['MASTER_PORT'] = '8818'

set_start_method('spawn', force=True)

class IMPALA_CPU(IMPALABase):
    def __init__(self, env_name, model_init_method, save_dir, vision_dim, propri_dim, action_dim):
        super().__init__(env_name, model_init_method, save_dir, vision_dim, propri_dim, action_dim)

    def _arrange(self, repeat, cpu_per_worker, cpu_per_learner):
        n_workers = len(self.env_name) * repeat
        if n_workers * cpu_per_worker >= _N_CPU:
            raise ValueError('CPU resource not enough')
        n_learners = int((_N_CPU - 3 - n_workers * cpu_per_worker) // cpu_per_learner)
        world_size = 1 + n_workers + n_learners
        return n_workers, n_learners, world_size

    def train(self, max_step, max_episodes, model_update_freq=5, batch_size=10, repeat=1,
              simulator_params={}, learner_params={}, cpu_per_worker=2, cpu_per_learner=2):

        self.max_step = max_step
        n_workers, n_learners, world_size = self._arrange(repeat, cpu_per_worker, cpu_per_learner)

        training_done = Value('I', 0) # 'I' unsigned int

        param_server = Process(target=run_parameter_server, 
                               args=(0, world_size, training_done, n_learners, self.save_dir, n_workers,
                                     max_step, self.vision_dim, self.propri_dim, self.action_dim))
        param_server.start()

        print('Setting %d simulators...' % (repeat * len(self.env_name)))
        simulators = []
        for k in range(repeat):
            for i, env_i in enumerate(self.env_name):
                j = k * len(self.env_name) + i + 1
                simulator = DistributedAgent(j, ('cpu', cpu_per_worker, world_size, n_workers), 
                                             (training_done, n_learners),
                                             env_i, max_step,
                                             **simulator_params)
                simulator.start()
                simulators.append(simulator)

        time.sleep(5)

        print('Setting %d learners on CPU...' % n_learners)
        max_episodes_per_learner = int(max_episodes // n_learners)
        ready = (Value('i', 0), Event(), Event())
        pbar_state = Value('i', 0)
        learners = []
        for i in range(1 + n_workers, world_size):
            learner = Learner(i, ('cpu', cpu_per_learner, world_size, n_learners), 
                              training_done, ready, pbar_state, 
                              p_hat=1, c_hat=1,
                              max_episodes=max_episodes_per_learner, batch_size=batch_size,
                              distributed=True, **learner_params)
            learner.start()
            learners.append(learner)
        assert len(learners) == n_learners

        while pbar_state.value == 0:
            time.sleep(0.1)
        with tqdm.tqdm(total=max_episodes_per_learner * n_learners) as pbar:
            while pbar_state.value < max_episodes_per_learner * n_learners:
                pbar.n = int(pbar_state.value)
                pbar.refresh()
                time.sleep(1)
        
        for learner in learners:
            learner.join()

        if training_done.value != n_learners:
            print('Learner terminated with error. Kill all processes.')
            for simulator in simulators:
                simulator.kill()
            return

        for simulator in simulators:
            print('+', end='')
            simulator.join()

        print('All simulators terminated successfully')

        param_server.join()

        # self.model, _ = load_checkpoint(self.model, os.path.join(self.save_dir, 'model.pt'))

