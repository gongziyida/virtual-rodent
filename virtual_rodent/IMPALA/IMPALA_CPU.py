import os, time
import copy
import numpy as np
import torch
from torch.multiprocessing import Queue, Event, set_start_method

from virtual_rodent.utils import load_checkpoint
from .base import IMPALABase, _N_CPU
from .DistributedAgent import DistributedAgent
from .Learner import Learner
from .Recorder import StatsRecorder, VideoRecorder

set_start_method('spawn', force=True)

class IMPALA_CPU(IMPALABase):
    def __init__(self, env_name, model, save_dir):
        super().__init__(env_name, model, save_dir)
        # self.model = torch.jit.script(self.model)

    def train(self, max_step, max_episodes, model_update_freq=5, batch_size=10, repeat=1,
              simulator_params={}, learner_params={}, cpu_per_actor=2, cpu_per_learner=2):

        n_actor = len(self.env_name) * repeat
        if n_actor * cpu_per_actor >= _N_CPU:
            raise ValueError('CPU resource not enough')
        n_learner = int((_N_CPU - n_actor * cpu_per_actor) // cpu_per_learner)

        training_done, sample_queue, state_dict, record_queue = self.shared_training_resources()

        print('Setting %d simulators...' % (repeat * len(self.env_name)))
        behavior_model = copy.deepcopy(self.model)
        behavior_model.share_memory()

        simulators = []
        action_traffic = []
        for k in range(repeat):
            for i, env_i in enumerate(self.env_name):
                j = k * len(self.env_name) + i
                simulator = DistributedAgent(j, ('cpu',cpu_per_actor), sample_queue, 
                                             (training_done, n_learner), behavior_model, 
                                             env_i, max_step, **simulator_params)
                simulator.start()
                simulators.append(simulator)

        time.sleep(5)

        self.model.share_memory()

        recorder = StatsRecorder(state_dict, record_queue, (training_done, n_learner), self.save_dir)

        print('Setting %d learners on CPU...' % n_learner)
        learners = []
        for i in range(n_learner):
            learner = Learner(i, ('cpu',cpu_per_learner), sample_queue, record_queue, 
                              training_done, self.model, state_dict, p_hat=1, c_hat=1,
                              max_episodes=int(max_episodes//n_learner), batch_size=batch_size,
                              **learner_params)
            learner.start()
            learners.append(learner)

        recorder.start()

        for learner in learners:
            learner.join()

        if training_done.value != n_learner:
            print('Learner terminated with error. Kill all processes.')
            for simulator in simulators:
                simulator.kill()
            return

        for simulator in simulators:
            simulator.join()

        recorder.join()

        self.model, _ = load_checkpoint(self.model, os.path.join(self.save_dir, 'model.pt'))


    def record(self, env_name=None, simulators_params={}, save_full_record={}, cpu_per_recorder=2):
        """
        parameters
        ----------
        env_name: list or None
            If None, runs in the environments used for training
        simulators_params: dict
            If not empty, the keys must be the name of the enviornments
            See virtual_rodent/simulation/simulate
        save_full_record: dict
            If not empty, the keys must be the name of the enviornments, and items be boolean.
            If True is given to any environment, the whole return from the simulator will be saved.
            See virtual_rodent/simulation/simulate
        """
        self.model, _ = load_checkpoint(self.model, os.path.join(self.save_dir, 'model.pt'))

        if env_name is None:
            env_name = self.env_name
        
        self.model.share_memory()
        recorders = [VideoRecorder(i, ('cpu', cpu_per_recorder), self.model, env_i,
                                   self.save_dir,
                                   simulators_params.get(env_i, {}),
                                   save_full_record.get(env_i, False))
                     for i, env_i in enumerate(env_name)]

        for recorder in recorders:
            recorder.start()

        for recorder in recorders:
            recorder.join()
