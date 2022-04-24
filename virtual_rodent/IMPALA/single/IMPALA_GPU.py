import os, time
import copy
import numpy as np
import torch
from torch.multiprocessing import Queue, Event, set_start_method

from virtual_rodent.utils import load_checkpoint
from .base import IMPALABase, _N_GPU
from .CentralizedAgent import CentralizedAgent
from .Simulator import Simulator
from .Learner import Learner
from .Recorder import StatsRecorder, VideoRecorder

set_start_method('spawn', force=True)

class IMPALA_GPU(IMPALABase):
    def __init__(self, env_name, model, save_dir, vision_dim, propri_dim, action_dim):
        super().__init__(env_name, model, save_dir, vision_dim, propri_dim, action_dim)

    def train(self, max_step, max_episodes, model_update_freq=5, batch_size=10, repeat=1,
              simulator_params={}, learner_params={}):

        self.max_step = max_step
        self.n_workers = repeat * len(self.env_name)
        training_done, sample_queue, state_dict, record_queue = self.shared_training_resources()

        # Processes
        print('Setting %d simulators...' % self.n_workers)
        simulators = []
        action_traffic = []
        for k in range(repeat):
            for i, env_i in enumerate(self.env_name):
                action_queue = Queue()
                action_made = Event()
                input_given = Event()
                action_traffic_i = (action_queue, action_made, input_given)
                j = k * len(self.env_name) + i
                simulator = Simulator(j, ('cpu', 1), sample_queue, training_done, 
                                      action_traffic_i, env_i, max_step,
                                      **simulator_params)
                simulator.start()
                action_traffic.append(action_traffic_i)
                simulators.append(simulator)

        print('Setting worker...')
        behavior_model = copy.deepcopy(self.model)
        worker = CentralizedAgent(('gpu',(0,)), action_traffic, training_done, 
                    behavior_model, state_dict, batch_size, max_step, model_update_freq)
        worker.start()

        time.sleep(3)

        recorder = StatsRecorder(state_dict, record_queue, (training_done, 1), self.save_dir)

        print('Setting 1 learner on GPU...')
        learner_devices = ('gpu',(0,)) if _N_GPU == 1 else ('gpu',tuple(range(1, _N_GPU)))
        learner = Learner(0, learner_devices, sample_queue, record_queue, training_done, 
                          self.model, state_dict, p_hat=1, c_hat=1,
                          max_episodes=max_episodes, batch_size=batch_size,
                          **learner_params)
        learner.start()
        recorder.start()

        learner.join()

        if training_done.value != 1:
            print('Learner terminated with error. Kill all processes.')
            worker.kill()
            for simulator in simulators:
                simulator.kill()
            return

        for (_, action_made, input_given) in action_traffic:
            action_made.set()
            input_given.set()

        worker.join()

        for simulator in simulators:
            simulator.join()

        recorder.join()

        self.model, _ = load_checkpoint(self.model, os.path.join(self.save_dir, 'model.pt'))


    def record(self, env_name=None, simulators_params={}, save_full_record={}):
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
        
        recorders = [VideoRecorder(i, ('gpu',(i%_N_GPU,)), copy.deepcopy(self.model), 
                                   env_i, self.save_dir,
                                   simulators_params.get(env_i, {}),
                                   save_full_record.get(env_i, False))
                     for i, env_i in enumerate(env_name)]

        for recorder in recorders:
            recorder.start()

        for recorder in recorders:
            recorder.join()
