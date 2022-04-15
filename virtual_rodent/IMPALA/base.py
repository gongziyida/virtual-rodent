import os, time, tqdm
import copy
import numpy as np
import torch
from torch.multiprocessing import Queue, Value, Event, Manager, set_start_method

from virtual_rodent.utils import load_checkpoint
from .Agent import Agent
from .Simulator import Simulator
from .Learner import Learner
from .Recorder import Recorder

set_start_method('spawn', force=True)
_N_GPU = torch.cuda.device_count()
_N_CPU = int(os.environ['SLURM_NTASKS'])

class IMPALA:
    def __init__(self, env_name, model, save_dir):
        """ Multi-actor-single-learner IMPALA in Pytorch
        parameters
        ----------
        env_name: list of str
            list of the name of environment
            The number of environments determine the number of actors (processes)
        model: nn.Module
            model does not need to be on CUDA. It will be handled in the method
        """
        self.env_name = env_name
        self.model = model

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, max_step, max_episodes, model_update_freq=5, batch_size=10, repeat=1,
              simulator_params={}, learner_params={}, ratio_cpu=1/3):
        training_done = Value('I', 0) # 'I' unsigned int
        sample_queue = Queue()
        state_dict = Manager().dict(copy.deepcopy(self.model.state_dict()))

        # Processes
        print('Setting %d simulators...' % (repeat * len(self.env_name)))
        simulators = []
        action_traffic = []
        for k in range(repeat):
            for i, env_i in enumerate(self.env_name):
                action_queue = Queue()
                action_made = Event()
                input_given = Event()
                action_traffic_i = (action_queue, action_made, input_given)
                j = k * len(self.env_name) + i
                simulator = Simulator(j, 'cpu', sample_queue, training_done, 
                                      action_traffic_i, env_i, max_step,
                                      **simulator_params)
                simulator.start()
                action_traffic.append(action_traffic_i)
                simulators.append(simulator)

        n_cpu_left = _N_CPU - len(simulators) - 1
        print('Setting actor...')
        n_actor_cpu = int(n_cpu_left * ratio_cpu)
        actor_devices = n_actor_cpu if _N_GPU == 0 else (0,)
        actor = Agent(actor_devices, action_traffic, training_done, copy.deepcopy(self.model), 
                      state_dict, batch_size, max_step, model_update_freq)
        actor.start()

        time.sleep(5)

        print('Setting learner...')
        # self.model.share_memory()
        learner_devices = n_cpu_left - n_actor_cpu if _N_GPU <= 1 else tuple(range(1, _N_GPU))
        learner = Learner(learner_devices, sample_queue, training_done, self.model, 
                          state_dict, p_hat=1, c_hat=1,
                          save_dir=self.save_dir, 
                          max_episodes=max_episodes, 
                          batch_size=batch_size, 
                          **learner_params)
        learner.start()

        learner.join()

        if training_done.value != 1:
            print('Learner terminated with error. Kill all processes.')
            actor.kill()
            for simulator in simulators:
                simulator.kill()
            return

        for (_, action_made, input_given) in action_traffic:
            action_made.set()
            input_given.set()

        actor.join()
        #actor.terminate()

        for simulator in simulators:
            simulator.join()
            #simulator.terminate()

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
        
        if _N_GPU > 0:
            recorders = [Recorder((i%_N_GPU,), copy.deepcopy(self.model), env_i,
                                  self.save_dir,
                                  simulators_params.get(env_i, {}),
                                  save_full_record.get(env_i, False))
                         for i, env_i in enumerate(env_name)]
        else:
            n = _N_CPU // len(env_name)
            recorders = [Recorder(int(i*n), copy.deepcopy(self.model), env_i,
                                  self.save_dir,
                                  simulators_params.get(env_i, {}),
                                  save_full_record.get(env_i, False))
                         for i, env_i in enumerate(env_name)]

        for recorder in recorders:
            recorder.start()

        for recorder in recorders:
            recorder.join()
