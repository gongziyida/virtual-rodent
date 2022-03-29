import os, time
import copy
import torch
from torch.multiprocessing import Queue, Value, Manager, set_start_method

from .Actor import Actor
from .Learner import Learner
from .Recorder import Recorder

set_start_method('spawn', force=True)
_N_CUDA = torch.cuda.device_count()

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

    def train(self, max_step, max_episode, repeat=1, gpu_simulation=False):
        training_done = Value('I', 0) # 'I' unsigned int
        sample_queue = Queue()
        state_dict = Manager().dict(copy.deepcopy(self.model.state_dict()))

        # Processes
        actors = []
        for _ in range(repeat):
            for i, env_i in enumerate(self.env_name):
                if gpu_simulation:
                    n_cuda_actor = _N_CUDA if _N_CUDA == 1 else _N_CUDA - 1
                    actor = Actor(i%n_cuda_actor, sample_queue, training_done,
                                  copy.deepcopy(self.model), state_dict, 
                                  env_i, max_step)

                else:
                    actor = Actor('cpu', sample_queue, training_done,
                                  copy.deepcopy(self.model), state_dict, 
                                  env_i, max_step)
                actors.append(actor)
        
        if gpu_simulation:
            learner = Learner(_N_CUDA - 1, sample_queue, training_done, 
                              self.model, state_dict, max_episode, p_hat=2, c_hat=1,
                              save_dir=self.save_dir)
        else: # Occupy all gpus
            learner = Learner('all', sample_queue, training_done, 
                              self.model, state_dict, max_episode, p_hat=2, c_hat=1,
                              save_dir=self.save_dir)

        for actor in actors:
            actor.start()
        learner.start()

        learner.join()

        if training_done.value != 1:
            print('Learner terminated with error')
            with training_done.get_lock():
                training_done.value = 1
                while not sample_queue.empty(): # Clear
                    sample_queue.get(timeout=1)
                sample_queue.close()

        for actor in actors:
            actor.join()

        self.model.load_state_dict(state_dict)
        self.model = self.model.cpu()


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
        if env_name is None:
            env_name = self.env_name

        recorders = [Recorder(i%_N_CUDA, copy.deepcopy(self.model), env_i, 
                              self.save_dir,
                              simulators_params.get(env_i, {}),
                              save_full_record.get(env_i, False))
                     for i, env_i in enumerate(env_name)]

        for recorder in recorders:
            recorder.start()

        for recorder in recorders:
            recorder.join()
