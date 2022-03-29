import os, time
import copy
import torch
from torch.multiprocessing import Queue, Value, Event, Manager, set_start_method

from .Actor import Actor
from .Simulator import Simulator
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

    def train(self, max_step, max_episode, repeat=1):
        training_done = Value('I', 0) # 'I' unsigned int
        sample_queue = Queue()
        state_dict = Manager().dict(copy.deepcopy(self.model.state_dict()))

        # Processes
        simulators = []
        action_traffic = []
        for _ in range(repeat):
            for i, env_i in enumerate(self.env_name):
                action_queue = Queue()
                action_made = Event()
                input_given = Event()
                assert not action_made.is_set()
                assert not input_given.is_set()
                action_traffic_i = (action_queue, action_made, input_given)
                simulator = Simulator('cpu', sample_queue, training_done, 
                                      action_traffic_i, env_i, max_step)
                simulator.start()
                action_traffic.append(action_traffic_i)
                simulators.append(simulator)

        actor = Actor(0, action_traffic, training_done, 
                      copy.deepcopy(self.model), state_dict)
        actor.start()

        learner_devices = (0,) if _N_CUDA == 1 else tuple(range(1, _N_CUDA))
        learner = Learner(learner_devices, sample_queue, training_done, 
                          self.model, state_dict, max_episode, p_hat=2, c_hat=1,
                          save_dir=self.save_dir)
        learner.start()

        learner.join()

        if training_done.value != 1:
            print('Learner terminated with error. Kill all processes.')
            actor.kill()
            for simulator in simulators:
                simulator.kill()
            return


        actor.join()

        for simulator in simulators:
            simulator.join()

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
