import os, time
import copy
import numpy as np
import torch
from torch.multiprocessing import Process, Queue, Value, Event, Manager

from virtual_rodent.utils import load_checkpoint
from virtual_rodent.environment import MAPPER

_N_GPU = torch.cuda.device_count()
_N_CPU = int(os.environ['SLURM_NTASKS'])

class TrainingTerminated(Exception):
    pass

class IMPALABase:
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

    def shared_training_resources(self):
        training_done = Value('I', 0) # 'I' unsigned int
        sample_queue = Queue()
        state_dict = Manager().dict(copy.deepcopy(self.model.state_dict()))
        record_queue = Queue()
        return training_done, sample_queue, state_dict, record_queue


class WorkerBase(Process):
    def __init__(self, ID, DEVICE_INFO, model=None, env_name=None):
        super().__init__()
        self.ID = ID
        self.DEVICE_TYPE, self.DEVICE_ID = DEVICE_INFO
        if self.DEVICE_TYPE not in ('cpu', 'gpu'):
            raise ValueError('Invalid device type')
        self.model = model
        self.env_name = env_name

    def setup(self):
        self.PID = os.getpid()

        if self.DEVICE_TYPE == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'
            torch.set_num_threads(self.DEVICE_ID)
            self.device = torch.device('cpu')
        elif self.DEVICE_TYPE == 'gpu': # dm_control/mujoco maps onto EGL_DEVICE_ID
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['EGL_DEVICE_ID'] = str(self.DEVICE_ID[0])
            if len(self.DEVICE_ID) > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.DEVICE_ID])
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:%d' % self.DEVICE_ID[0])

        if self.model is not None:
            if self.DEVICE_TYPE == 'gpu' and len(self.DEVICE_ID) > 1:
                self.model = torch.nn.DataParallel(self.model, dim=1).to(self.device) # Batch second
            else:
                self.model = self.model.to(self.device)

        if self.env_name is not None:
            self.env, self.propri_attr = MAPPER[self.env_name]()
