import os, time, random
import copy
import numpy as np
import torch
from torch.multiprocessing import Process, Queue, Value, Event, Manager

from virtual_rodent.utils import load_checkpoint
from virtual_rodent.environment import MAPPER

_N_GPU = torch.cuda.device_count()
_N_CPU = int(os.environ['SLURM_NTASKS'])

class SampleQueue:
    def __init__(self, max_len, n_workers, T, vision_dim, propri_dim, action_dim):
        self._max_len = max_len
        self._n_workers = n_workers
        self._max_len_per_worker = max_len // n_workers

        self._vision_cache = torch.zeros(self._max_len_per_worker, n_workers, T, *vision_dim,
                                         dtype=torch.float32)
        self._vision_cache.share_memory_()
        self._propri_cache = torch.zeros(self._max_len_per_worker, n_workers, T, *propri_dim,
                                         dtype=torch.float32)
        self._propri_cache.share_memory_()
        self._action_cache = torch.zeros(self._max_len_per_worker, n_workers, T, action_dim,
                                         dtype=torch.float32)
        self._action_cache.share_memory_()
        self._reward_cache = torch.zeros(self._max_len_per_worker, n_workers, T, 1,
                                         dtype=torch.float32)
        self._reward_cache.share_memory_()
        self._log_policy_cache = torch.zeros(self._max_len_per_worker, n_workers, T, 1,
                                         dtype=torch.float32)
        self._log_policy_cache.share_memory_()
        self._done_cache = torch.zeros(self._max_len_per_worker, n_workers, T + 1,
                                         dtype=torch.bool)
        self._done_cache.share_memory_()

        self.__idx = torch.zeros(n_workers, dtype=torch.int32)
        self.__idx.share_memory_()
        self.__num = torch.zeros(n_workers, dtype=torch.int32)
        self.__num.share_memory_()

    def put(self, worker_id, item):
        b = self.__idx[worker_id]
        assert b >= 0 and b < self._max_len_per_worker
        self._vision_cache[b, worker_id] = item['vision'].detach().cpu().clone()
        self._propri_cache[b, worker_id] = item['proprioception'].detach().cpu().clone()
        self._action_cache[b, worker_id] = item['action'].detach().cpu().clone()
        self._reward_cache[b, worker_id] = item['reward'].detach().cpu().unsqueeze(-1).clone()
        self._log_policy_cache[b, worker_id] = item['log_policy'].detach().cpu().clone()
        assert item['done'].detach().cpu()[0]
        self._done_cache[b, worker_id] = item['done'].detach().cpu().clone()
        assert self._done_cache[b, worker_id].sum() >= 1, self._done_cache[b, worker_id]
        self.__idx[worker_id] = int((b + 1) % self._max_len_per_worker) # Circulate
        self.__num[worker_id] += 1

    def sample(self, num):
        assert (self.__num > 0).all()
        wid = [random.randint(0, self._n_workers - 1) for _ in range(num)] # end inclusive
        eid = [random.randint(0, min(self._max_len_per_worker, self.__num[i]) - 1) for i in wid]
        batch = dict(
                vision=torch.stack([self._vision_cache[eid[i], wid[i]]
                                    for i in range(num)], 1),
                proprioception=torch.stack([self._propri_cache[eid[i], wid[i]]
                                            for i in range(num)], 1),
                action=torch.stack([self._action_cache[eid[i], wid[i]]
                                    for i in range(num)], 1),
                log_policy=torch.stack([self._log_policy_cache[eid[i], wid[i]]
                                        for i in range(num)], 1),
                reward=torch.stack([self._reward_cache[eid[i], wid[i]]
                                    for i in range(num)], 1),
                done=torch.stack([self._done_cache[eid[i], wid[i]]
                                  for i in range(num)], 1)
            )
        for i in range(num):
            a = self._done_cache[eid[i], wid[i]].sum()
            assert a >= 1, (a, eid[i], wid[i], self.__num[wid[i]], self.__idx[wid[i]])
        return batch

    def __len__(self):
        if (self.__num == 0).any():
            return 0 # Not ready
        else:
            return int(torch.clamp(self.__num, max=self._max_len_per_worker).sum())


class TrainingTerminated(Exception):
    pass


class IMPALABase:
    def __init__(self, env_name, model, save_dir, vision_dim, propri_dim, action_dim):
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
        self.vision_dim = vision_dim
        self.propri_dim = propri_dim
        self.action_dim = action_dim

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def shared_training_resources(self):
        training_done = Value('I', 0) # 'I' unsigned int
        state_dict = Manager().dict(copy.deepcopy(self.model.state_dict()))
        record_queue = Queue()
        print(1)
        sample_queue = SampleQueue(1000, self.n_workers, self.max_step,
                                   self.vision_dim, self.propri_dim, self.action_dim)
        print(2)
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
