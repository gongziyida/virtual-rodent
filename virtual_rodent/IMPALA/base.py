import os, time
import torch
from torch.multiprocessing import Queue, Value, set_start_method

from .Actor import Actor
from .Learner import Learner

set_start_method('spawn', force=True)
_N_CUDA = torch.cuda.device_count()

class IMPALA:
    def __init__(self, env_name, model, save_dir, max_step, max_episode):
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
        self.model = model.to('cuda:%d' % (_N_CUDA - 1))
        self.save_dir = save_dir
        self.max_step = max_step
        self.max_episode = max_episode

        self.model.share_memory()
        self._training_done = Value('I', 0) # 'I' unsigned int
        self._sample_queue = Queue()
        
        n_cuda_actor = _N_CUDA if _N_CUDA == 1 else _N_CUDA - 1
        # Processes
        self._actors = [Actor(i%n_cuda_actor, self._sample_queue, self._training_done, 
                              model, env_i, max_step)
                        for i, env_i in enumerate(env_name)]

        self._learner = Learner(_N_CUDA - 1, self._sample_queue, self._training_done, 
                                model=self.model, episodes=max_episode, p_hat=2, c_hat=1, 
                                save_dir=save_dir)

    def __call__(self):
       # simulate (no grad)
        for actor in self._actors:
            actor.start()
        self._learner.start()

        while self._sample_queue.empty():
            time.sleep(0.1)

        self._learner.join()

        if self._training_done.value != 1:
            print('Learner terminated with error')
            with self._training_done.get_lock():
                self._training_done.value = 1

        for actor in self._actors:
            actor.join()


    def save(self, episode):
        torch.save(dict(model=self.model.state_dict(), 
                        parameters=self.parameters,
                        episode=episode),
                   os.path.join(self.save_path, '%d.pt' %  episode))
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.parameters = checkpoint['parameters']
        print('Loaded checkpoint at Episode %d' % checkpoint['episode'])
