import time
from queue import Empty # Exception
import torch
from torch.multiprocessing import Process

from virtual_rodent.utils import save_checkpoint

class Learner(Process):
    def __init__(self, EGL_ID, queue, training_done, model, episodes, save_dir, 
                 save_every=None, batch_size=3):
        super().__init__()
        # Constants
        self.EGL_ID = EGL_ID
        self.device = torch.device('cuda:%d' % EGL_ID) 
        self.episodes = episodes
        self.save_every = episodes // 10 if save_every is None else save_every
        self.save_dir = save_dir
        self.batch_size = batch_size
        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.model = model


    def run(self):
        time.sleep(10) # Wait for the simulators to init and gather some samples
        for episode in range(self.episodes):
            n_batch = 0
            # Get enough samples before proceeding to train
            while n_batch < self.batch_size:
                try:
                    x = self.queue.get(timeout=1)
                    # TODO: map each of x to current device
                    print(n_batch, len(x))
                    del x
                    n_batch += 1
                except Empty:
                    pass
            print('Episode %d' % episode)
            # if episode % self.save_every == 0 and episode > 0:
                # save_checkpoint(self.model, episode)

        with self.training_done.get_lock():
            self.training_done.value = 1
