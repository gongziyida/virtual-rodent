import time
from tqdm import tqdm
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
        time.sleep(1) # Wait for the simulators to init and gather some samples
        for episode in tqdm(range(episodes)):
            n_batch = 0
            # Get enough samples before proceeding to train
            while n_batch < self.batch_size:
                try:
                    print(n_batch, self.queue.get(timeout=1))
                    n_batch += 1
                except Empty:
                    pass
            # if episode % self.save_every == 0 and episode > 0:
                # save_checkpoint(self.model, episode)

        with self.training_done.get_lock():
            self.training_done.value = 1
