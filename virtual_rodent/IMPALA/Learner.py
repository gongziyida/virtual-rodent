import time
from queue import Empty # Exception
import torch
from torch.multiprocessing import Process

from virtual_rodent.utils import save_checkpoint

class Learner(Process):
    def __init__(self, EGL_ID, queue, training_done, model, episodes, discount, p_hat, c_hat,
                 save_dir, save_every=None, batch_size=3):
        super().__init__()
        # Constants
        self.EGL_ID = EGL_ID
        self.device = torch.device('cuda:%d' % EGL_ID) 
        self.episodes = episodes
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.save_every = episodes // 10 if save_every is None else save_every
        self.save_dir = save_dir
        self.batch_size = batch_size
        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.model = model

    
    def sample(self):
        n_batch = 0
        while n_batch < self.batch_size: # Get enough samples before training
            try:
                x = self.queue.get(timeout=1)
                if n_batch == 0:
                    returns = {k: [] for k in x.keys()}
                for k in returns.keys():
                    returns[k].append(x[k].to(self.device)) # Map to current device
                del x # Free the link to the Providers' memory
                n_batch += 1
            except Empty:
                pass

        for k in returns.keys(): # Stack, new shape (T+1 or T, batch, ...) for tensor
            if torch.is_tensor(returns[k][0]): 
                returns[k] = torch.stack(returns[k], dim=1)
        assert returns['vision'].shape == returns['proprioception'].shape
        return returns

    def run(self):
        time.sleep(5) # Wait for the simulators to init and gather some samples
        for episode in range(self.episodes):
            # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
            batch = self.sample()

            print('Episode %d' % episode) # TODO: summarywriter
            
            vision, propri = batch['vision'], batch['proprioception']
            values, pis = model(vision.view(-1, *vision.shape[2:]),
                                propri.view(-1, *propri.shape[2:]))
            
            log_pi_action = pis.log_prob(batch['action'])

            p = torch.clamp(, max=self.p_hat)
            c = torch.clamp(, max=self.c_hat)
            dV = (batch['reward'] + self.discount * (values[1:] - values[:-1])) * p

            vtrace = torch.zeros(*vision.shape[:2]).to(self.device) # (T+1, batch)
            for i in range(len(vtrace) - 1, -1, -1): # Backward
                vtrace[i] = 

            # TODO: run and v-trace
            # if episode % self.save_every == 0 and episode > 0:
                # save_checkpoint(self.model, episode)

        with self.training_done.get_lock():
            self.training_done.value = 1
