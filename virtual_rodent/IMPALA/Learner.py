import os, time, pickle
import copy
from tqdm import tqdm
from queue import Empty # Exception
import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import Process

from virtual_rodent.network.helper import fetch_reset_idx
from virtual_rodent.utils import save_checkpoint, Cache

_ATTRIBUTES = ('vision', 'proprioception', 'action', 'log_policy', 'reward', 'done')


class Learner(Process):
    def __init__(self, DEVICE_ID, queue, training_done, model, state_dict, 
                 n_batches, p_hat, c_hat, save_dir,
                 discount=0.99, entropy_bonus=True, clip_gradient=40, batch_size=5, lr=5e-4,
                 policy_weight=1, value_weight=0.5, entropy_weight=1e-2, reduction='mean', 
                 save_window=None):
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.save_dir = save_dir
        self.reduction = reduction
        self.batch_size = batch_size
        self.batch_cache = Cache(max_len=int(batch_size*1.5))
        self.episodes = int(n_batches * batch_size)
        self.save_window = max(self.episodes//50, 2) if save_window is None else save_window

        self.model = model

        self.last_saved = time.time()

        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.state_dict = state_dict

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, 
                                             weight_decay=0, eps=1e-4)

    def setup(self):
        if len(self.DEVICE_ID) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.DEVICE_ID])
            self.device = torch.device('cuda')
            self.model = nn.DataParallel(self.model, dim=1).to(self.device) # Batch second
        else:
            self.device = torch.device('cuda:%d' % self.DEVICE_ID[0])
            self.model = self.model.to(self.device)

        self.PID = os.getpid()
        print('[%s] Training on cuda %s' % (self.PID, self.DEVICE_ID))

        keys = ('total_loss', 'sum_rewards', 'sum_vtrace', 'policy_loss', 'policy_weight', 
                'value_loss', 'value_weight', 'entropy', 'entropy_weight')
        stats = {k: np.ones(self.episodes) * np.inf for k in keys}
        stats['policy_weight'][:] = self.policy_weight
        stats['value_weight'][:]= self.value_weight
        stats['entropy_weight'][:] = self.entropy_weight

        return stats

    
    def fetch_new_sample(self):
        try:
            x = self.queue.get(timeout=0.01)
            sample = dict()
            for k in x.keys():
            # Map to current device
                sample[k] = x[k].to(self.device) if torch.is_tensor(x[k]) else x[k]
            del x # Free the link to the Providers' memory
            self.batch_cache.add(sample)
        except Empty:
            return

    def fetch_batch(self):
        counter = 0
        while not self.queue.empty() and counter < self.batch_size: # Consume
            self.fetch_new_sample()
            counter += 1
        while len(self.batch_cache) <= self.batch_size: # Get enough samples first
            if self.queue.empty():
                time.sleep(0.1)
            self.fetch_new_sample()
        
        batch = self.batch_cache.sample(num=self.batch_size) # list of dict
        returns = {k: [b[k] for b in batch] for k in _ATTRIBUTES}

        for k in returns.keys(): # Stack: (T+1 or T, batch, ...) if tensor
            if torch.is_tensor(returns[k][0]):
                returns[k] = torch.stack(returns[k], dim=1)
        assert returns['vision'].shape[:2] == returns['proprioception'].shape[:2]
        return returns


    def V_trace(self, values, rewards, target_behavior_ratio, reset_idx):
        with torch.no_grad():
            p = torch.clamp(target_behavior_ratio, max=self.p_hat)
            c = torch.clamp(target_behavior_ratio, max=self.c_hat)
            dV = (rewards + self.discount * values[1:] - values[:-1]) * p

            vtrace = torch.zeros(*values.shape).to(self.device) # (T+1, batch, 1)
            vtrace[-1] = values[-1]
            for i in range(1, len(vtrace)): 
                j = len(vtrace) - 1 - i # Backward
                vtrace[j] = values[j] + dV[j] + \
                            self.discount * c[j] * (vtrace[j+1] - values[j+1])
        return vtrace.detach(), p.detach()

    def run(self):
        stats = self.setup()

        # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
        # First batch takes longer to wait. 
        # Do it here for tqdm to estimate the runtime correctly
        batch = self.fetch_batch()

        for k in tqdm(range(0, self.episodes, self.batch_size), disable=False):
            self.optimizer.zero_grad()

            vision, propri = batch['vision'], batch['proprioception']
            reset_idx = fetch_reset_idx(batch['done'], vision.shape[0], self.batch_size)
            # Output (T+1, batch, 1). Done state included 
            values, _, log_target_policy, entropy = self.model((vision, propri, reset_idx),
                                                               action=batch['action'])
            
            rewards = batch['reward'][:-1].unsqueeze(-1) # (T, batch, 1)

            # The last action correspond to state T + 2 and is not in the calculation
            log_target_policy = log_target_policy[:-1]
            log_behavior_policy = batch['log_policy'][:-1]
            ratio = torch.exp(log_target_policy - log_behavior_policy)
            
            vtrace, p = self.V_trace(values, rewards, ratio, reset_idx)

            # Gradient descent for value function (L2). The last is 0 anyway
            # (T+1, batch, 1) -> (batch,)
            value_loss = ((vtrace - values).pow(2) / 2).mean((0, 2))
            # Gradient ascent for policy only
            advantage = rewards + self.discount * vtrace[1:] - values[:-1]
            # (T, batch, 1) -> (batch,)
            policy_loss = -(p * log_target_policy * advantage.detach()).mean((0, 2))
            # (T, batch, 1) -> (batch,)
            entropy = entropy.mean((0, 2))

            total_loss = value_loss * self.value_weight + policy_loss * self.policy_weight - \
                         entropy * self.entropy_weight

            stats['sum_rewards'][k:k+self.batch_size] = rewards.detach().cpu().numpy().mean((0, 2))
            stats['sum_vtrace'][k:k+self.batch_size] = vtrace.detach().cpu().numpy().mean((0, 2))
            stats['total_loss'][k:k+self.batch_size] = total_loss.detach().cpu().numpy()
            stats['policy_loss'][k:k+self.batch_size] = policy_loss.detach().cpu().numpy()
            stats['value_loss'][k:k+self.batch_size] = value_loss.detach().cpu().numpy()
            stats['entropy'][k:k+self.batch_size] = entropy.detach().cpu().numpy()

            total_loss = total_loss.mean() if self.reduction == 'mean' else total_loss.sum()
            total_loss.backward()

            if self.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()
            
            self.save(stats, k)

            # CUDA cannot be shared as `load_state_dict()` will raise error when copying between GPUs
            self.state_dict.update(copy.deepcopy(self.model.cpu().state_dict())) 
            self.model = self.model.to(self.device) # .cpu is inplace for nn.module
            
            batch = self.fetch_batch()

        with self.training_done.get_lock():
            self.training_done.value = 1
        del self.batch_cache # Free the storage of variables from the producers


    def save(self, stats, episode):
        t = time.time() - self.last_saved
        if (t > 1200) or (episode % (self.episodes//10) == 0) or (episode == self.episodes - 1):
            save_checkpoint(self.model, episode, os.path.join(self.save_dir, 'model.pt'))

        if episode % (self.episodes//10) == 0 or episode == self.episodes - self.batch_size:
            # Update the stats periodically
            with open(os.path.join(self.save_dir, 'training_stats.pkl'), 'wb') as f:
                pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
