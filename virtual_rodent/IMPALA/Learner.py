import os, time
from queue import Empty # Exception
import torch
import torch.nn as nn
from torch.multiprocessing import Process

from virtual_rodent.utils import save_checkpoint, Cache

_ATTRIBUTES = ('vision', 'proprioception', 'action', 'log behavior policy', 'reward', 'done')

class Learner(Process):
    def __init__(self, EGL_ID, queue, training_done, model, episodes, p_hat, c_hat, save_dir,
                 discount=0.99, entropy_bonus=True, clip_gradient=40, batch_size=3, 
                 policy_weight=1, value_weight=1, entropy_weight=1e-2, save_every=None):
        super().__init__()
        # Constants
        self.EGL_ID = EGL_ID
        self.device = torch.device('cuda:%d' % EGL_ID) 
        self.episodes = episodes
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.save_every = max(episodes//10, 2) if save_every is None else save_every
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.batch_cache = Cache(max_len=batch_size*2)
        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.model = model # Assume model is already on this device

        self.optimizer = torch.optim.Adam(model.parameters())

    
    def fetch_new_sample(self):
        try:
            x = self.queue.get(timeout=1)
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
                time.sleep(0.5)
            self.fetch_new_sample()
        
        batch = self.batch_cache.sample(num=self.batch_size) # list of dict
        returns = {k: [b[k] for b in batch] for k in _ATTRIBUTES}

        for k in returns.keys(): # Stack: (T+1 or T, batch, ...) if tensor
            if torch.is_tensor(returns[k][0]):
                returns[k] = torch.stack(returns[k], dim=1)
        assert returns['vision'].shape[:2] == returns['proprioception'].shape[:2]
        return returns


    def V_trace(self, values, rewards, target_behavior_ratio):
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
        stats = {'total_loss': [], 'sum_rewards': [], 'sum_vtrace': [],
                'policy_loss': [], 'policy_weight': self.policy_weight, 
                'value_loss': [], 'value_weight': self.value_weight, 
                'entropy': [], 'entropy_weight': self.entropy_weight
                }
        for episode in range(self.episodes):
            # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
            batch = self.fetch_batch()

            print('Episode %d' % episode) 
            
            self.optimizer.zero_grad()

            vision, propri = batch['vision'], batch['proprioception']
            # Output (T+1, batch, ...). Done state included; pi is torch.Distribution
            values, pis, reset_idx = self.model(vision, propri, batch['done'])
            
            rewards = batch['reward'][:-1].unsqueeze(-1) # (T, batch, 1)

            # The last action correspond to state T + 2 and is not in the calculation
            # Assume action elements are independent
            log_target_policy = pis.log_prob(batch['action'])[:-1].sum(-1, keepdim=True)
            log_behavior_policy = batch['log behavior policy'][:-1].sum(-1, keepdim=True)
            ratio = torch.exp(log_target_policy - log_behavior_policy)
            
            vtrace, p = self.V_trace(values, rewards, ratio)

            # Gradient descent for value function (L2)
            value_loss = (vtrace - values).pow(2).sum()
            # Gradient ascent for policy only
            advantage = rewards + self.discount * vtrace[1:] - values[:-1]
            policy_loss = (-p * log_target_policy * advantage.detach()).sum()

            total_loss = value_loss * self.value_weight + policy_loss * self.policy_weight

            if self.entropy_bonus: # Optional entropy bonus
                entropy = -(log_target_policy * torch.exp(log_target_policy)).sum()
                total_loss += entropy * self.entropy_weight

            total_loss.backward()

            if self.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()
            
            stats['sum_rewards'].append(rewards.sum())
            stats['sum_vtrace'].append(vtrace.sum())
            stats['total_loss'].append(total_loss.item())
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            if self.entropy_bonus: # Optional entropy bonus
                stats['entropy'].append(entropy.item())
                
            if (episode + 1) % self.save_every == 0 or episode == self.episodes - 1:
                save_checkpoint(self.model, episode + 1, 
                                os.path.join(self.save_dir, 'model%d.pt'%episode))
                torch.save(stats, os.path.join(self.save_dir, 'training_stats.pt'))

        with self.training_done.get_lock():
            self.training_done.value = 1
