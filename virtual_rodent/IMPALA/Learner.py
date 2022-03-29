import os, time
import copy
from tqdm import tqdm
from queue import Empty # Exception
import torch
import torch.nn as nn
from torch.multiprocessing import Process

from virtual_rodent.utils import save_checkpoint, Cache

_ATTRIBUTES = ('vision', 'proprioception', 'action', 'log_policy', 'reward', 'done')

class Learner(Process):
    def __init__(self, DEVICE_ID, queue, training_done, model, state_dict, 
                 episodes, p_hat, c_hat, save_dir,
                 discount=0.99, entropy_bonus=True, clip_gradient=40, batch_size=5, lr=5e-4,
                 policy_weight=1, value_weight=1, entropy_weight=1e-2, save_window=None):
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.episodes = episodes
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.save_window = max(episodes//50, 2) if save_window is None else save_window
        self.save_dir = save_dir
        self.batch_cache = Cache(max_len=batch_size*2)

        # Variables
        if DEVICE_ID == 'all':
            self.device = torch.device('cuda')
            self.model = nn.DataParallel(model).to(self.device)
            self.batch_size = batch_size * torch.cuda.device_count()
        else:
            self.device = torch.device('cuda:%d' % DEVICE_ID)
            self.model = model.to(self.device)
            self.batch_size = batch_size

        self.best_loss = 0
        self.last_saved = 0

        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.state_dict = state_dict

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    
    def fetch_new_sample(self):
        try:
            x = self.queue.get(timeout=0.1)
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

# TODO: Parallel learner
    def run(self):
        PID = os.getpid()
        print('[%s] Training on %s...' % (PID, self.device))
        stats = {'total_loss': [], 'sum_rewards': [], 'sum_vtrace': [],
                'policy_loss': [], 'policy_weight': self.policy_weight, 
                'value_loss': [], 'value_weight': self.value_weight, 
                'entropy': [], 'entropy_weight': self.entropy_weight
                }

        # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
        # First batch takes longer to wait. 
        # Do it here for tqdm to estimate the runtime correctly
        batch = self.fetch_batch()

        for episode in tqdm(range(self.episodes), disable=False):
            self.optimizer.zero_grad()

            vision, propri = batch['vision'], batch['proprioception']
            # Output (T+1, batch, ...). Done state included; pi is torch.Distribution
            values, pis, reset_idx = self.model(vision, propri, batch['done'])
            
            rewards = batch['reward'][:-1].unsqueeze(-1) # (T, batch, 1)

            # The last action correspond to state T + 2 and is not in the calculation
            # Assume action elements are independent
            log_target_policy = pis.log_prob(batch['action'])[:-1].sum(-1, keepdim=True)
            log_behavior_policy = batch['log_policy'][:-1].sum(-1, keepdim=True)
            ratio = torch.exp(log_target_policy - log_behavior_policy)
            
            vtrace, p = self.V_trace(values, rewards, ratio)

            # Gradient descent for value function (L2)
            value_loss = (vtrace - values).pow(2).sum() / 2
            # Gradient ascent for policy only
            advantage = rewards + self.discount * vtrace[1:] - values[:-1]
            assert (log_target_policy < 0).all()
            assert (log_behavior_policy < 0).all()
            assert (ratio >= 0).all(), ratio[ratio < 0]
            assert (p > 0).all()
            policy_loss = (-p * log_target_policy * advantage.detach()).sum()
            #assert (policy_loss >= 0).all(), advantage[policy_loss < 0]

            total_loss = value_loss * self.value_weight + policy_loss * self.policy_weight

            if self.entropy_bonus: # Optional entropy bonus
                entropy = -(log_target_policy * torch.exp(log_target_policy)).sum()
                total_loss -= entropy * self.entropy_weight

            total_loss.backward()

            if self.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()
            self.model._update_episode()
            # print('learner', self.model._episode.data)
           
            stats['sum_rewards'].append(rewards.sum().item())
            stats['sum_vtrace'].append(vtrace.sum().item())
            stats['total_loss'].append(total_loss.item())
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            if self.entropy_bonus: # Optional entropy bonus
                stats['entropy'].append(entropy.item())
            
            if stats['total_loss'][-1] < 0:
                print(stats['policy_loss'][-1], stats['value_loss'][-1], stats['entropy'][-1])
            self.save(stats, episode)

            # CUDA cannot be shared as `load_state_dict()` will raise error when copying between GPUs
            self.state_dict.update(copy.deepcopy(self.model.cpu().state_dict())) 
            self.model = self.model.to(self.device) # .cpu is inplace for nn.module
            
            batch = self.fetch_batch()

        with self.training_done.get_lock():
            self.training_done.value = 1
        del self.batch_cache # Free the storage of variables from the producers

    def save(self, stats, episode):
        # Save the model if it's good
        if (stats['total_loss'][-1] <= self.best_loss and episode - self.last_saved > 1000) \
        or (episode - self.last_saved > self.episodes//10):
            self.best_loss = stats['total_loss'][-1]
            self.last_saved = episode
            save_checkpoint(self.model, episode, os.path.join(self.save_dir, 'model.pt'))

        if (episode + 1) % (self.episodes//10) == 0 or episode == self.episodes - 1:
            # Update the stats periodically
            torch.save(stats, os.path.join(self.save_dir, 'training_stats.pt'))
