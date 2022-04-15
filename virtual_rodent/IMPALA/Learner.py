import os, time, pickle
import copy
from tqdm import tqdm
from queue import Empty # Exception
import torch
import torch.nn as nn
from torch.multiprocessing import Process

from virtual_rodent.network.helper import fetch_reset_idx
from virtual_rodent.utils import save_checkpoint, Cache

_ATTRIBUTES = ('vision', 'proprioception', 'action', 'log_policy', 'reward', 'done')


class Learner(Process):
    def __init__(self, DEVICE_ID, queue, training_done, model, state_dict, 
                 max_episodes, p_hat, c_hat, save_dir,
                 discount=0.99, entropy_bonus=True, clip_gradient=1, batch_size=5, lr=1e-4,
                 actor_weight=1, critic_weight=0.5, entropy_weight=1e-2, reduction='mean', 
                 lr_scheduler=False, save_window=None):
        super().__init__()
        # Constants
        if hasattr(DEVICE_ID, '__iter__'):
            self.DEVICE_ID = DEVICE_ID
            self.N_DEVICES = len(DEVICE_ID)
        elif type(DEVICE_ID) == int:
            self.DEVICE_ID = 'cpu'
            self.N_DEVICES = DEVICE_ID
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.save_dir = save_dir
        self.reduction = reduction
        self.batch_size = batch_size
        self.batch_cache = Cache(max_len=int(batch_size*20))
        self.episode = 0
        self.max_episodes = max_episodes
        self.save_window = max(self.max_episodes//50, 2) if save_window is None else save_window

        self.rewards = dict()

        self.model = model

        self.last_saved = time.time()

        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.state_dict = state_dict

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-4)
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                             base_lr=lr, max_lr=lr * 5, step_size_up=10000, step_size_down=10000, 
                             mode='triangular2')
        else: 
            self.scheduler = None


    def setup(self):
        self.PID = os.getpid()
        if self.DEVICE_ID == 'cpu':
            torch.set_num_threads(self.N_DEVICES)
            self.device = torch.device('cpu')
            print('[%s] Training on %d CPUs' % (self.PID, self.N_DEVICES))
        else:
            if self.N_DEVICES > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.DEVICE_ID])
                self.device = torch.device('cuda')
                self.model = nn.DataParallel(self.model, dim=1).to(self.device) # Batch second
            else:
                self.device = torch.device('cuda:%d' % self.DEVICE_ID[0])
                self.model = self.model.to(self.device)

            print('[%s] Training on cuda %s' % (self.PID, self.DEVICE_ID))

        keys = ('total_loss', 'mean_vtrace', 'mean_values', 
                'actor_loss', 'critic_loss', 'entropy', 'learning_rate')
        stats = {k: [] for k in keys}

        return stats

    
    def fetch_new_sample(self):
        try:
            x = self.queue.get(timeout=0.05)
            env_name = x['env_name']
            sample = dict()
            for k in x.keys(): # Map to current device
                if k == 'env_name':
                    continue
                sample[k] = x[k].to(self.device) if torch.is_tensor(x[k]) else x[k]
            self.batch_cache.add(sample)
            rewards = sample['reward'].detach().cpu().numpy()
            rewards = rewards.sum() / rewards.shape[0]
            self.rewards[env_name].append(rewards)
            self.episode += 1
        except Empty:
            return 
        except KeyError:
            self.rewards[env_name] = [rewards]

    def fetch_batch(self):
        while len(self.batch_cache) < self.batch_size: # Get enough samples first
            self.fetch_new_sample()
        while not self.queue.empty():
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
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            assert next_values.shape == values.shape
            assert (next_values[-1] == 0).all()
            for i in reset_idx:
                assert len(i) == 2, reset_idx
            dV = (rewards + self.discount * next_values - values) * p
            
            vtrace = torch.zeros_like(values) # (T, batch, 1)
            # vtrace[-1] = values[-1]
            for i in range(1, vtrace.shape[0]): 
                j = vtrace.shape[0] - 1 - i # Backward
                diff = 0 if i == 0 else self.discount * c[j] * (vtrace[j+1] - values[j+1])
                vtrace[j] = values[j] + dV[j] + diff
                """
                if i > 0:
                    print(i, 'dV %.3f %.3f' % (dV[j].mean().item(), dV[j].var().item()), 
                             '\tc %.3f %.3f' % (c[j].mean().item(), c[j].var().item()), 
                             '\tvtrace+1 %.3f %.3f' % (vtrace[j+1].mean().item(), vtrace[j+1].var().item()),
                             '\tvalues+1 %.3f %.3f' % (values[j+1].mean().item(), values[j+1].var().item()))
                """
            
            advantage = torch.zeros_like(values)
            advantage[:-1] = rewards[:-1] + self.discount * vtrace[1:] - values[:-1]
            advantage[-1] = rewards[-1] - values[-1]
            
            """
            vtrace_ = torch.zeros_like(values) # (T, batch, 1)
            for s in range(len(vtrace_)):
                aux = dV[s] if s != len(vtrace_) - 1 else 0 # t == s
                for t in range(s + 1, T): # t = s + 1 to s + T - 1
                    aux += self.discount**(t - s) * torch.prod(c[s:t], dim=0) * dV[t]
                vtrace_[s] = values[s] + aux
            assert (vtrace_ - vtrace) < 1e-10
            """
            return vtrace.detach(), p.detach(), advantage.detach()


    def run(self):
        stats = self.setup()

        # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
        # First batch takes longer to wait. 
        # Do it here for tqdm to estimate the runtime correctly
        batch = self.fetch_batch()

        for k in tqdm(range(0, self.max_episodes, self.batch_size), disable=False):
            self.optimizer.zero_grad()

            vision, propri = batch['vision'], batch['proprioception']
            reset_idx = fetch_reset_idx(batch['done'], vision.shape[0], self.batch_size)
            # Output (T+1, batch, 1). Done state included 
            values, (a, log_target_policy, entropy) = self.model((vision, propri, reset_idx),
                                                                  action=batch['action'])
            # print(log_target_policy)
            rewards = batch['reward'].unsqueeze(-1) # (T, batch, 1)

            # The last action correspond to state T + 2 and is not in the calculation
            log_target_policy = log_target_policy
            log_behavior_policy = batch['log_policy']
            ratio = torch.exp(log_target_policy - log_behavior_policy)
            
            vtrace, p, advantage = self.V_trace(values, rewards, ratio, reset_idx)

            # Gradient descent for value function (L2). 
            # (T+1, batch, 1) -> (batch,)
            critic_loss = nn.functional.mse_loss(values, vtrace, reduction='none').mean((0, 2)) 
            # Gradient ascent for policy only
            # (T, batch, 1) -> (batch,)
            actor_loss = (-p * log_target_policy * advantage).mean((0, 2))
            # (T, batch, 1) -> (batch,)
            entropy = entropy.mean((0, 2)) 
            # print(critic_loss, actor_loss, entropy, sep='\n')

            total_loss = critic_loss * self.critic_weight + actor_loss * self.actor_weight - \
                         entropy * self.entropy_weight

            stats['mean_vtrace'] += list(vtrace.detach().cpu().numpy().mean((0, 2)))
            stats['mean_values'] += list(values.detach().cpu().numpy().mean((0, 2)))
            stats['total_loss'] += list(total_loss.detach().cpu().numpy())
            stats['actor_loss'] += list(actor_loss.detach().cpu().numpy())
            stats['critic_loss'] += list(critic_loss.detach().cpu().numpy())
            stats['entropy'] += list(entropy.detach().cpu().numpy())

            total_loss = total_loss.mean() if self.reduction == 'mean' else total_loss.sum()
            total_loss.backward()

            if self.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                aux = [self.scheduler.get_last_lr()] * self.batch_size
                stats['learning_rate'] += aux
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
        if (t > 1200) or (episode % (self.max_episodes//10) == 0) or (episode == self.max_episodes - 1):
            save_checkpoint(self.model, episode, os.path.join(self.save_dir, 'model.pt'))

        if episode % (self.max_episodes//10) == 0 or episode == self.max_episodes - self.batch_size:
            # Update the stats periodically
            with open(os.path.join(self.save_dir, 'training_stats.pkl'), 'wb') as f:
                pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.save_dir, 'rewards.pkl'), 'wb') as f:
                pickle.dump(self.rewards, f, protocol=pickle.HIGHEST_PROTOCOL)
