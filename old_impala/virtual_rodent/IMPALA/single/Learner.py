import os, time
import copy
from tqdm import tqdm
from queue import Empty # Exception
import torch
import torch.nn as nn

from virtual_rodent.network.helper import fetch_reset_idx
from .base import WorkerBase

_ATTRIBUTES = ('vision', 'proprioception', 'action', 'log_policy', 'reward', 'done')

def prepare_store(**kwargs):
    d = dict()
    for k, v in kwargs.items():
        if k in ('vtrace', 'value'):
            d['mean_' + k] = v.detach().cpu().numpy().mean((0, 2))
        else:
            d[k] = v.detach().cpu().numpy()
    return d

class Learner(WorkerBase):
    def __init__(self, ID, DEVICE_INFO, queue, recorder, training_done, pbar_state, 
                 model, state_dict, max_episodes, p_hat, c_hat,
                 discount=0.99, entropy_bonus=True, clip_gradient=1, batch_size=5, lr=1e-4,
                 actor_weight=1, critic_weight=0.5, entropy_weight=1e-2, reduction='mean', 
                 lr_scheduler=False, distributed=False):
        super().__init__(ID, DEVICE_INFO, model)
        # Constants
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.reduction = reduction
        self.batch_size = batch_size
        self.episode = 0
        self.max_episodes = max_episodes
        self.not_alone = distributed

        # Shared resources
        self.queue = queue
        self.training_done = training_done
        self.pbar_state = pbar_state
        self.state_dict = state_dict

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-4)
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                             base_lr=lr, max_lr=lr * 5, step_size_up=10000, step_size_down=10000,
                             mode='triangular2')
        else:
            self.scheduler = None

    def fetch_batch(self):
        while len(self.queue) < self.batch_size: # Get enough samples first
            time.sleep(0.1)
        return self.queue.sample(num=self.batch_size)

    def V_trace(self, values, rewards, target_behavior_ratio, reset_idx):
        with torch.no_grad():
            p = torch.clamp(target_behavior_ratio, max=self.p_hat)
            c = torch.clamp(target_behavior_ratio, max=self.c_hat)
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
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


    def _run(self):
        self.setup()

        # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
        # First batch takes longer to wait. 
        # Do it here for tqdm to estimate the runtime correctly
        batch = self.fetch_batch()
        """
        if self.not_alone:
            time.sleep(5) # Make sure every learner should at least get the init batch
        """
        for k in tqdm(range(0, self.max_episodes, self.batch_size), 
                      desc='L%d' % self.ID, position=self.ID, disable=True):
            self.optimizer.zero_grad()

            run_model_time = time.time()
            vision, propri = batch['vision'], batch['proprioception']
            reset_idx = fetch_reset_idx(batch['done'], vision.shape[0], self.batch_size)
            # Output (T+1, batch, 1). Done state included 
            values, (a, log_target_policy, entropy) = self.model(vision=vision, 
                                                                 propri=propri, 
                                                                 reset_idx=reset_idx, 
                                                                 action=batch['action'])
            run_model_time = time.time() - run_model_time
            run_vtrace_time = time.time()
            # print(log_target_policy)
            rewards = batch['reward'] # (T, batch, 1)

            # The last action correspond to state T + 2 and is not in the calculation
            log_behavior_policy = batch['log_policy']
            ratio = torch.exp(log_target_policy - log_behavior_policy)
            
            vtrace, p, advantage = self.V_trace(values, rewards, ratio, reset_idx)
            run_vtrace_time = time.time() - run_vtrace_time

            loss_time = time.time()
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
            loss_time = time.time() - loss_time
            
            backward_time = time.time()
            total_loss = total_loss.mean() if self.reduction == 'mean' else total_loss.sum()
            total_loss.backward()

            if self.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
            backward_time = time.time() - backward_time
            
            optimizer_time = time.time()
            self.optimizer.step()
            self.model._dummy += 1
            if self.scheduler is not None:
                self.scheduler.step()
                aux = [self.scheduler.get_last_lr()] * self.batch_size
            optimizer_time = time.time() - optimizer_time

            state_dict_time = time.time()
            # CUDA cannot be shared as `load_state_dict()` will raise error when copying between GPUs
            self.state_dict.update(copy.deepcopy(self.model.cpu().state_dict())) 
            self.model = self.model.to(self.device) # .cpu is inplace for nn.module
            state_dict_time = time.time() - state_dict_time
            """
            with self.pbar_state.get_lock():
                self.pbar_state.value += self.batch_size
            """
            fetch_batch_time = time.time()
            batch = self.fetch_batch()
            fetch_batch_time = time.time() - fetch_batch_time
            print(self.ID,
                'run_model_time %.3f' % run_model_time,
                'run_vtrace_time %.3f' % run_vtrace_time,
                'loss_time %.3f' % loss_time,
                'backward_time %.3f' % backward_time,
                'optimizer_time %.3f' % optimizer_time,
                'state_dict_time %.3f' % state_dict_time,
                'fetch_batch_time %.3f' % fetch_batch_time,
                sep='\n')
        
        with self.training_done.get_lock():
            self.training_done.value += 1
        print(self.ID, 'Learner terminated')

    def run(self):
        try:
            self._run()
        except:
            print('Learner %d terminated with error.' % self.ID)
            with self.training_done.get_lock():
                self.training_done.value += 1
            raise

