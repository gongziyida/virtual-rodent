import os, socket, time, random
import copy
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd

from virtual_rodent.network.helper import fetch_reset_idx
from .base import WorkerBase
from .ParameterServer import get_target_model, param_rref, forward, fetch_batch, remote_method

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
    def __init__(self, ID, DEVICE_INFO, training_done, ready, pbar_state, 
                 max_episodes, p_hat, c_hat,
                 discount=0.99, entropy_bonus=True, clip_gradient=1, batch_size=5, lr=1e-4,
                 actor_weight=1, critic_weight=0.5, entropy_weight=1e-2, reduction='mean', 
                 lr_scheduler=False, distributed=False):
        super().__init__(ID, DEVICE_INFO)
        # Constants
        self.discount, self.p_hat, self.c_hat = discount, p_hat, c_hat
        self.entropy_bonus = entropy_bonus
        self.clip_gradient = clip_gradient
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.reduction = reduction
        self.lr = lr
        self.batch_size = batch_size
        self.episode = 0
        self.max_episodes = max_episodes
        self.not_alone = distributed

        # Shared resources
        self.training_done = training_done
        self.ready = ready
        self.pbar_state = pbar_state
    
    def _wait(self, goal):
        with self.ready[0].get_lock():
            if goal == 'step':
                self.ready[0].value += 1
                last = self.ready[0].value == self.GROUP_SIZE
                i = 1
            elif goal == 'forward':
                self.ready[0].value -= 1
                last = self.ready[0].value == 0
                i = 2
        if last: # Last one awakes the other processes
            self.ready[i].set()
            # print('Learner %d signals for %s' % (self.ID, goal))
            time.sleep(0.1) # In case another process is about wait
            self.ready[i].clear()
        else:
            # print('Learner %d waits for %s' % (self.ID, goal))
            self.ready[i].wait()
        
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


    def _run(self):
        self.setup()
        model_rref = rpc.remote('ParameterServer', get_target_model)
        model_param_rref = remote_method(param_rref, model_rref)
        optimizer = DistributedOptimizer(torch.optim.RMSprop, model_param_rref, 
                                         lr=self.lr, eps=1e-4)

        # (T+1 or T, batch, ...) for tensors, (batch)(T+1 or T)(...) for lists
        # First batch takes longer to wait. 
        # Do it here for tqdm to estimate the runtime correctly
        batch = rpc.rpc_sync('ParameterServer', fetch_batch, args=(self.batch_size))

        for k in tqdm(range(0, self.max_episodes, self.batch_size), desc='L%d' % self.ID, 
                      position=self.ID - (self.WORLD_SIZE - self.GROUP_SIZE), disable=True):
            # optimizer.zero_grad()

            with dist_autograd.context() as context_id:
                run_model_time = time.time()
                vision, propri = batch['vision'], batch['proprioception']
                reset_idx = fetch_reset_idx(batch['done'], vision.shape[0], self.batch_size)

                # Output (T+1, batch, 1). Done state included 
                values, (a, log_target_policy, entropy) = remote_method(forward, model_rref,
                                                                        vision=vision, 
                                                                        propri=propri, 
                                                                        reset_idx=reset_idx, 
                                                                        action=batch['action'])
                run_model_time = time.time() - run_model_time
                run_vtrace_time = time.time()
                # print(log_target_policy)
                rewards = batch['reward'] # (T, batch, 1)

                # The last action correspond to state T + 2 and is not in the calculation
                log_target_policy = log_target_policy
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
                
                record_time = time.time()
                rpc.rpc_sync('ParameterServer', record_loss, 
                             args=(prepare_store(total_loss=total_loss, actor_loss=actor_loss,
                                                critic_loss=critic_loss, entropy=entropy,
                                                vtrace=vtrace, value=value),))
                record_time = time.time() - record_time

                backward_time = time.time()
                total_loss = total_loss.mean() if self.reduction == 'mean' else total_loss.sum()
                # total_loss.backward()
                dist_autograd.backward(context_id, [total_loss])

                #if self.clip_gradient is not None:
                    #nn.utils.clip_grad_norm_(param_rref, self.clip_gradient)
                backward_time = time.time() - backward_time
                
                # print(self.ready[0].value, '/', self.GROUP_SIZE)
                optimizer_time = time.time()
                self._wait('step')
                # print('Learner %d steps' % self.ID)

                optimizer.step(context_id)
                self._wait('forward')
                # print('Learner %d forwards' % self.ID)
                optimizer_time = time.time() - optimizer_time

                
                with self.pbar_state.get_lock():
                    self.pbar_state.value += self.batch_size
                
                fetch_batch_time = time.time()
                batch = rpc.rpc_sync('ParameterServer', fetch_batch, args=(self.batch_size))
                fetch_batch_time = time.time() - fetch_batch_time
                """
                print(self.ID,
                    'run_model_time %.3f' % run_model_time,
                    'run_vtrace_time %.3f' % run_vtrace_time,
                    'loss_time %.3f' % loss_time,
                    'record_time %.3f' % record_time,
                    'backward_time %.3f' % backward_time,
                    'optimizer_time %.3f' % optimizer_time,
                    'fetch_batch_time %.3f' % fetch_batch_time,
                    sep='\n')
                """
        
        with self.training_done.get_lock():
            self.training_done.value += 1
        print(self.ID, 'Learner terminated')

    def run(self):
        os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
        os.environ['MASTER_PORT'] = '8818'
        rpc.init_rpc('Learner%d' % self.ID, rank=self.ID, world_size=self.WORLD_SIZE)
        print('Learner running')
        try:
            self._run()
            rpc.shutdown()
        except:
            print('Learner %d terminated with error.' % self.ID)
            with self.training_done.get_lock():
                self.training_done.value += 1
            raise

