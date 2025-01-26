import sys, os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.multiprocessing as mp
from sharedAdam import SharedAdam

from virtual_rodent.environment import MAPPER
from virtual_rodent.visualization import video
from virtual_rodent.simulation import *
from virtual_rodent.network.Merel2019 import make_model
from virtual_rodent.utils import load_checkpoint


mp.set_start_method('spawn', force=True) # Required

def update_target(buffer, model, opt, discount, p_max, c_max, entropy_weight):
    ''' Update the target policy
        buffer: dict
            Must contains `'vision'`, `'propri'`, `'reward'`, `'action'`, 
            `'actor_hc'`, `'critic_hc'`. Except for the latter two hidden layer states,
            all must be torch tensors. The first two dimensions of `'vision'`, `'propri'`, 
            and `'action'` are T and 1.
    '''
    # for compatability with neural nets
    ret = model(vision=buffer['vision'], propri=buffer['propri'],
                actor_hc=buffer['actor_hc'], critic_hc=buffer['critic_hc'], 
                action=buffer['action'])
    # squeeze for vtrace calculation
    values, log_target, entropy = ret[0].squeeze(), ret[1][1].squeeze(), ret[1][2].squeeze()
    vtrace, p, advantage = V_trace(buffer, values, log_target, discount, p_max, c_max)

    policy_losses = (-log_target * advantage * p).sum()
    value_losses = F.mse_loss(values, vtrace, reduction='sum') / 2
    loss = policy_losses + value_losses - entropy_weight * entropy.sum()

    opt.zero_grad()
    loss.backward()
    opt.step()
    model.detach_hc()

def V_trace(buffer, values, log_target, discount, p_max, c_max):
    ''' Calculate V trace
        values, log_target: torch.tensor
            Should have shape (T,)
    '''
    with torch.no_grad():
        target_behavior_ratio = torch.exp(log_target - buffer['log_prob'])
        
        p = torch.clamp(target_behavior_ratio, max=p_max)
        c = torch.clamp(target_behavior_ratio, max=c_max)
        
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        
        dV = (buffer['reward'] + discount * next_values - values) * p
        
        vtrace = torch.zeros_like(values)
        vtrace[-1] = values[-1] + dV[-1] # initial condition
        for i in range(vtrace.shape[0]-2, -1, -1): # backward
            correction = discount * c[i] * (vtrace[i+1] - values[i+1])
            vtrace[i] = values[i] + dV[i] + correction
            
        advantage = buffer['reward'] - values
        advantage[:-1] += discount * vtrace[1:]
        
    return vtrace.detach(), p.detach(), advantage.detach()


def main(env_name, max_episode, max_step, update_period, n_workers, save_dir,
         discount=0.99, p_max=10, c_max=2, entropy_weight=0.01, 
         model_state_dict_path=None):
    target_model = make_model()
    if model_state_dict_path is not None:
        target_model.load_state_dict(torch.load(model_state_dict_path, weights_only=True))
    target_model.share_memory()
    opt = SharedAdam(target_model.parameters(), lr=1e-4)  # global optimizer
    
    ext_cam = (0,)
    ext_cam_size = (200, 200)
    global_episode = mp.Value('i', 0)
    res_queue, buffer_queue = mp.Queue(), mp.Queue()

    workers = [Worker(i, env_name, target_model, opt, max_episode, max_step, 
                      update_period, discount, entropy_weight, 
                      global_episode, res_queue, buffer_queue)
               for i in range(n_workers)]
               # for i in range(mp.cpu_count()-2)]

    for w in workers:
        w.start() 

    res = []
    n_workers_done = 0
    
    with tqdm(total=max_episode) as pbar:
        while n_workers_done < n_workers:
            pbar.update(global_episode.value - pbar.n)
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                n_workers_done += 1

            end_time = time.time()

            # update periodically
            if (pbar.n % 5 == 0 and pbar.n > 1) or pbar.n >= max_episode - 1:
                while not buffer_queue.empty():
                    buffer = buffer_queue.get()
                    update_target(buffer, target_model, opt, discount, 
                                  p_max, c_max, entropy_weight)

            end = n_workers_done == n_workers # for convenience
            # save
            if pbar.n % 20 == 0 or end:
                np.save(os.path.join(save_dir, 'reward.npy'), np.array(res))
            if (pbar.n % 100 == 0 or end) and pbar.n > 1:
                torch.save(target_model.state_dict(), 
                           os.path.join(save_dir, f'weights{pbar.n}.pth'))
        
    for w in workers:
        w.join()

def read_hyperparam(param_path):
    with open(param_path, 'r') as f:
        txt = f.read().split('\n')
    kwargs = dict()
    for t in txt:
        k, v = t.split('=')
        if v == 'None':
            v = None
        elif v.isdigit():
            v = int(v)
        if k == 'n_workers' and v <= 0:
            v = mp.cpu_count()-1
        kwargs[k] = v
    return kwargs

if __name__ == "__main__":
    main('gaps', save_dir='./results/', **read_hyperparam('./hyperparam'))