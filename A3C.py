import os
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


save_dir = './results/'
mp.set_start_method('spawn', force=True)
    
def main(env_name, max_episode, max_step, update_period):
    gamma = 0.99
    target_model = make_model()
    target_model.share_memory()
    opt = SharedAdam(target_model.parameters(), lr=1e-4)  # global optimizer
    
    ext_cam = (0,)
    ext_cam_size = (200, 200)
    global_episode, global_reward = mp.Value('i', 0), mp.Value('d', 0.)
    res_queue = mp.Queue()
    workers = [Worker(i, env_name, target_model, opt, max_episode, max_step, gamma, 
                      update_period, global_episode, global_reward, res_queue) 
               for i in range(2)]
               # for i in range(mp.cpu_count()-2)]
    
    for w in workers:
        w.start() 

    res = []
    end = False
    with tqdm(total=max_episode) as pbar:
        while not end:
            pbar.update(global_episode.value - pbar.n)
            r = res_queue.get()
            end = r is None
            if r is not None:
                res.append(r)
            if pbar.n % 20 == 0 or end:
                np.save(os.path.join('./results', 'reward.npy'), np.array(res))
            if (pbar.n % 100 == 0 or end) and pbar.n > 1:
                torch.save(target_model.state_dict(), 
                           os.path.join('./results', f'weights{pbar.n}.pth'))
        
    for w in workers:
        w.join() 

if __name__ == "__main__":
    main('gaps', max_episode=4000, max_step=210, update_period=30)