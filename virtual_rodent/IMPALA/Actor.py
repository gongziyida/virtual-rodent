import os, time
import copy
import torch
from torch.multiprocessing import Process

from virtual_rodent.environment import MAPPER


class Actor(Process):
    def __init__(self, DEVICE_ID, queue, exit, model, state_dict, env_name, max_step, 
                 model_update_freq=3):
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.max_step = max_step 
        self.device = torch.device('cpu' if DEVICE_ID == 'cpu' else 'cuda:%d' % DEVICE_ID)
        self.model_update_freq = model_update_freq

        # Variables
        self.model = model.to(self.device) # Need disabling CPU binding
        self.state_dict = state_dict

        # Shared resources
        self.queue = queue
        self.exit = exit
        # Environment name; instantiate when started
        self.env_name = env_name

    def run(self):
        PID = os.getpid()
        print('\n[%s] Setting env "%s" on %s' % (PID, self.env_name, self.device))
        if self.DEVICE_ID == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'
        else: # dm_control/mujoco maps onto EGL_DEVICE_ID
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['EGL_DEVICE_ID'] = str(self.DEVICE_ID) 
        self.env = MAPPER[self.env_name]()
        print('\n[%s] Simulating on env "%s"' % (PID, self.env_name))
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import simulator
        elif str(os.environ['SIMULATOR_IMPALA']) == 'hop_simple':
            from virtual_rodent._test_simulation import simulator
        
        batch_count = 0
        with torch.no_grad():
            while self.exit.value == 0:
                if batch_count > 0 and batch_count % self.model_update_freq == 0:
                    self.model.load_state_dict(self.state_dict)
                    self.model = self.model.to(self.device)
                batch_count += 1

                for i, ret in simulator(self.env, self.model, self.device): # Restart simulation
                    if self.exit.value == 1:
                        break

                    if i == 0: # Init buffer
                        local_buffer = {k: [] for k in ret.keys()}

                    for k in local_buffer.keys(): 
                        local_buffer[k].append(ret[k])

                    if i == self.max_step: # Share
                        for k in local_buffer.keys(): # Stack list of tensor
                            if torch.is_tensor(local_buffer[k][0]):
                                local_buffer[k] = torch.stack(local_buffer[k], dim=0)
                        if self.exit.value == 0:
                            self.queue.put(local_buffer)
                        break
