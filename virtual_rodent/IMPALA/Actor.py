import os
import torch
from torch.multiprocessing import Process

from virtual_rodent.environment import MAPPER


class Actor(Process):
    def __init__(self, EGL_ID, queue, exit, model, env_name, max_step):
        super().__init__()
        # Constants
        self.EGL_ID = EGL_ID
        self.max_step = max_step 
        self.device = torch.device('cuda:%d' % EGL_ID) 
        # Shared resources
        self.queue = queue
        self.exit = exit
        self.model = model
        # Environment name; instantiate when started
        self.env_name = env_name

    def run(self):
        PID = os.getpid()
        print('\n[%s] Setting env "%s" on EGL device %d' % (PID, self.env_name, self.EGL_ID))
        os.environ['EGL_DEVICE_ID'] = str(self.EGL_ID) # dm_control/mujoco maps onto EGL_DEVICE_ID
        self.env = MAPPER[self.env_name]()
        print('\n[%s] Simulating on env "%s"' % (PID, self.env_name))
        from virtual_rodent.simulation import simulator

        with torch.no_grad():
            while self.exit.value == 0:
                model = self.model.to(self.device) # Need disabling CPU binding

                for i, ret in simulator(self.env, model, self.device): # Restart simulation
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
                        self.queue.put(local_buffer)
                        break
