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
        print('[%s] Setting env "%s" on EGL device %d' % (PID, self.env_name, self.EGL_ID))
        os.environ['EGL_DEVICE_ID'] = self.EGL_ID # dm_control/mujoco maps onto device EGL_DEVICE_ID
        self.env = importlib.import_module(MAPPER[self.env_name])
        print('[%s] Simulating on env "%s"' % (PID, self.env_name))
        with torch.no_grad():
            while self.exit.value == 0:
                local_buffer = []
                model = self.model.to(device)

                for i, ret in simulator(env, model, device):
                    local_buffer.append(ret)
                    if i == self.max_step or self.exit.value == 0:
                        self.queue.put(local_buffer)
                        break
