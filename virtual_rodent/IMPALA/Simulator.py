import os, time
import copy
import numpy as np
import torch
from torch.multiprocessing import Process

from virtual_rodent.environment import MAPPER

QUEUE, ACTION_MADE, INPUT_GIVEN = 0, 1, 2

class Simulator(Process):
    def __init__(self, DEVICE_ID, sample_queue, exit, action_traffic, 
                 env_name, max_step):
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.max_step = max_step 
        self.device = torch.device('cpu' if DEVICE_ID == 'cpu' else 'cuda:%d' % DEVICE_ID)

        # Shared resources
        self.sample_queue = sample_queue
        self.action_traffic = action_traffic
        self.exit = exit
        # Environment name; instantiate when started
        self.env_name = env_name

    def set_env(self):
        self.PID = os.getpid()

        if self.DEVICE_ID == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'
        else: # dm_control/mujoco maps onto EGL_DEVICE_ID
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['EGL_DEVICE_ID'] = str(self.DEVICE_ID) 

        print('\n[%s] Setting env "%s" on %s with %s' % \
                (self.PID, self.env_name, self.device, os.environ['MUJOCO_GL']))
        self.env = MAPPER[self.env_name]()
        print('\n[%s] Simulating on env "%s"' % (self.PID, self.env_name))


    def send_input(self, vision, proprioception, last_done):
        self.action_traffic[QUEUE].put((vision, proprioception, [last_done, False]))
        self.action_traffic[INPUT_GIVEN].set()

    def fetch_action(self):
        self.action_traffic[ACTION_MADE].wait()
        self.action_traffic[ACTION_MADE].clear()
        output = self.action_traffic[QUEUE].get()
        action = output[0].detach().cpu()
        log_policy = output[1].detach().cpu()
        return action, log_policy

    def run(self):
        self.set_env()
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import get_vision, get_proprioception
        elif str(os.environ['SIMULATOR_IMPALA']) == 'hop_simple':
            from virtual_rodent._test_simulation import get_vision, get_proprioception
        
        while self.exit.value == 0:
            time_step = self.env.reset()
            last_done = True

            # Note that the model requires knowing if state -1 is done and state 0 is restart
            local_buffer = dict(vision=[], proprioception=[], action=[], 
                                log_policy=[], reward=[], done=[True])
            step = 0
            while step <= self.max_step:
                # Get state, reward and discount
                vision = torch.from_numpy(get_vision(time_step)).to(self.device)
                proprioception = torch.from_numpy(get_proprioception(time_step)).to(self.device)
                reward = time_step.reward

                self.send_input(vision, proprioception, last_done)
                action, log_policy = self.fetch_action()

                # Proceed
                time_step = self.env.step(np.tanh(action.numpy()))
                step += 1
                last_done = time_step.last()

                # Record state t, action t, reward t and done t+1; reward at start is 0
                local_buffer['vision'].append(vision)
                local_buffer['proprioception'].append(proprioception)
                local_buffer['action'].append(action)
                local_buffer['log_policy'].append(log_policy)
                local_buffer['reward'].append(torch.tensor(0 if reward is None else reward))
                local_buffer['done'].append(last_done)

                # Reset
                if time_step.last(): 
                    time_step = self.env.reset()
                    assert not time_step.last()

            for k in local_buffer.keys(): # Stack list of tensor
                if k != 'done':
                    local_buffer[k] = torch.stack(local_buffer[k], dim=0)
            if self.exit.value == 0:
                self.sample_queue.put(local_buffer)

