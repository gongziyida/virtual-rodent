import os, time
import copy
import numpy as np
import torch
from torch.multiprocessing import Process

from virtual_rodent.network.helper import fetch_reset_idx

QUEUE, ACTION_MADE, INPUT_GIVEN = 0, 1, 2

class Agent(Process):
    def __init__(self, DEVICE_ID, action_traffic, exit, model, state_dict, model_update_freq):
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.device = torch.device('cpu' if DEVICE_ID == 'cpu' else 'cuda:%d' % DEVICE_ID)
        self.model_update_freq = model_update_freq
        self.batch_size = len(action_traffic)

        # Variables
        self.model = model.to(self.device) # Need disabling CPU binding
        self.state_dict = state_dict

        # Shared resources
        self.action_traffic = action_traffic
        self.exit = exit

    def fetch_input(self):
        vision, proprioception, done = [], [], []
        for i in range(self.batch_size):
            self.action_traffic[i][INPUT_GIVEN].wait()
            self.action_traffic[i][INPUT_GIVEN].clear()

            inputs = self.action_traffic[i][QUEUE].get()
            vision.append(inputs[0])
            proprioception.append(inputs[1])
            assert len(inputs[2]) == 2 # done should have state -1
            done.append(inputs[2]) 

        vision = torch.stack(vision, dim=0).unsqueeze(0).to(self.device)
        proprioception = torch.stack(proprioception, dim=0).unsqueeze(0).to(self.device)
        done = torch.stack(done, dim=1).to(self.device)
        assert done.shape[1] == self.batch_size
        assert vision.shape[0] == 1 and proprioception.shape[0] == 1
        assert vision.shape[1] == self.batch_size and proprioception.shape[1] == self.batch_size
        return vision, proprioception, done

    def send_action(self, actions, log_policies):
        if len(actions.shape) == 1 and self.batch_size == 1:
            assert len(log_policies.shape) == 1, '%s' % log_policies.shape
            actions, log_policies = actions.unsqueeze(0), log_policies.unsqueeze(0)
        for i in range(self.batch_size): 
            assert actions.shape[0] == 1 and log_policies.shape[0] == 1, '%s, %s' % (actions.shape, log_policies.shape)
            self.action_traffic[i][QUEUE].put((actions[0, i], log_policies[0, i]))
            self.action_traffic[i][ACTION_MADE].set()

    def run(self):
        self.PID = os.getpid()
        print('\n[%s] Setting actor on %s' % (self.PID, self.device))
        
        step = 0
        with torch.no_grad():
            while self.exit.value == 0:
                if step > 0 and step % self.model_update_freq == 0:
                    self.model.load_state_dict(self.state_dict)
                    self.model = self.model.to(self.device)
                step += 1

                vision, proprioception, done = self.fetch_input()

                reset_idx = fetch_reset_idx(done, 1, self.batch_size)
                _, actions, log_policies, _ = self.model((vision, proprioception, reset_idx))

                self.send_action(actions, log_policies)
