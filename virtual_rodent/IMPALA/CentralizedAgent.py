import os, time, tqdm
import copy
import numpy as np
import torch

from virtual_rodent.network.helper import fetch_reset_idx
from .base import WorkerBase, TrainingTerminated 

QUEUE, ACTION_MADE, INPUT_GIVEN = 0, 1, 2

class CentralizedAgent(WorkerBase):
    def __init__(self, DEVICE_INFO, action_traffic, exit, model, state_dict, 
                 batch_size, max_step, model_update_freq):
        super().__init__(0, DEVICE_INFO, model)
        # Constants
        self.batch_size = batch_size
        self.max_step = max_step
        self.model_update_freq = model_update_freq
        self.n_simulators = len(action_traffic)

        # Variables
        self.state_dict = state_dict

        # Shared resources
        self.action_traffic = action_traffic
        self.exit = exit

    def fetch_input(self, n):
        vision, proprioception, done = [], [], []
        for i in range(n): # Note the order of envs in base.py
            # print('Agent waiting for process ', i, '...')
            self.action_traffic[i][INPUT_GIVEN].wait()
            self.action_traffic[i][INPUT_GIVEN].clear()
            if self.exit.value == 1:
                raise TrainingTerminated

            inputs = self.action_traffic[i][QUEUE].get()
            vision.append(inputs[0])
            proprioception.append(inputs[1])
            assert len(inputs[2]) == 2 # done should have state -1
            done.append(inputs[2]) 

        vision = torch.stack(vision, dim=0).unsqueeze(0).to(self.device)
        proprioception = torch.stack(proprioception, dim=0).unsqueeze(0).to(self.device)
        done = torch.stack(done, dim=1).to(self.device)
        assert done.shape[1] == n
        assert vision.shape[0] == 1 and proprioception.shape[0] == 1
        assert vision.shape[1] == n and proprioception.shape[1] == n
        return vision, proprioception, done

    def send_action(self, actions, log_policies, n):
        if len(actions.shape) == 1 and self.n_simulators == 1:
            assert len(log_policies.shape) == 1, '%s' % log_policies.shape
            actions, log_policies = actions.unsqueeze(0), log_policies.unsqueeze(0)
        for i in range(n): 
            assert actions.shape[0] == 1 and log_policies.shape[0] == 1, '%s, %s' % (actions.shape, log_policies.shape)
            self.action_traffic[i][QUEUE].put((actions[0, i].detach().cpu(), log_policies[0, i].detach().cpu()))
            self.action_traffic[i][ACTION_MADE].set()


    def run(self):
        self.setup()

        with torch.no_grad():
            print('Preparing init batch...')

            # For the rest of training
            step, episode = 0, 0
            while self.exit.value == 0:
                if step > 0 and step % self.max_step == 0:
                    step, episode = 0, episode + 1

                if episode % self.model_update_freq == 0:
                    self.model.load_state_dict(self.state_dict)
                    self.model = self.model.to(self.device)

                try:
                    vision, proprioception, done = self.fetch_input(self.n_simulators)
                except TrainingTerminated:
                    break

                reset_idx = fetch_reset_idx(done, 1, self.n_simulators)
                _, (actions, log_policies, _) = self.model(vision=vision, propri=proprioception, 
                                                           reset_idx=reset_idx)

                self.send_action(actions, log_policies, self.n_simulators)
                
                step += 1
