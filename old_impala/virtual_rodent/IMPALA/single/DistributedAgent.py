import os, time
import copy
import numpy as np
import torch

from virtual_rodent.network.helper import fetch_reset_idx
from .base import WorkerBase, TrainingTerminated 

class DistributedAgent(WorkerBase):
    def __init__(self, ID, DEVICE_INFO, sample_queue, recorder, exit,
                 model, state_dict, env_name, max_step, model_update_freq):
        super().__init__(ID, DEVICE_INFO, model, env_name)
        # Constants
        self.max_step = max_step
        self.model_update_freq = model_update_freq

        # Shared resources
        self.state_dict = state_dict
        self.sample_queue = sample_queue
        self.recorder = recorder
        self.exit, self.exit_value = exit

    def run(self):
        self.setup()
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import get_vision, get_proprioception
        else:
            from virtual_rodent._test_simulation import get_vision, get_proprioception

        action_spec = self.env.action_spec()

        episode = 0
        with torch.no_grad():
            while self.exit.value != self.exit_value:
                if episode % self.model_update_freq == 0:
                    self.model.load_state_dict(self.state_dict)
                    self.model = self.model.to(self.device)
                episode += 1

                time_step = self.env.reset()
                last_done = True

                # Note that the model requires knowing if state -1 is done and state 0 is restart
                local_buffer = dict(vision=[], proprioception=[], action=[], 
                                    log_policy=[], reward=[], done=[torch.tensor(True)])
                step = 0
                sum_reward = 0
                while step < self.max_step:
                    # Get state, reward and discount
                    
                    vision = torch.from_numpy(get_vision(time_step)).to(self.device)
                    proprioception = torch.from_numpy(
                                            get_proprioception(time_step, self.propri_attr)
                                        ).to(self.device)

                    reset_idx = fetch_reset_idx(torch.tensor([last_done, False]).unsqueeze(1), 1, 1)
                    _, (action, log_policy, _) = self.model(vision=vision, propri=proprioception, 
                                                            reset_idx=reset_idx)
                    action, log_policy = action.view(-1), log_policy.view(-1)

                    time_step = self.env.step(np.clip(action.numpy(), \
                                              action_spec.minimum, action_spec.maximum))
                    
                    step += 1
                    done = time_step.last()
                    reward = time_step.reward 
                    sum_reward += reward

                    # Record state t, action t, reward t and done t+1
                    local_buffer['vision'].append(vision)
                    local_buffer['proprioception'].append(proprioception)
                    local_buffer['action'].append(action)
                    local_buffer['log_policy'].append(log_policy)
                    local_buffer['reward'].append(torch.tensor(reward))
                    local_buffer['done'].append(torch.tensor(done))

                    # Reset
                    if done: 
                        time_step = self.env.reset()
                        assert not time_step.last()

                    last_done = done


                for k in local_buffer.keys(): # Stack list of tensor
                    local_buffer[k] = torch.stack(local_buffer[k], dim=0)

                self.sample_queue.put(self.ID, local_buffer)
                self.recorder.put((self.env_name, sum_reward / self.max_step))
        print(self.ID, self.exit.value, self.exit_value, episode)
