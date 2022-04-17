import os, time
import copy
import numpy as np
import torch

from .base import WorkerBase, TrainingTerminated 

QUEUE, ACTION_MADE, INPUT_GIVEN = 0, 1, 2

class Simulator(WorkerBase):
    def __init__(self, ID, DEVICE_INFO, sample_queue, exit, action_traffic, 
                 env_name, max_step):
        super().__init__(ID, DEVICE_INFO, env_name=env_name)
        # Constants
        self.max_step = max_step 

        # Shared resources
        self.sample_queue = sample_queue
        self.action_traffic = action_traffic
        self.exit = exit

    def send_input(self, vision, proprioception, last_done):
        self.action_traffic[QUEUE].put((vision, proprioception, torch.tensor([last_done, False])))
        self.action_traffic[INPUT_GIVEN].set()
        # print(self.ID, 'sent input to the agent.')

    def fetch_action(self):
        # print(self.ID, 'waiting for agent response...')
        self.action_traffic[ACTION_MADE].wait()
        self.action_traffic[ACTION_MADE].clear()
        if self.exit.value == 1:
            raise TrainingTerminated
        try:
            output = self.action_traffic[QUEUE].get()
        except (FileNotFoundError, ConnectionResetError) as e:
            if self.exit.value == 1:
                raise TrainingTerminated
            else:
                raise
        action = output[0].detach().cpu().clone()
        log_policy = output[1].detach().cpu().clone()
        return action, log_policy

    def run(self):
        self.setup()
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import get_vision, get_proprioception
        else:
            from virtual_rodent._test_simulation import get_vision, get_proprioception

        action_spec = self.env.action_spec()
        
        while self.exit.value == 0:
            time_step = self.env.reset()
            last_done = True

            # Note that the model requires knowing if state -1 is done and state 0 is restart
            local_buffer = dict(vision=[], proprioception=[], action=[], 
                                log_policy=[], reward=[], done=[torch.tensor(True)])
            step = 0
            while step < self.max_step:
                # Get state, reward and discount
                
                vision = torch.from_numpy(get_vision(time_step)).to(self.device)
                proprioception = torch.from_numpy(
                                        get_proprioception(time_step, self.propri_attr)
                                    ).to(self.device)

                self.send_input(vision, proprioception, last_done)
                try:
                    action, log_policy = self.fetch_action()
                except TrainingTerminated:
                    break
                """
                vision = torch.ones(1).to(self.device)
                proprioception = torch.from_numpy(time_step.state).to(self.device)
                self.send_input(vision, proprioception, torch.zeros(1))
                action, log_policy = self.fetch_action()

                # Proceed
                time_step = self.env.step(action.numpy())
                """
                time_step = self.env.step(np.clip(action.numpy(), \
                                          action_spec.minimum, action_spec.maximum))
                
                step += 1
                done = time_step.last()
                reward = time_step.reward 

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

            local_buffer['env_name'] = self.env_name

            self.sample_queue.put(local_buffer)

