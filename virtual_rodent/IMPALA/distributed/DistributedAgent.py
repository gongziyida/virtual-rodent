import os, socket, time
import copy
import numpy as np
import torch

import torch.distributed.rpc as rpc

from virtual_rodent.network.helper import fetch_reset_idx
from .base import WorkerBase, TrainingTerminated 
from .ParameterServer import get_behavior_model, forward, store_batch, remote_method

class DistributedAgent(WorkerBase):
    def __init__(self, ID, DEVICE_INFO, exit, env_name, max_step):
        super().__init__(ID, DEVICE_INFO, None, env_name)
        # Constants
        self.max_step = max_step

        # Shared resources
        self.exit, self.exit_value = exit

    def _run(self):
        self.setup()
        model_rref = rpc.remote('ParameterServer', get_behavior_model)

        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import get_vision, get_proprioception
        else:
            from virtual_rodent._test_simulation import get_vision, get_proprioception

        action_spec = self.env.action_spec()

        episode = 0
        with torch.no_grad():
            while self.exit.value != self.exit_value:
                episode += 1

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

                    reset_idx = fetch_reset_idx(torch.tensor([last_done, False]).unsqueeze(1), 1, 1)
                    _, (action, log_policy, _) = remote_method(forward, model_rref,
                                                               vision=vision, 
                                                               propri=proprioception, 
                                                               reset_idx=reset_idx)
                    action, log_policy = action.view(-1), log_policy.view(-1)

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

                rpc.rpc_sync('ParameterServer', store_batch, 
                             args=(self.ID, self.env_name, local_buffer))
        print(self.ID, self.exit.value, self.exit_value, episode)

    def run(self):
        os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
        os.environ['MASTER_PORT'] = '8818'
        print('agent', os.environ['MASTER_ADDR'])
        rpc.init_rpc('DistributedAgent%d' % self.ID, rank=self.ID, world_size=self.WORLD_SIZE)
        print('DistributedAgent running')
        try:
            self._run()
            rpc.shutdown()
        except:
            print('Agent %d terminated with error.' % self.ID)
            raise
        
