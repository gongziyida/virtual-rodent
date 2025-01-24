import time, os
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from virtual_rodent.environment import MAPPER
from virtual_rodent.network.Merel2019 import make_model
from virtual_rodent.visualization import video

eps = np.finfo(np.float32).eps.item()

def get_propri(time_step, propri_attr):
    ret = []
    for pa in propri_attr:
        ret.append(time_step.observation['walker/' + pa])
    return np.concatenate(ret).astype(np.float32)

def get_vision(time_step):
    vis = np.moveaxis(time_step.observation['walker/egocentric_camera'], -1, 0) # Channel as axis 0
    vis = vis.astype(np.float32) / 255
    return vis


def simulate(env, model, propri_attr, max_step, device, reset=True, time_step=None,
             ext_cam=(0,), ext_cam_size=(200, 200)):
    """Simulate until stop criteron is met
    """
    start_time = time.time()

    returns = dict({'cam%d'%i: [] for i in ext_cam})
    returns.update(dict(vision=[], propri=[], action=[], reward=[], log_prob=[]))
    
    if reset:
        time_step = env.reset()
        if hasattr(model, 'reset_rnn'):
            model.reset_rnn()
    else:
        if time_step is None:
            raise ValueError('`time_step` must be given if not reset.')

    action_spec = env.action_spec()

    for step in range(max_step):
        if time_step.last():
            break
        # Get state, reward and discount
        vision = torch.from_numpy(get_vision(time_step)).to(device)
        propri = torch.from_numpy(get_propri(time_step, propri_attr)).to(device)

        _, (action, log_policy, _) = model(vision=vision, propri=propri)

        time_step = env.step(np.clip(action.detach().cpu().squeeze().numpy(), 
                                     action_spec.minimum, action_spec.maximum))

        # Record state t, action t, reward t and done t+1; reward at start is 0
        returns['vision'].append(vision)
        returns['propri'].append(propri)
        returns['action'].append(action)
        returns['reward'].append(torch.tensor(time_step.reward))
        returns['log_prob'].append(log_policy)
        for i in ext_cam:
            cam = env.physics.render(camera_id=i, 
                    height=ext_cam_size[0], width=ext_cam_size[1])
            returns['cam%d'%i].append(cam)

    end_time = time.time()
    returns['time'] = end_time - start_time
    returns['T'] = step 
    return returns

class Worker(mp.Process):
    def __init__(self, id_, env_name, model, opt, max_episode, max_step, discount, update_period, 
                 global_episode, global_reward, res_queue, 
                 vtrace_param=None, ext_cam=(0,), device=torch.device('cpu')):
        super(Worker, self).__init__()
        self.id = id_
        self.episode, self.reward = global_episode, global_reward
        self.res_queue = res_queue
        self.target_model, self.opt = model, opt
        self.env_name = env_name
        self.update_period = update_period
        self.device = device
        self.max_episode, self.max_step = max_episode, max_step
        self.discount = discount
        self.vtrace_param = vtrace_param
        self.ext_cam = ext_cam

    def run(self):
        print(f'[{os.getpid()}] Start Worker{self.id}')
        self.behavior_model = make_model()
        self.env, self.propri_attr = MAPPER[self.env_name]()
        while self.episode.value < self.max_episode:
            i_episode = int(self.episode.value)
            save = i_episode % 50 == 0
            ret = self.simulate(ext_cam=self.ext_cam if save else set())
            self.record(ret['episode_reward'])
            if save:
                self.save(ret, i_episode)

        self.res_queue.put(None) # necessary to signal the termination of this worker

    def simulate(self, ext_cam=(0,), ext_cam_size=(200, 200)):
        start_time = time.time()
    
        buffer = dict(vision=[], propri=[], action=[], value=[], reward=[], log_prob=[], 
                      actor_hc=None, critic_hc=None)
        ret = {'cam%d'%i: [] for i in ext_cam}
        ret['episode_reward'] = 0
        
        time_step = self.env.reset()
        if hasattr(self.behavior_model, 'reset_rnn'):
            self.behavior_model.reset_rnn()
        
        action_spec = self.env.action_spec()
    
        for step in range(self.max_step):
            # Get state, reward and discount
            vision = torch.from_numpy(get_vision(time_step))
            propri = torch.from_numpy(get_propri(time_step, self.propri_attr))
            vision, propri = vision.to(self.device), propri.to(self.device)
    
            value, (action, log_prob, _) = self.behavior_model(vision=vision, 
                                                               propri=propri)
    
            time_step = self.env.step(np.clip(action.detach().cpu().squeeze().numpy(), 
                                              action_spec.minimum, action_spec.maximum))
    
            buffer['vision'].append(vision)
            buffer['propri'].append(propri)
            buffer['action'].append(action.squeeze())
            buffer['value'].append(value.squeeze())
            buffer['log_prob'].append(log_prob.squeeze())
            buffer['reward'].append(torch.tensor(time_step.reward))
            ret['episode_reward'] += buffer['reward'][-1].item()
            for i in ext_cam:
                cam = self.env.physics.render(camera_id=i, height=ext_cam_size[0], 
                                              width=ext_cam_size[1])
                ret['cam%d'%i].append(cam)
    
            if (step + 1) % self.update_period == 0 or time_step.last():
                self.update(buffer)
    
            if time_step.last():
                break
    
        end_time = time.time()
        ret['time'] = end_time - start_time
        ret['T'] = step
        return ret

    def update(self, buffer):
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        if self.vtrace_param is None: # original td error
            # calculate the true value using rewards returned from the environment
            for r in buffer['reward'][::-1]:
                # calculate the discounted value
                R = r.item() + self.discount * R
                returns.insert(0, R)
        
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
        
            for log_prob, value, R in zip(buffer['log_prob'], buffer['value'], returns):
                advantage = R - value.item()
        
                policy_losses.append(-log_prob.squeeze() * advantage)
                value_losses.append(F.mse_loss(value, R) / 2)
        
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
            self.opt.zero_grad()
            loss.backward()
            for bp, tp in zip(self.behavior_model.parameters(), self.target_model.parameters()):
                tp.grad = bp.grad

        else: # V-trace in IMAPLA
            ret = self.target_model(vision=torch.stack(buffer['vision']).unsqueeze(1), 
                                    propri=torch.stack(buffer['propri']).unsqueeze(1),
                                    actor_hc=buffer['actor_hc'], critic_hc=buffer['critic_hc'], 
                                    action=torch.concat(buffer['action']))
            values, (_, log_target, _) = ret
            vtrace, p, advantage = self.V_trace(buffer, values, log_target)

            policy_losses = (-log_target.squeeze() * advantage * p).sum()
            value_losses = F.mse_loss(values, vtrace, reduction='sum') / 2
            loss = policy_losses + value_losses
            self.opt.zero_grad()
            loss.backward()
    
        self.opt.step()
        self.behavior_model.load_state_dict(self.target_model.state_dict())
        buffer.update(dict(vision=[], propri=[], action=[], value=[], reward=[], log_prob=[]))
        buffer['actor_hc'], buffer['critic_hc'] = self.behavior_model.detach_hc()

    def V_trace(self, buffer, values, log_target):
        with torch.no_grad():
            # (T, 1, action dim)
            log_behavior = torch.concat(batch['log_policy'])
            target_behavior_ratio = torch.exp(log_target - log_behavior)
            
            p = torch.clamp(target_behavior_ratio, max=self.vtrace_param['p_max'])
            c = torch.clamp(target_behavior_ratio, max=self.vtrace_param['c_max'])
            
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            dV = (buffer['reward'] + self.discount * next_values - values) * p
            
            vtrace = torch.zeros_like(values)
            vtrace[-1] = values[-1] + dV[-1] # initial condition
            for i in range(vtrace.shape[0]-2, -1, -1): # backward
                correction = self.discount * c[i] * (vtrace[i+1] - values[i+1])
                vtrace[i] = values[i] + dV[i] + correction
                
            advantage = buffer['reward'] - values
            advantage[:-1] += self.discount * vtrace[1:]
            
            return vtrace.detach(), p.detach(), advantage.detach()
            
    def record(self, reward):
        with self.episode.get_lock():
            self.episode.value += 1
        with self.reward.get_lock():
            if self.reward.value == 0.:
                self.reward.value = reward
            else:
                self.reward.value = self.reward.value * 0.99 + reward * 0.01
        self.res_queue.put(self.reward.value)

    def save(self, ret, i_episode):
        for i in self.ext_cam:
            anim = video(ret[f'cam{i}'])
            fname = f'{self.env_name}_w{self.id}_{i_episode}_cam{i}.gif'
            anim.save(os.path.join('./results', fname), writer='pillow')
            plt.clf()
