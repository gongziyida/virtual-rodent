import time
import numpy as np
import torch

from dm_control import composer
from dm_control.locomotion.walkers import rodent

_CTRL_DT = .02
_PHYS_DT = 0.001

"""
appendages_pos (15): head and 4 paw positions, projected to egocentric frame, and reshaped to 1D
joints_pos/vel (30): angle and angular velocity of the 1D hinge joints
tendons_pos/vel (8): extension and extension velocity of the 1D tendons
sensors_accelerometer/velocimeter/gyro (3): Acceleration/velocity/angular velocity of the body
world_zaxis (3): the world's z-vector (-gravity) in this Walker's torso frame
sensors_touch (4): touch sensors (forces) in the 4 paws
"""
PROPRIOCEPTION_ATTRIBUTES = ['appendages_pos', 'joints_pos', 'joints_vel', 'tendons_pos', 'tendons_vel',
                             'sensors_accelerometer', 'sensors_velocimeter', 'sensors_gyro',
                             'sensors_touch', 'world_zaxis']

def make_env(arena, Task, walker=None, time_limit=30, random_state=None, **kwargs):
    """
        Parameters
        ----------
        arena: dm_control.composer.Arena
        Task: function
            constructor that returns dm_control.composer.Task

    """
    if walker is None: # Build a position-controlled rodent walker
        walker = rodent.Rat(observable_options={'egocentric_camera': dict(enabled=True)})
    
    # Build task
    task = Task(walker, arena, physics_timestep=_PHYS_DT, control_timestep=_CTRL_DT, **kwargs)

    # Build environment
    env = composer.Environment(task, time_limit=time_limit, random_state=random_state, 
                               strip_singleton_obs_buffer_dim=True)

    return env


def get_proprioception(time_step):
    ret = []
    for pa in PROPRIOCEPTION_ATTRIBUTES:
        ret.append(time_step.observation['walker/' + pa])
    return np.concatenate(ret).astype(np.float32)

def get_vision(time_step):
    vis = np.moveaxis(time_step.observation['walker/egocentric_camera'], -1, 0) # Channel as axis 0
    vis = vis.astype(np.float32) / 255
    return vis


def simulate(env, model, stop_criteron, device, reset=True, time_step=None,
             ext_cam=False, ext_cam_id=(0,), ext_cam_size=(200, 200)):
    """Simulate until stop criteron is met
    """
    start_time = time.time()

    returns = dict([('cam%d'%i, []) for i in ext_cam_id])
    returns.update(dict(vision=[], proprioception=[], action=[], reward=[], log_policy=[]))
    
    if reset:
        time_step = env.reset()
        if hasattr(model, 'reset_rnn'):
            model.reset_rnn()
    else:
        if time_step is None:
            raise ValueError('`time_step` must be given if not reset.')

    step = 0
    stop = False
    while not stop:
        # Get state, reward and discount
        vision = torch.from_numpy(get_vision(time_step)).to(device)
        proprioception = torch.from_numpy(get_proprioception(time_step)).to(device)
        reward = time_step.reward

        # Return value and distribution pi
        _, pi, _ = model(vision, proprioception, [[step == 0, False]]) 
        action = pi.sample()
        log_policy = pi.log_prob(action)
        time_step = env.step(np.tanh(action.detach().cpu().numpy()))

        # Record state t, action t, reward t and done t+1; reward at start is 0
        returns['vision'].append(vision)
        returns['proprioception'].append(proprioception)
        returns['action'].append(action)
        returns['reward'].append(torch.tensor(0 if reward is None else reward))
        returns['log_policy'].append(log_policy)
        if ext_cam: # Record external camera
            for i in range(len(ext_cam_id)):
                cam = env.physics.render(camera_id=ext_cam_id[i], 
                        height=ext_cam_size[0], width=ext_cam_size[1])
                returns['cam%d'%ext_cam_id[i]].append(cam)

        step += 1
        stop = stop_criteron(time_step, step)

    end_time = time.time()
    returns['time'] = end_time - start_time
    returns['T'] = step 
    return returns


def simulator(env, model, device,
              ext_cam=False, ext_cam_id=(0,), ext_cam_size=(200, 200)):
    """Simulation generator, starts from beginning, and reset upon simulation timeout
    """
    time_step = env.reset()
    assert not time_step.last()
    if hasattr(model, 'reset_rnn'):
        model.reset_rnn()
    returns = dict()
    
    step = 0
    done = True
    while True:
        # Get state, reward and discount
        vision = torch.from_numpy(get_vision(time_step)).to(device)
        proprioception = torch.from_numpy(get_proprioception(time_step)).to(device)
        reward = time_step.reward

        _, pi, _ = model(vision, proprioception, [[done, False]]) 
        action = pi.sample()
        log_behavior_policy = pi.log_prob(action)
        time_step = env.step(np.tanh(action.detach().cpu().numpy()))
        done = time_step.last()

        # Record state t, action t, reward t and done t+1; reward at start is 0
        returns['vision'] = vision
        returns['proprioception'] = proprioception
        returns['action'] = action
        returns['log_policy'] = log_behavior_policy 
        returns['reward'] = torch.tensor(0 if reward is None else reward)
        returns['done'] = done
        if ext_cam: # Record external camera
            for i in range(len(ext_cam_id)):
                cam = env.physics.render(camera_id=ext_cam_id[i], 
                        height=ext_cam_size[0], width=ext_cam_size[1])
                returns['cam%d'%ext_cam_id[i]] = cam

        yield step, returns
        step += 1

        if done: 
            time_step = env.reset()
            assert not time_step.last()

