import time
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


def get_proprioception(observation):
    ret = []
    for pa in PROPRIOCEPTION_ATTRIBUTES:
        ret.append(observation['walker/' + pa])
    return np.concatenate(ret)


def simulate(env, model, stop_criteron, device, reset=True, time_step=None,
             ext_cam=False, ext_cam_id=(0,), ext_cam_size=(200, 200)):
    """Simulate until stop criteron is met
    """
    start_time = time.time()

    returns = dict([('cam%d'%i, []) for i in ext_cam_id])
    returns.update(dict(vision=[], proprioception=[], action=[], reward=[], discount=[]))
    
    if reset:
        time_step = env.reset()
    else:
        if time_step is None:
            raise ValueError('`time_step` must be given if not reset.')

    step = 0
    stop = False
    while not stop:
        # Get state, reward and discount
        vision = torch.from_numpy(time_step.observation['walker/egocentric_camera']).to(device)
        proprioception = torch.from_numpy(get_proprioception(time.observation)).to(device)
        reward = time_step.reward
        discount = time_step.discount
        
        value, action = self.model(vision, proprioception) # Act

        returns['vision'].append(vision)
        returns['proprioception'].append(proprioception)
        returns['action'].append(action)
        returns['reward'].append(reward)
        returns['discount'].append(discount)

        time_step = env.step(action.cpu().detach().numpy())
        
        if ext_cam: # Record external camera
            for i in range(len(canera_id)):
                cam = env.physics.render(camera_id=ext_cam_id[i], 
                        height=ext_cam_size[0], width=ext_cam_size[1])
                returns['cam%d'%i].append(cam)

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
    returns = dict()
    
    step = 0
    while True:
        # Get state, reward and discount
        vision = torch.from_numpy(time_step.observation['walker/egocentric_camera']).to(device)
        proprioception = torch.from_numpy(get_proprioception(time.observation)).to(device)
        reward = time_step.reward
        discount = time_step.discount
        
        returns['vision'] = vision
        returns['proprioception'] = proprioception
        returns['reward'] = reward
        returns['discount'] = discount
        returns['done'] = time_step.last()
        if ext_cam: # Record external camera
            for i in range(len(canera_id)):
                cam = env.physics.render(camera_id=ext_cam_id[i], 
                        height=ext_cam_size[0], width=ext_cam_size[1])
                returns['cam%d'%i] = cam

        if time_step.last():
            returns['action'] = torch.zeros_like(action)
            time_step = env.reset()
            assert not time_step.last()
        else:
            value, action = self.model(vision, proprioception) # Act
            returns['action'] = action
            time_step = env.step(action.cpu().detach().numpy())
        
        yield step, returns
        step += 1
