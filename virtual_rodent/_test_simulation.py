import time
import numpy as np
import torch

PROPRIOCEPTION_ATTRIBUTES = ['position', 'velocity', 'touch']

def get_proprioception(time_step):
    ret = []
    for pa in PROPRIOCEPTION_ATTRIBUTES:
        ret.append(time_step.observation[pa])
    return np.concatenate(ret).astype(np.float32)

def get_vision(time_step):
    return np.zeros(1).astype(np.float32)


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
        vision = torch.tensor(0).to(device)
        proprioception = torch.from_numpy(get_proprioception(time_step)).to(device)
        reward = time_step.reward

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
        vision = torch.tensor(0).to(device)
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

