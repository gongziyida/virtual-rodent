import time

from dm_control import composer
from dm_control.locomotion.walkers import rodent

# Visualization
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

_CTRL_DT = .02
_PHYS_DT = 0.001

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

def simulate(env, policy, train=False, get_frame=False, cam_id=(0,), cam_size=(200, 200)):
    start_time = time.time()
    res = dict([('cam%d'%i, []) for i in cam_id])
    res.update({'reward': [], 'discount': [], 'observation': []})

    # Step through the environment for one episode with random actions.
    time_step = env.reset()

    counter = 0
    while not time_step.last():
        counter += 1

        # TODO: what function to call to generate action
        action = policy.next()
        time_step = env.step(action)
        returns['reward'].append(time_step.reward)
        returns['discount'].append(time_step.discount)
        returns['observation'].append(time_step.observation)
        if get_frame:
            for i in range(len(canera_id)):
                cam = env.physics.render(camera_id=cam_id[i], height=cam_size[0], width=cam_size[1])
                returns['cam%d'%i].append(cam)
        if train: # TODO: implement training
            pass

    end_time = time.time()
    res['runtime'] = end_time - start_time
    res['loops'] = counter
    return res

def display_video(frames, framerate=30, dpi=70):
    """ For IPython do the following on the return `anim`:
        ```
            from IPython.display import HTML
            HTML(anim.to_html5_video())
        ```
    """
    height, width, _ = frames[0].shape
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return anim
