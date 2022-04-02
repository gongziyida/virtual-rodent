from dm_control import composer
from dm_control.locomotion.walkers import rodent

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


