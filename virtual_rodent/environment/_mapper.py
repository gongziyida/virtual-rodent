import functools

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.tasks import reach
from dm_control.locomotion.walkers import rodent

"""
appendages_pos (15): head and 4 paw positions, projected to egocentric frame, and reshaped to 1D
joints_pos/vel (30): angle and angular velocity of the 1D hinge joints
tendons_pos/vel (8): extension and extension velocity of the 1D tendons
sensors_accelerometer/velocimeter/gyro (3): Acceleration/velocity/angular velocity of the body
world_zaxis (3): the world's z-vector (-gravity) in this Walker's torso frame
sensors_touch (4): touch sensors (forces) in the 4 paws
"""
RODENT_PROPRIOCEPTION_ATTRIBUTES = ('appendages_pos', 'joints_pos', 'joints_vel', 'tendons_pos', 
                                    'tendons_vel', 'sensors_accelerometer', 'sensors_velocimeter', 
                                    'sensors_gyro', 'sensors_touch', 'world_zaxis')


def __rodent_escape_bowl(random_state=None, physics_dt=0.002, ctrl_dt=0.02):
  """Requires a rodent to climb out of a bowl-shaped terrain."""

  # Build a position-controlled rodent walker.
  walker = rodent.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a bowl-shaped arena.
  arena = bowl.Bowl(
      size=(20., 20.),
      aesthetic='outdoor_natural')

  # Build a task that rewards the agent for being far from the origin.
  task = escape.Escape(
      walker=walker,
      arena=arena,
      physics_timestep=physics_dt,
      control_timestep=ctrl_dt)

  return composer.Environment(time_limit=20,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True), \
         RODENT_PROPRIOCEPTION_ATTRIBUTES


def __rodent_run_gaps(random_state=None, physics_dt=0.002, ctrl_dt=0.02):
  """Requires a rodent to run down a corridor with gaps."""

  # Build a position-controlled rodent walker.
  walker = rodent.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
  # platforms are uniformly randomized.
  arena = corr_arenas.GapsCorridor(
      platform_length=distributions.Uniform(.4, .8),
      gap_length=distributions.Uniform(.05, .2),
      corridor_width=2,
      corridor_length=40,
      aesthetic='outdoor_natural')

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      target_velocity=1.0,
      contact_termination=True,
      terminate_at_height=-0.3,
      physics_timestep=physics_dt,
      control_timestep=ctrl_dt)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True), \
         RODENT_PROPRIOCEPTION_ATTRIBUTES


def __rodent_maze_forage(random_state=None, physics_dt=0.002, ctrl_dt=0.02):
  """Requires a rodent to find all items in a maze."""

  # Build a position-controlled rodent walker.
  walker = rodent.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a maze with rooms and targets.
  wall_textures = labmaze_textures.WallTextures(style='style_01')
  arena = mazes.RandomMazeWithTargets(
      x_cells=11,
      y_cells=11,
      xy_scale=.5,
      z_height=.3,
      max_rooms=4,
      room_min_size=4,
      room_max_size=5,
      spawns_per_room=1,
      targets_per_room=3,
      wall_textures=wall_textures,
      aesthetic='outdoor_natural')

  # Build a task that rewards the agent for obtaining targets.
  task = random_goal_maze.ManyGoalsMaze(
      walker=walker,
      maze_arena=arena,
      target_builder=functools.partial(
          target_sphere.TargetSphere,
          radius=0.05,
          height_above_ground=.125,
          rgb1=(0, 0, 0.4),
          rgb2=(0, 0, 0.7)),
      target_reward_scale=50.,
      contact_termination=False,
      physics_timestep=physics_dt,
      control_timestep=ctrl_dt)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True), \
         RODENT_PROPRIOCEPTION_ATTRIBUTES


def __rodent_two_touch(random_state=None, physics_dt=0.002, ctrl_dt=0.02):
  """Requires a rodent to tap an orb, wait an interval, and tap it again."""

  # Build a position-controlled rodent walker.
  walker = rodent.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build an open floor arena
  arena = floors.Floor(
      size=(10., 10.),
      aesthetic='outdoor_natural')

  # Build a task that rewards the walker for touching/reaching orbs with a
  # specific time interval between touches
  task = reach.TwoTouch(
      walker=walker,
      arena=arena,
      target_builders=[
          functools.partial(target_sphere.TargetSphereTwoTouch, radius=0.02),
      ],
      randomize_spawn_rotation=True,
      target_type_rewards=[25.],
      shuffle_target_builders=False,
      target_area=(1.5, 1.5),
      physics_timestep=physics_dt,
      control_timestep=ctrl_dt,
  )
  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True), \
         RODENT_PROPRIOCEPTION_ATTRIBUTES


def __suite_hopper_stand():
    from dm_control import suite
    return suite.load('hopper', 'stand'), ['position', 'velocity', 'touch']

def __suite_hopper_hop():
    from dm_control import suite
    return suite.load('hopper', 'hop'), ['position', 'velocity', 'touch']

def __suite_cheetah():
    from dm_control import suite
    return suite.load('cheetah', 'run'), ['position', 'velocity']

######## Customized environment constructors ########

######## Mapper ########
MAPPER = {
        'bowl': __rodent_escape_bowl,
        'gaps': __rodent_run_gaps,
        'maze': __rodent_maze_forage,
        'two-touch': __rodent_two_touch,
        '_test_hopper_stand': __suite_hopper_stand,
        '_test_hopper_hop': __suite_hopper_hop,
        '_test_cheetah': __suite_cheetah,
        }

ENV_NAMES = MAPPER.keys()
