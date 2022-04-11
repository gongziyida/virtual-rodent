__DM_CONTROL_PREFIX = 'dm_control.locomotion.examples.basic_rodent_2020.'
__DIY_PREFIX = 'virtual_rodent.environment.'

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

######## Built-in environment constructors ########
def __builtin_bowl():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_escape_bowl
    return rodent_escape_bowl(), RODENT_PROPRIOCEPTION_ATTRIBUTES

def __builtin_gaps():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_run_gaps
    return rodent_run_gaps(), RODENT_PROPRIOCEPTION_ATTRIBUTES


def __builtin_maze():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_maze_forage
    return rodent_maze_forage(), RODENT_PROPRIOCEPTION_ATTRIBUTES


def __builtin_2touch():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_two_touch
    return rodent_two_touch(), RODENT_PROPRIOCEPTION_ATTRIBUTES


def __suite_hopper_stand():
    from dm_control import suite
    return suite.load('hopper', 'stand'), ['position', 'velocity', 'touch']

def __suite_hopper_hop():
    from dm_control import suite
    return suite.load('hopper', 'hop'), ['position', 'velocity', 'touch']

def __suite_cheetah():
    from dm_control import suite
    return suite.load('cheetah', 'run'), ['position', 'velocity']

def __simple_test():
    from ._test_env import TestEnv
    return TestEnv(), None

######## Customized environment constructors ########

######## Mapper ########
MAPPER = {
        'built-in bowl': __builtin_bowl,
        'built-in gaps': __builtin_gaps,
        'built-in maze': __builtin_maze,
        'built-in two-touch': __builtin_2touch,
        '_test_hopper_stand': __suite_hopper_stand,
        '_test_hopper_hop': __suite_hopper_hop,
        '_test_cheetah': __suite_cheetah,
        '_simple_test': __simple_test,
        }

ENV_NAMES = MAPPER.keys()
