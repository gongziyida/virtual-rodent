__DM_CONTROL_PREFIX = 'dm_control.locomotion.examples.basic_rodent_2020.'
__DIY_PREFIX = 'virtual_rodent.environment.'

######## Built-in environment constructors ########
def __builtin_bowl():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_escape_bowl
    return rodent_escape_bowl()

def __builtin_gaps():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_run_gaps
    return rodent_run_gaps()

def __builtin_maze():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_maze_forage
    return rodent_maze_forage()

def __builtin_2touch():
    from dm_control.locomotion.examples.basic_rodent_2020 import rodent_two_touch
    return rodent_two_touch()

######## Customized environment constructors ########

######## Mapper ########
MAPPER = {
        'built-in bowl': __builtin_bowl,
        'built-in gaps': __builtin_gaps,
        'built-in maze': __builtin_maze,
        'built-in two-touch': __builtin_2touch,
        }

