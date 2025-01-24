import os
os.environ['model_init_method_path'] = os.path.join(os.getcwd(), 'test_network.py')
from itertools import product
import torch
import torch.nn as nn
from virtual_rodent.IMPALA import IMPALA_CPU as IMPALA
from virtual_rodent.utils import load_checkpoint

env, propri_dim, action_dim = '_test_cheetah', 17, 6
def run(a, lr, batch_size, cpu_per_learner):
    
    # model, _ = load_checkpoint(model, './simple_test_IMPALA_out_%s/model.pt' % a)

    impala = IMPALA([env], None, './out_cpu_%s_cheetah_%.0E' % (a, lr), 
                    (1,), (propri_dim,), action_dim)
    
    impala.train(max_step=300, max_episodes=int(5e4), repeat=15, batch_size=batch_size,
            cpu_per_learner=cpu_per_learner, learner_params={'lr': lr})
    
    # Generate videos
    # impala.record()
    

if __name__ == '__main__':
    os.environ['SIMULATOR_IMPALA'] = 'test'
    run('rnn', 1e-4, 3, 2)
