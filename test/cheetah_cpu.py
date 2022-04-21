import os
from itertools import product
import torch
import torch.nn as nn
from virtual_rodent.network.propri_enc import MLPEnc
from virtual_rodent.IMPALA import IMPALA_CPU as IMPALA
from virtual_rodent.utils import load_checkpoint

env, propri_dim, action_dim = '_test_cheetah', 17, 6

def run(a, lr, batch_size, cpu_per_learner):
    
    if a == 'mlp':
        import virtual_rodent.network._test_model_mlp as TestModel
    elif a == 'rnn':
        import virtual_rodent.network._test_model_rnn as TestModel
    
    emb_dim = 100 # propri_dim
    propri_enc = MLPEnc(propri_dim, emb_dim, hidden_dims=(300,))
    actor = TestModel.Actor(emb_dim, action_dim)
    critic = TestModel.Critic(emb_dim)
    model = TestModel.TestModel(propri_enc, [propri_dim], actor, critic, action_dim) 
    # model, _ = load_checkpoint(model, './simple_test_IMPALA_out_%s/model.pt' % a)

    impala = IMPALA([env], model, './out_cpu_%s_cheetah_%.0E' % (a, lr), 
                    (1,), (propri_dim,), action_dim)
    
    impala.train(max_step=300, max_episodes=int(1e4), repeat=2, batch_size=batch_size,
            cpu_per_learner=cpu_per_learner, learner_params={'lr': lr})
    
    # Generate videos
    # impala.record()
    

if __name__ == '__main__':
    os.environ['SIMULATOR_IMPALA'] = 'test'
    run('rnn', 1e-4, 3, 2)
