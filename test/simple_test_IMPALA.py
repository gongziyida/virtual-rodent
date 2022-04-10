import os
from itertools import product
import torch
import torch.nn as nn
from virtual_rodent.network.propri_enc import MLPEnc
from virtual_rodent.IMPALA import IMPALA
from virtual_rodent.utils import load_checkpoint

# env, propri_dim, action_dim = '_test_hopper', 15, 4
env, propri_dim, action_dim = '_test_cheetah', 17, 6
# env, propri_dim, action_dim = '_simple_test', 2, 2

def run(a, lr, lr_scheduler):
    
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
    """
    import virtual_rodent.network._simple as TestModel
    actor = TestModel.Actor()
    critic = TestModel.Critic()
    model = TestModel.TestModel(actor, critic) 
    """
    impala = IMPALA([env], model, './out_%s_%.0E_%d' % (a, lr, int(lr_scheduler)))
    
    # Note repeat should be smaller than the number of CPU cores available - 3
    impala.train(max_step=300, max_episodes=int(1e6), repeat=45, batch_size=20,
                 learner_params={'lr': lr , 'lr_scheduler': lr_scheduler})
    
    # Generate videos
    impala.record()
    

if __name__ == '__main__':
    os.environ['SIMULATOR_IMPALA'] = 'test'
    for lr, lr_schedule in product((1e-4, 3e-4), (False, True)):
        run('mlp', lr, lr_schedule)
