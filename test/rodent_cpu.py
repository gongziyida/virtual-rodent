import os
from itertools import product
import torch
import torch.nn as nn
from virtual_rodent import VISION_DIM, PROPRI_DIM, ACTION_DIM
from virtual_rodent.network.vision_enc import ResNet18Enc
from virtual_rodent.network.propri_enc import MLPEnc
import virtual_rodent.network.Merel2019 as Merel2019
from virtual_rodent.IMPALA import IMPALA_CPU as IMPALA
from virtual_rodent.utils import load_checkpoint

def run(env, lr, batch_size, cpu_per_learner):
    vision_enc = ResNet18Enc()
    vision_emb_dim = vision_enc.get_emb_dim(VISION_DIM)

    propri_emb_dim = 50 # propri_dim
    propri_enc = MLPEnc(PROPRI_DIM[0], propri_emb_dim, hidden_dims=(100,))

    critic_in_dim = vision_emb_dim + propri_emb_dim
    critic = Merel2019.Critic(critic_in_dim)

    actor_in_dim = critic_in_dim + PROPRI_DIM[0] + critic.hidden_dim
    actor = Merel2019.Actor(actor_in_dim, ACTION_DIM)

    model = Merel2019.MerelModel(vision_enc, propri_enc, VISION_DIM, PROPRI_DIM, 
                                 actor, critic, ACTION_DIM) 
    # model, _ = load_checkpoint(model, './simple_test_IMPALA_out_%s/model.pt' % a)

    folder_name = '_'.join([e[-4:] for e in env])
    impala = IMPALA(env, model, './out_cpu_%s_rodent_%.0E' % (folder_name, lr), 
                    VISION_DIM, PROPRI_DIM, ACTION_DIM)
    
    # Note repeat should be smaller than the number of CPU cores available - 3
    impala.train(max_step=500, max_episodes=int(5e5), repeat=10, batch_size=batch_size,
            cpu_per_learner=cpu_per_learner, learner_params={'lr': lr})
    
    # Generate videos
    # impala.record()

    
if __name__ == '__main__':
    os.environ['SIMULATOR_IMPALA'] = 'rodent'
    envs = [
     #'built-in bowl', 
     'built-in gaps', 
     #'built-in maze', 
     'built-in two-touch',
    ]
    run(envs, 1e-4, 2, 2)
