import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from virtual_rodent import VISION_DIM, PROPRI_DIM, ACTION_DIM
from virtual_rodent.environment import MAPPER
from virtual_rodent.visualization import video
from virtual_rodent.simulation import simulate
from virtual_rodent.network.vision_enc import ResNet18Enc
from virtual_rodent.network.propri_enc import MLPEnc
import virtual_rodent.network.Merel2019 as Merel2019
from virtual_rodent.utils import load_checkpoint

eps = np.finfo(np.float32).eps.item()
save_dir = './results/'

def update(gamma, optimizer, log_probs, values, rewards):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in rewards[::-1]:
        # calculate the discounted value
        R = r.item() + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob.squeeze() * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, R))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

def main(max_episode):
    # run infinitely many episodes
    env_name = 'gaps'
    env, propri_attr = MAPPER[env_name]()
    
    vision_enc = ResNet18Enc()
    vision_emb_dim = vision_enc.get_emb_dim(VISION_DIM)
    
    propri_emb_dim = 20 # propri_dim
    propri_enc = MLPEnc(PROPRI_DIM[0], propri_emb_dim, hidden_dims=(50,))
    
    critic_in_dim = vision_emb_dim + propri_emb_dim
    critic = Merel2019.Critic(critic_in_dim)
    
    actor_in_dim = critic_in_dim + PROPRI_DIM[0] + critic.hidden_dim
    actor = Merel2019.Actor(actor_in_dim, ACTION_DIM)
    
    model = Merel2019.MerelModel(vision_enc, propri_enc, VISION_DIM, PROPRI_DIM, 
                                 actor, critic, ACTION_DIM) 
    optimizer = optim.Adam(model.parameters(), lr=5e-2)
    ext_cam = (0,)
    ext_cam_size = (200, 200)

    rewards = []
    for i_episode in tqdm(range(max_episode)):
        save = (i_episode % 500 == 0) or (i_episode == max_episode - 1)
        
        ret = simulate(env, model, propri_attr, max_step=80, device=torch.device('cpu'), 
                       ext_cam=ext_cam if save else set())
        rewards.append(torch.stack(ret['reward']).squeeze().detach().cpu().numpy())
        update(gamma=0.9, optimizer=optimizer, 
               log_probs=ret['log_policy'], values=ret['value'], rewards=ret['reward'])

        if save:
            for i in ext_cam:
                anim = video(ret[f'cam{i}'])
                fname = f'{env_name}_{i_episode+1}_cam{i}.gif'
                anim.save(os.path.join(save_dir, fname), writer='pillow')
                plt.clf()
            if i_episode >= 500:
                torch.save(model.state_dict(), 
                           os.path.join(save_dir, f'weights{i_episode+1}.pth'))
    
    np.save(os.path.join(save_dir, 'rewards.npy'), np.stack(rewards, 0))

if __name__ == "__main__":
    main(5000)