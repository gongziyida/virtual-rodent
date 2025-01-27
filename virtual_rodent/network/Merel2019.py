import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from virtual_rodent.network.base import ModuleBase, ActorBase

from virtual_rodent import VISION_DIM, PROPRI_DIM, ACTION_DIM
from virtual_rodent.network.vision_enc import ResNet18Enc
from virtual_rodent.network.propri_enc import MLPEnc

class MerelModel(ModuleBase):
    def __init__(self, vision_enc, propri_enc, vision_dim, propri_dim, 
                 actor, critic, action_dim):
        super().__init__([vision_enc, propri_enc], [vision_dim, propri_dim], 
                         actor, critic, action_dim)

    def forward(self, vision, propri, action_raw=None, train=True):
        v_dim, p_dim = vision.shape, propri.shape
        assert len(v_dim) == 3 or len(v_dim) == 5
        assert len(p_dim) == 1 or len(p_dim) == 3
        if len(v_dim) == 3:
            vision = vision.unsqueeze(0).unsqueeze(0)
        elif len(v_dim) != 5:
            raise ValueError(v_dim)
        if len(p_dim) == 1:
            propri = propri.unsqueeze(0).unsqueeze(0)
        elif len(p_dim) != 3:
            raise ValueError(p_dim)

        vision_ft, propri_ft = self.encode([vision, propri])
        
        ft_emb = torch.cat((vision_ft, propri_ft), dim=-1)

        assert len(ft_emb.shape) == 3 # should have shape (T, batch, embedding)
        
        value, core_h = self.critic(ft_emb)

        ft_emb = torch.cat((propri, ft_emb, core_h.detach()), dim=-1)
        action_raw, action, log_prob, entropy = self.actor(ft_emb, action_raw, train)

        return value, (action_raw, action, log_prob, entropy)
    
class Actor(nn.Module):
    def __init__(self, in_dim, action_dim, logit_scale=0, hidden_dim=8):
        super().__init__()
        self.in_dim, self.hidden_dim, self.action_dim = in_dim, hidden_dim, action_dim
        dim_, self.n_per_col = 15, 2
        self.net = nn.Sequential(nn.Linear(in_dim, 15), nn.ReLU(), 
                                 nn.Linear(15, hidden_dim), nn.ReLU(), 
                                 nn.Linear(hidden_dim, dim_), nn.ReLU(), 
                                 nn.Linear(dim_, dim_*2))
        
        w = torch.from_numpy(_gen_binary_mat(dim_, action_dim, self.n_per_col)).to(torch.float32)
        self.proj = nn.Parameter(w, requires_grad=False)
    
    def forward(self, x, action_raw=None, train=True):
        '''
        returns
        -------
        (action_raw, action): torch.tensor
            Shape (T, batch, action_dim)
        log_prob: torch.tensor
            Shape (T, batch)
        entropy: torch.tensor
            Shape (T, batch)
        '''
        aux = self.net(x)
        
        loc, log_scale = aux[...,:aux.shape[-1]//2], aux[...,aux.shape[-1]//2:]
        # loc = (loc - loc.mean(dim=-1, keepdim=True)) / loc.std(dim=-1, keepdim=True) / 4
        # log_scale = (log_scale - log_scale.mean(dim=-1, keepdim=True)) / \
        #             log_scale.std(dim=-1, keepdim=True)
        
        sd = torch.sqrt(torch.tensor(self.n_per_col))
        if train:
            scale = torch.exp(log_scale) * 0.5
            # scale = torch.ones_like(loc) / 3
            
            pi = Normal(loc, scale)
            entropy = pi.entropy().mean(dim=-1)
    
            if action_raw is None:
                action_raw = pi.sample()
            correction = torch.log(1 - F.tanh(action_raw)**2 + 1e-8).sum(dim=-1)
            log_prob = pi.log_prob(action_raw).sum(dim=-1) - correction
        else:
            action_raw = loc.detach()
            log_prob, entropy = None, None
        action = F.tanh(action_raw) @ self.proj / sd # to high dim
        return action_raw, action, log_prob, entropy

class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 15), nn.ReLU(), 
                                 nn.Linear(15, hidden_dim), nn.ReLU())
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.net(x)
        return self.proj(out), out
        

def _gen_binary_mat(K, N, d):
    ''' Generate a rank-K K x N binary matrix (K < N) where each column has at most d 1s
    '''
    rng = np.random.default_rng()
    idx = list(range(K))
    mat = np.zeros((K, N))
    choosen_tuples = []
    while np.linalg.matrix_rank(mat) != K: # force full ranks
        mat[:] = 0
        for i in range(N):
            t = tuple(map(int,rng.choice(idx, size=d, replace=False)))
            while t in choosen_tuples:
                t = tuple(map(int,rng.choice(idx, size=d, replace=False)))
            choosen_tuples.append(t)
            mat[t,i] = 1/np.sqrt(d)
    assert np.allclose(np.linalg.norm(mat, axis=0), 1)
    return mat

def make_model(vision_emb_dim=20, propri_emb_dim=20, propri_hidden_dim=(50,)):
    vision_enc = ResNet18Enc(vision_emb_dim)
    propri_enc = MLPEnc(PROPRI_DIM[0], propri_emb_dim, hidden_dims=propri_hidden_dim)
    
    critic_in_dim = vision_emb_dim + propri_emb_dim
    critic = Critic(critic_in_dim)
    
    actor_in_dim = critic_in_dim + PROPRI_DIM[0] + critic.hidden_dim
    actor = Actor(actor_in_dim, ACTION_DIM, logit_scale=1)
    
    model = MerelModel(vision_enc, propri_enc, VISION_DIM, PROPRI_DIM, 
                       actor, critic, ACTION_DIM) 
    return model