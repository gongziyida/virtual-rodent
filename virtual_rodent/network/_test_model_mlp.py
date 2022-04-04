import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase
from virtual_rodent.network.propri_enc import MLPEnc
from virtual_rodent.network.visual_enc import ResNet18Enc
from virtual_rodent.network.helper import fetch_reset_idx, iter_over_batch_with_reset

class TestModelMLP(ModuleBase):
    def __init__(self, propri_enc, propri_dim, actor, critic, action_dim):
        super().__init__([propri_enc], [propri_dim], actor, critic, action_dim)

    def forward(self, propri):
        ft_emb = self.encode([propri])

        action, log_prob, entropy = self.actor(ft_emb)
        value = self.critic(ft_emb)
        return value, action, log_prob, entropy

class ActorMLP(ActorBase):
    def __init__(self, in_dim, action_dim):
        super().__init__(in_dim, action_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, 200),
                                   nn.LeakyReLU(),
                                   nn.Linear(200, action_dim))

    def forward(self, x):
        dims = x.shape
        if len(dims) > 2: # Flatten batch dims and reshape back
            mean = self.net(x.view(-1, dims[-1])).view(*dims[:-1], -1)
        elif len(dims) == 1: # Add batch dim and restore
            mean = self.net(x.unsqueeze(0)).squeeze(0)
        else: # len == 2
            mean = self.net(x)
        return self.make_action(mean)

class CriticMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__(in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, 200),
                                   nn.LeakyReLU(),
                                   nn.Linear(200, 1))

    def forward(self, x):
        dims = x.shape
        if len(dims) > 2: # Flatten batch dims and reshape back
            return self.net(x.view(-1, dims[-1])).view(*dims[:-1], 1)
        elif len(dims) == 1: # Add batch dim and restore
            return self.net(x.unsqueeze(0)).squeeze(0)
        else: # len == 2
            return self.net(x)

