import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase

class TestModel(ModuleBase):
    def __init__(self, actor, critic):
        super().__init__([None], [2], actor, critic, 2)

    def forward(self, state, action=None):
        propri = state[1]
        ft_emb = self.encode([propri])[0]
        assert (ft_emb == propri).all()

        action, log_prob, entropy = self.actor(ft_emb, action)
        value = self.critic(torch.cat((ft_emb, action), -1))
        return value, (action, log_prob, entropy)

class Actor(ActorBase):
    def __init__(self):
        super().__init__(2, 2)
        self.net = nn.Linear(2, 2)

    def forward(self, state, action=None):
        dims = state.shape
        if len(dims) > 2: # Flatten batch dims and reshape back
            mean = self.net(state.view(-1, dims[-1])).view(*dims[:-1], -1)
        elif len(dims) == 1: # Add batch dim and restore
            mean = self.net(state.unsqueeze(0)).squeeze(0)
        else: # len == 2
            mean = self.net(state)
        return self.make_action(mean, action)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 1))

    def forward(self, state):
        dims = state.shape
        if len(dims) > 2: # Flatten batch dims and reshape back
            return self.net(state.view(-1, dims[-1])).view(*dims[:-1], 1)
        elif len(dims) == 1: # Add batch dim and restore
            return self.net(state.unsqueeze(0)).squeeze(0)
        else: # len == 2
            return self.net(state)

