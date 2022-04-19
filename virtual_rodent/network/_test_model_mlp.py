import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase

class TestModel(ModuleBase):
    def __init__(self, propri_enc, propri_dim, actor, critic, action_dim):
        super().__init__([propri_enc], [propri_dim], actor, critic, action_dim)

    def forward(self, propri, vision=None, reset_idx=None, action=None):
        ft_emb = self.encode([propri])[0]

        action, log_prob, entropy = self.actor(ft_emb, action)
        value = self.critic(ft_emb)
        return value, (action, log_prob, entropy)

class Actor(ActorBase):
    def __init__(self, in_dim, action_dim):
        super().__init__(in_dim, action_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, action_dim))

    def forward(self, state, action=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
        mean = self.net(state)
        return self.make_action(mean, action)

class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 1))

    def forward(self, state):
        # type: (Tensor) -> Tensor
        return self.net(state)

