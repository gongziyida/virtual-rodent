import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase
from virtual_rodent.network.propri_enc import MLPEnc
from virtual_rodent.network.visual_enc import ResNet18Enc
from virtual_rodent.network.helper import fetch_reset_idx, iter_over_batch_with_reset

class TestModelRNN(ModuleBase):
    def __init__(self, propri_enc, propri_dim, actor, critic, action_dim):
        super().__init__([propri_enc], [propri_dim], actor, critic, action_dim)

    def forward(self, state):
        propri, done = state
        ft_emb = self.encode([propri])

        # The feature embedding should have shape (T, batch, embedding)
        dims = ft_emb.shape
        if len(dims) == 1:
            ft_emb = ft_emb.view(1, 1, *dims)
            T, batch = 1, 1
        elif len(dims) == 3:
            T, batch = ft_emb.shape[:2]
        else: 
            raise ValueError('%s' % dims)

        reset_idx = fetch_reset_idx(done, T, batch)
        
        action, log_prob, entropy = self.actor((ft_emb, reset_idx))
        value = self.critic((ft_emb, reset_idx))

        return value, action, log_prob, entropy

class ActorRNN(ActorBase):
    def __init__(self, in_dim, action_dim, hidden_dim=8):
        super().__init__(in_dim, action_dim)
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x, reset_idx = state
        out, self.hc = iter_over_batch_with_reset(self.net, x, reset_idx, self.hc)
        return self.make_action(out)


class CriticRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__(in_dim)
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x, reset_idx = state
        value, self.hc = iter_over_batch_with_reset(self.net, x, reset_idx, self.hc)
        return value
