import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase
from virtual_rodent.network.helper import iter_over_batch_with_reset

class TestModel(ModuleBase):
    def __init__(self, propri_enc, propri_dim, actor, critic, action_dim):
        super().__init__([propri_enc], [propri_dim], actor, critic, action_dim)

    def forward(self, state, action=None):
        _, propri, reset_idx = state
        ft_emb = self.encode([propri])[0]

        # The feature embedding should have shape (T, batch, embedding)
        dims = ft_emb.shape
        if len(dims) == 1:
            ft_emb = ft_emb.view(1, 1, *dims)
        elif len(dims) < 3: 
            raise ValueError('%s' % dims)
        
        action, log_prob, entropy = self.actor((ft_emb, reset_idx), action)
        value = self.critic((ft_emb, reset_idx))

        return value, action, log_prob, entropy

class Actor(ActorBase):
    def __init__(self, in_dim, action_dim, hidden_dim=8):
        super().__init__(in_dim, action_dim)
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, action=None):
        x, reset_idx = state
        rnn_out, self.hc = iter_over_batch_with_reset(self.net, x, reset_idx, self.hc)
        return self.make_action(self.proj(rnn_out), action)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x, reset_idx = state
        rnn_out, self.hc = iter_over_batch_with_reset(self.net, x, reset_idx, self.hc)
        return self.proj(rnn_out)
