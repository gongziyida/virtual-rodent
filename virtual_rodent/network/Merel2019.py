import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase, RNNBase
# from virtual_rodent.network.helper import iter_over_batch_with_reset

class MerelModel(ModuleBase):
    def __init__(self, vision_enc, propri_enc, vision_dim, propri_dim, 
                 actor, critic, action_dim):
        super().__init__([vision_enc, propri_enc], [vision_dim, propri_dim], 
                         actor, critic, action_dim)

    def forward(self, state, action=None):
        vision, propri, reset_idx = state
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
        
        value, core_h = self.critic((ft_emb, reset_idx))

        ft_emb = torch.cat((propri, ft_emb, core_h.detach()), dim=-1)
        action, log_prob, entropy = self.actor((ft_emb, reset_idx), action)

        return value, (action, log_prob, entropy)

class Actor(ActorBase):
    def __init__(self, in_dim, action_dim, hidden_dim=8):
        super().__init__(in_dim, action_dim)
        self.hidden_dim = hidden_dim
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False, num_layers=3)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, action=None):
        x, reset_idx = state
        rnn_out, self.hc = self.iter_over_batch_with_reset(x, reset_idx, self.hc)
        return self.make_action(self.proj(rnn_out), action)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x, reset_idx = state
        rnn_out, self.hc = self.iter_over_batch_with_reset(x, reset_idx, self.hc)
        return self.proj(rnn_out), rnn_out
