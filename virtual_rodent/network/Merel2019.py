import torch
import torch.nn as nn

from virtual_rodent.network.base import ModuleBase, ActorBase

from virtual_rodent import VISION_DIM, PROPRI_DIM, ACTION_DIM
from virtual_rodent.network.vision_enc import ResNet18Enc
from virtual_rodent.network.propri_enc import MLPEnc

class MerelModel(ModuleBase):
    def __init__(self, vision_enc, propri_enc, vision_dim, propri_dim, 
                 actor, critic, action_dim):
        super().__init__([vision_enc, propri_enc], [vision_dim, propri_dim], 
                         actor, critic, action_dim)

    def forward(self, vision, propri, actor_hc=None, critic_hc=None, action=None):
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
        
        value, core_h = self.critic(ft_emb, critic_hc)

        ft_emb = torch.cat((propri, ft_emb, core_h.detach()), dim=-1)
        action, log_prob, entropy = self.actor(ft_emb, actor_hc, action)

        return value, (action, log_prob, entropy)

    def reset_rnn(self):
        self.actor.reset_rnn()
        self.critic.reset_rnn()
        
    def detach_hc(self):
        return self.actor.detach_hc(), self.critic.detach_hc()

class Actor(ActorBase):
    def __init__(self, in_dim, action_dim, logit_scale=0, hidden_dim=8):
        super().__init__(in_dim, action_dim, logit_scale)
        self.hidden_dim = hidden_dim
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False, num_layers=2)
        self.hc = None # Hidden and cell layer activations
        self.loc = nn.Linear(hidden_dim, action_dim, bias=False)
        self.logit_scale = nn.Linear(hidden_dim, action_dim, bias=False)
        self.logit_scale.weight.data[:] = 0.
        nn.init.normal_(self.loc.weight, std=0.5/torch.sqrt(torch.tensor(hidden_dim)))
    
    def forward(self, x, hc=None, action=None):
        if hc is None:
            hc = self.hc
        if hc is None:
            rnn_out, self.hc = self.net(x)
        else:
            rnn_out, self.hc = self.net(x, hc)
        loc = self.loc(rnn_out)
        scale = nn.functional.sigmoid(self.logit_scale(rnn_out))/2
        return self.make_action(loc, scale, action)

    def reset_rnn(self):
        self.hc = None

    def detach_hc(self):
        self.hc = tuple([_.detach() for _ in self.hc])
        return tuple([_.clone() for _ in self.hc])

class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.hc = None # Hidden and cell layer activations
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x,  hc=None):
        if hc is None:
            hc = self.hc
        if hc is None:
            rnn_out, self.hc = self.net(x)
        else:
            rnn_out, self.hc = self.net(x, hc)
        return self.proj(rnn_out), rnn_out
        
    def reset_rnn(self):
        self.hc = None
    
    def detach_hc(self):
        self.hc = tuple([_.detach() for _ in self.hc])
        return tuple([_.clone() for _ in self.hc])

def make_model():
    vision_enc = ResNet18Enc()
    vision_emb_dim = vision_enc.get_emb_dim(VISION_DIM)
    
    propri_emb_dim = 20 # propri_dim
    propri_enc = MLPEnc(PROPRI_DIM[0], propri_emb_dim, hidden_dims=(50,))
    
    critic_in_dim = vision_emb_dim + propri_emb_dim
    critic = Critic(critic_in_dim)
    
    actor_in_dim = critic_in_dim + PROPRI_DIM[0] + critic.hidden_dim
    actor = Actor(actor_in_dim, ACTION_DIM, logit_scale=1)
    
    model = MerelModel(vision_enc, propri_enc, VISION_DIM, PROPRI_DIM, 
                       actor, critic, ACTION_DIM) 
    return model