import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ModuleBase(nn.Module):
    def __init__(self, encoders, in_dims, actor, critic, action_dim):
        super().__init__()
        self.in_dims = in_dims
        self.idx_encoded = [i for i, enc in enumerate(encoders) if enc is not None]
        self.encoders = [encoders[i] for i in self.idx_encoded]
        if len(self.encoders) > 0:
            self.encoders = nn.ModuleList(self.encoders)
            with torch.no_grad():
                self.ft_dims = [encoders[i](torch.randn(1, *in_dim[i])).squeeze().data.shape[0]
                                for i in self.idx_encoded]

        self.actor = actor
        self.critic = critic
        self.action_dim = action_dim

    def encode(self, inputs):
        if len(self.idx_encoded) == 0:
            return inputs

        ret = []
        for i, x in enumerate(inputs):
            ret.append(self.encoders[i](x) if i in self.idx_encoded else x)
        return ret

    def forward(self, state, action=None):
        raise NotImplementedError

    def get_encoders(self):
        return self.encoders

    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic


class ActorBase(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.in_dim = in_dim
        self.action_dim = action_dim
        self.log_scale = nn.Parameter(torch.full((action_dim,), 0.5))

    def make_action(self, loc, action=None):
        '''
        returns
        -------
        action_: torch.tensor
            Shape (T, batch, action_dim)
        log_prob: torch.tensor
            Shape (T, batch, 1)
        entropy: torch.tensor
            Shape (T, batch, 1)
        '''
        scale = torch.clamp(torch.exp(self.log_scale), 1e-3, 10)
        scale = scale.view(*([1] * len(loc.shape[:-1])), *scale.shape)
        assert len(scale.shape) == len(loc.shape)
        pi = MultivariateNormal(loc, torch.diag_embed(scale))
        action_ = pi.sample() if action is None else action
        log_prob = pi.log_prob(action_).unsqueeze(-1)
        entropy = pi.entropy().unsqueeze(-1)
        return action_, log_prob, entropy

    def forward(self, state, action=None):
        raise NotImplementedError

