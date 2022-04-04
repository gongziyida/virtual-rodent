import torch
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self, encoders, in_dims, actor, critic, action_dim):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.in_dims
        with torch.no_grad():
            self.ft_dims = tuple([enc(torch.randn(1, *in_dim)).squeeze().data.shape[0]
                                  for enc, in_dim in zip(encoders, in_dims)])

        self.actor = actor
        self.critic = critic
        self.action_dim = action_dim

    def encode(self, inputs):
        return *[enc(i) for enc, i in zip(self.encoders, inputs)]

    def forward(self, state):
        raise NotImplementedError

    def get_encoders(self):
        return self.encoders

    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic


class ActorBase(nn.Module):
    def __init__(self, in_dim, action_dim, dist=torch.distributions.Normal):
        super().__init__()
        self.in_dim = in_dim
        self.action_dim = action_dim
        self.log_scale = nn.Parameter(torch.full((action_dim,), 0.5))
        self.dist = dist

    def make_action(self, loc, action=None):
        scale = torch.clamp(torch.exp(self.log_scale), 1e-3, 10)
        scale = scale.view(*([1] * len(loc.shape[:-1])), *scale.shape)
        assert len(scale.shape) == len(loc.shape)
        pi = self.dist(loc, scale)
        action_ = pi.sample() if action is None else action
        log_prob = pi.log_prob(action_)
        entropy = pi.entropy()
        return action_, log_prob, entropy

    def forward(self, state):
        raise NotImplementedError

