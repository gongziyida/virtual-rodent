import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributed.rpc import RRef

class ModuleBase(nn.Module):
    def __init__(self, encoders, in_dims, actor, critic, action_dim):
        super().__init__()
        self.in_dims = in_dims
        self.idx_encoded = [i for i, enc in enumerate(encoders) if enc is not None]
        self.encoders = [encoders[i] for i in self.idx_encoded]
        if len(self.encoders) > 0:
            self.encoders = nn.ModuleList(self.encoders)

        self.actor = actor
        self.critic = critic
        self.action_dim = action_dim

        self._dummy = nn.Parameter(torch.tensor(1), requires_grad=False)

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

    def parameters_rref(self):
        param_rrefs = []
        for param in self.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs


class ActorBase(nn.Module):
    def __init__(self, in_dim, action_dim, logit_scale=0):
        super().__init__()
        self.in_dim = in_dim
        self.action_dim = action_dim
    
    @torch.jit.ignore
    def make_action(self, loc, scale, action=None):
        '''
        returns
        -------
        action_: torch.tensor
            Shape (T, batch, action_dim)
        log_prob: torch.tensor
            Shape (T, batch)
        entropy: torch.tensor
            Shape (T, batch)
        '''
        pi = Normal(loc, scale)
        action_ = pi.sample() if action is None else action.detach()
        log_prob = pi.log_prob(action_).sum(dim=-1)
        entropy = pi.entropy().mean(dim=-1)
        return action_, log_prob, entropy

    def forward(self, state, action=None):
        raise NotImplementedError


