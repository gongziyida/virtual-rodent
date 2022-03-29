import torch
import torch.nn as nn

from .building_blocks import MLP
from .helper import fetch_reset_idx, iter_over_batch_with_reset 


class TestModelLinear(nn.Module):
    def __init__(self, propri_enc, sampling_dist=torch.distributions.Normal):
        super().__init__()
        self.propri_enc = propri_enc

        with torch.no_grad():
            propri_emb_dim = self.propri_enc(torch.rand(1, 15)).squeeze().data.shape[0]

        self.value = nn.Sequential(nn.LeakyReLU(), 
                                   nn.Linear(propri_emb_dim, 1))
        self.policy = nn.Sequential(nn.LeakyReLU(), 
                                    nn.Linear(propri_emb_dim, 4))

        self.sampling_dist = sampling_dist

        self._episode = nn.Parameter(torch.tensor(-1.0), requires_grad=False) # -1: init version

    def forward(self, visual, propri, done=None):
        if len(propri.shape) == 1:
            propri = propri.view(1, 1, *propri.shape)
            T, batch = 1, 1
            reset_idx = None
        else:
            T, batch = propri.shape[:2]
            reset_idx = fetch_reset_idx(done, T, batch)

        # Sensory encoding
        propri_ft = self.propri_enc(propri.view(-1, propri.shape[-1])).view(T, batch, -1)

        value = self.value(propri_ft)
        policy_out = self.policy(propri_ft)
        
        if T == 1 and batch == 1:
            value = value.squeeze()
            policy_out = policy_out.squeeze()
        pi = self.sampling_dist(policy_out, torch.tensor(1).to(policy_out.device))
        return value, pi, reset_idx

    def _reset_episode(self): # For testing
        self._episode *= 0
        self._episode -= 1
    def _update_episode(self, val=1): # For testing
        self._episode += val


class TestModel(nn.Module):
    def __init__(self, propri_enc, sampling_dist=torch.distributions.Normal):
        super().__init__()
        self.propri_enc = propri_enc

        with torch.no_grad():
            propri_emb_dim = self.propri_enc(torch.rand(1, 15)).squeeze().data.shape[0]

        self.core = nn.LSTM(propri_emb_dim, propri_emb_dim, batch_first=False)
        self.core_hc = None

        self.value = MLP(propri_emb_dim, out_dim=1, d=5)

        self.policy_in_dim = 4 + 4 + 15
        self.policy = nn.LSTM(self.policy_in_dim, 4, batch_first=False)
        self.policy_hc = None

        self.sampling_dist = sampling_dist

        self._episode = nn.Parameter(torch.tensor(-1.0), requires_grad=False) # -1: init version

    def reset_rnn(self):
        self.core_hc = None
        self.policy_hc = None

    def forward(self, visual, propri, done):
        """
        Parameters
        ----------
        visual/propri: torch.tensors
            If the visual/propri are batched sequence, the shape is assumed to be
            (T, batch, channel, length, width,) and (T, batch, keypoints,); or
            (channel, length, width,) and (keypoints,)
            In the second case ignore done and returns will be squeezed

            done: list or None
                If not None, assume shape: (batch)(T)
        returns
        -------
            value: torch.tensor
            pi: torch.Distribution
            reset_idx: nested list or None
        """
        if len(propri.shape) == 1:
            propri = propri.view(1, 1, *propri.shape)
            T, batch = 1, 1
        else:
            T, batch = propri.shape[:2]

        # Sensory encoding
        propri_ft = self.propri_enc(propri.view(-1, propri.shape[-1])).view(T, batch, -1)

        # RNNs
        reset_idx = fetch_reset_idx(done, T, batch)
        
        core_out, self.core_hc = iter_over_batch_with_reset(self.core, propri_ft, 
                                                            reset_idx, self.core_hc)

        policy_input = torch.cat((core_out.detach(), propri_ft, propri), dim=-1)
        policy_out, self.policy_hc = iter_over_batch_with_reset(self.policy, policy_input,
                                                                reset_idx, self.policy_hc)

        value = self.value(core_out)
        if T == 1 and batch == 1:
            value = value.squeeze()
            policy_out = policy_out.squeeze()
        pi = self.sampling_dist(policy_out, torch.tensor(1).to(policy_out.device))
        return value, pi, reset_idx

    def _reset_episode(self): # For testing
        self._episode *= 0
        self._episode -= 1
    def _update_episode(self, val=1): # For testing
        self._episode += val
