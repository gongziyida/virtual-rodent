import torch
import torch.nn as nn

from virtual_rodent import VISUAL_DIM, PROPRI_DIM, ACTION_DIM
from .building_blocks import MLP
from .helper import fetch_reset_idx, iter_over_batch_with_reset 


class MerelModel(nn.Module):
    def __init__(self, visual_enc, propri_enc, core_hidden_dim,
                 sampling_dist=torch.distributions.Normal):
        super().__init__()

        self.visual_enc = visual_enc

        self.propri_enc = propri_enc

        with torch.no_grad():
            visual_emb_dim = self.visual_enc(torch.rand(1, *VISUAL_DIM)).squeeze().data.shape[0]
            propri_emb_dim = self.propri_enc(torch.rand(1, PROPRI_DIM)).squeeze().data.shape[0]

        self.core_in_dim = visual_emb_dim + propri_emb_dim
        self.core_hidden_dim = core_hidden_dim
        self.core = nn.LSTM(self.core_in_dim, self.core_hidden_dim, batch_first=False)
        self.core_hc = None

        self.value = MLP(in_dim=self.core_hidden_dim, out_dim=1, d=5)

        self.policy_in_dim = self.core_in_dim + self.core_hidden_dim + PROPRI_DIM
        self.policy = nn.LSTM(self.policy_in_dim, ACTION_DIM, num_layers=3, batch_first=False)
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

            done: list
                Contains boolean. Assume shape: (batch)(T+1), including state -1
        returns
        -------
            value: torch.tensor
            pi: torch.Distribution
            reset_idx: nested list or None
        """
        if len(visual.shape) == 3:
            assert len(propri.shape) == 1
            visual = visual.view(1, 1, *visual.shape)
            propri = propri.view(1, 1, *propri.shape)
            T, batch = 1, 1
        else:
            assert len(visual.shape) == 5 and len(propri.shape) == 3
            T, batch = visual.shape[:2]
            assert T == propri.shape[0] and batch == propri.shape[1]

        # Sensory encoding
        visual_ft = self.visual_enc(visual.view(-1, *visual.shape[-3:])).view(T, batch, -1)
        propri_ft = self.propri_enc(propri.view(-1, propri.shape[-1])).view(T, batch, -1)
        ft_concat = torch.cat((visual_ft, propri_ft), dim=-1)
        assert len(ft_concat.shape) == 3

        # RNNs
        reset_idx = fetch_reset_idx(done, T, batch)
        
        core_out, self.core_hc = iter_over_batch_with_reset(self.core, ft_concat, 
                                                            reset_idx, self.core_hc)

        policy_input = torch.cat((core_out.detach(), ft_concat, propri), dim=-1)
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
