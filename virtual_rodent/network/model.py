import torch
import torch.nn as nn

from virtual_rodent import VISUAL_DIM, PROPRI_DIM, ACTION_DIM
from .building_blocks import MLP

class MerelModel(nn.Module):
    def __init__(self, visual_enc, propri_enc, core_hidden_dim,
                 sampling_dist=torch.distributions.Normal):
        super().__init__()

        self.visual_enc = visual_enc

        self.propri_enc = propri_enc

        with torch.no_grad():
            visual_emb_dim = self.visual_enc(torch.rand(1, *VISUAL_DIM)).squeeze().data.shape[0]
            propri_emb_dim = self.propri_enc(torch.rand(1, PROPRI_DIM)).squeeze().data.shape[0]

        # TODO: check LSTM dimensions
        self.core_in_dim = visual_emb_dim + propri_emb_dim
        self.core_hidden_dim = core_hidden_dim
        self.core = nn.LSTM(self.core_in_dim, self.core_hidden_dim)

        self.value = MLP(in_dim=self.core_hidden_dim, out_dim=1, d=5)

        self.policy_in_dim = self.core_in_dim + self.core_hidden_dim + PROPRI_DIM
        self.policy = nn.LSTM(self.policy_in_dim, ACTION_DIM, num_layers=3)

        self.sampling_dist = sampling_dist

    def forward(self, visual, propri, sample_size=1):
        if len(visual.shape) == 3:
            visual = visual.unsqueeze(0)
        if len(propri.shape) == 1:
            propri = propri.unsqueeze(0)

        visual_ft = self.visual_enc(visual)
        propri_ft = self.propri_enc(propri)
        ft_concat = torch.cat((visual_ft, propri_ft), dim=1)
        _, (core_h, _) = self.core(ft_concat)

        _, (pi, _) = self.policy(torch.cat((core_h.detach(), ft_concat, propri), dim=1))
        value = self.value(core_h)
        action = self.sample_policy(pi[-1].squeeze(), sample_size) # stacked LSTM, pick last layer
        return value, action

    def sample_policy(self, pi, sample_size):
        dist = self.sampling_dist(pi, torch.tensor(1).to(pi.device))
        return dist.sample([sample_size,])
