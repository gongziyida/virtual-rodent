import torch
import torch.nn as nn
from .helper import check_network

class MerelModel(nn.Module):
    def __init__(self, visual_enc, visual_dim, propri_enc, propri_dim, action_dim, core_hidden_dim,
                 sampling_dist=torch.distributions.Normal):
        super().__init__()

        self.visual_dim = visual_dim  # (input_dim, output_dim)
        self.visual_enc = visual_enc

        self.propri_dim = propri_dim  # (input_dim, output_dim)
        self.propri_enc = propri_enc

        # TODO: check LSTM dimensions
        self.core_in_dim = self.visual_dim[1] + self.propri_dim[1]
        self.core_hidden_dim = core_hidden_dim
        self.core = nn.LSTM(self.core_in_dim, self.core_hidden_dim)

        self.value = MLP(in_dim=self.core_hidden_dim, out_dim=1, d=5)

        self.policy_in_dim = self.core_in_dim + self.core_hidden_dim + self.propri_dim[0]
        self.action_dim = action_dim
        self.policy_hidden_dim = action_dim
        self.policy = nn.LSTM(self.policy_in_dim, self.policy_hidden_dim, num_layers=3)

        self.sampling_dist = sampling_dist

    def forward(self, visual, propri, sample_size=1):
        visual_ft = self.visual_enc(visual)
        propri_ft = self.propri_enc(propri)
        ft_concat = torch.cat(visual_ft, propri_ft)
        core_h = self.core(ft_concat)
        pi = self.policy(torch.cat(core_h.detach(), ft_concat, propri))
        return self.value(core_h), self.sample_policy(pi, sample_size)

    def sample_policy(self, pi, sample_size):
        dist = self.sampling_dist(pi, torch.tensor(1))
        return dist.sample([sample_size,])
