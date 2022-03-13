import torch
import torch.nn as nn
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, d=2):
        super().__init__()
        a = in_dim
        l = []
        while True:
            b = max(a // d, out_dim)
            l.extend([nn.Linear(a, b), nn.ReLU()])
            a = b
            if b == out_dim:
                del l[-1]
                break
        self.net = nn.Sequential(*l)

    def forward(self, x):
        return self.net(x)

class MerelModel(nn.Module):
    MIN_VISUAL_IN_DIM = (3, 224, 224) # Minimal visual input dimension required
    
    def __init__(self, visual_in_dim, propri_in_dim, propri_out_dim, action_dim, core_hidden_dim, 
                 sampling_dist=torch.distributions.Normal):
        super().__init__()
        self.visual_enc = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        if not all(map(lambda ref, x: x >= ref, zip(MIN_VISUAL_IN_DIM, visual_in_dim))):
            raise ValueError('Invalid `visual_in_dim`')
        self.visual_in_dim = visual_in_dim
        self.visual_out_dim = self.visual_enc(torch.rand(*visual_in_dim)).data.shape

        self.propri_in_dim = propri_in_dim
        self.propri_out_dim = propri_out_dim
        self.propri_enc = MLP(propri_in_dim, propri_out_dim)

        # TODO: check LSTM dimensions
        self.core_in_dim = self.visual_out_dim + self.propri_out_dim
        self.core_hidden_dim = core_hidden_dim
        self.core = nn.LSTM(self.core_in_dim, self.core_hidden_dim)

        self.policy_in_dim = self.core_in_dim + self.core_in_dim + self.propri_in_dim
        self.action_dim = action_dim
        self.policy_hidden_dim = action_dim
        self.policy = nn.LSTM(self.policy_in_dim, self.policy_hidden_dim, num_layers=3)

        self.sampling_dist = sampling_dist

    def forward(self, visual, propri, sample_size):
        visual_ft = self.visual_enc(visual)
        propri_ft = self.propri_enc(propri)
        ft_concat = torch.cat(visual_ft, propri_ft)
        value = self.core(ft_concat)
        pi = self.policy(torch.cat(value.detach(), ft_concat, propri))
        return value, self.sample_policy(pi, sample_size)

    def sample_policy(self, pi, sample_size):
        dist = self.sampling_dist(pi, torch.tensor(1))
        return dist.sample([sample_size,])
