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
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class ResNet18Enc(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])

    def forward(self, x):
        if len(x.shape) == 3: # add batch dim
            x = x.unsqueeze(0)
        return self.enc(x).view(x.shape[0], -1) # (batch, emb dim)
