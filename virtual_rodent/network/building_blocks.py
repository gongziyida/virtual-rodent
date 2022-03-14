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


class ResNet18_Enc(nn.Module):
    def __init__(self, in_dim, pretrained=True):
        super().__init__()

        self.enc = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        with torch.no_grad():
            self.in_dim, self.out_dim = in_dim, self.enc(torch.rand(*in_dim)).data.shape) # (input_dim, output_dim)

    def forward(self, x):
        return self.enc(x)
