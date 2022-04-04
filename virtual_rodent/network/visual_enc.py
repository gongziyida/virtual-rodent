import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Enc(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])

    def forward(self, x):
        dims = x.shape
        if len(dims) == 3: # add batch dim and restore
            return self.enc(x.unsqueeze(0)).squeeze(0)
        elif len(dims) > 4: # flatten batch dims and then reshape back
            return self.enc(x.view(-1, *dims[-3:])).view(*dims[:-3], -1)
        elif len(dims) == 4: 
            return self.enc(x).view(dims[0], -1)
        else:
            raise ValueError('Shape mismatch: %s' % dims)
