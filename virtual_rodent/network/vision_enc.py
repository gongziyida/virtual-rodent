import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Enc(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        self.dummy_param = nn.Parameter(torch.empty(0)) # For getting data type and device

    def forward(self, x):
        dims = x.shape
        if len(dims) == 3: # add batch dim and restore
            return self.enc(x.unsqueeze(0)).view(-1)
        elif len(dims) > 4: # flatten batch dims and then reshape back
            return self.enc(x.view(-1, *dims[-3:])).view(*dims[:-3], -1)
        elif len(dims) == 4: 
            return self.enc(x).view(dims[0], -1)
        else:
            raise ValueError('Shape mismatch: %s' % dims)

    def get_emb_dim(self, input_dim=None):
        if input_dim is None and not hasattr(self, 'emb_dim'):
            raise ValueError

        if input_dim is not None: # Calculate emb_dim
            x = torch.randn(*input_dim)
            x = x.to(device=self.dummy_param.device, dtype=self.dummy_param.dtype)
            self.emb_dim, = self.forward(x).shape

        return self.emb_dim
