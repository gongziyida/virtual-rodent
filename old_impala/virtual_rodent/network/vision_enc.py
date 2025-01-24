import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Enc(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1],
                                 nn.Flatten())
        self.dummy_param = nn.Parameter(torch.empty(0)) # For getting data type and device

    def forward(self, x):
        dims = x.shape
        if len(dims) == 3: # add batch dim and restore
            return self.enc(x.unsqueeze(0)).squeeze(0)
        elif len(dims) == 4:
            return self.enc(x)
        elif len(dims) == 5: # flatten and then reshape back
            T, batch_size = dims[:2]
            return self.enc(torch.flatten(x, 0, 1)).view(T, batch_size, -1)
        else:
            raise ValueError('Shape mismatch: %s' % dims)

    def get_emb_dim(self, input_dim=None):
        if input_dim is None and not hasattr(self, 'emb_dim'):
            raise ValueError('input_dim must be given for the first time.')

        if input_dim is not None: # Calculate emb_dim
            c, w, h = input_dim
            x = torch.randn(c, w, h)
            x = x.to(device=self.dummy_param.device, dtype=self.dummy_param.dtype)
            self.emb_dim, = self.forward(x).shape

        return self.emb_dim
