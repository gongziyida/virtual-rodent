import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class ResNet18Enc(nn.Module):
    def __init__(self, emb_dim, pretrained=True):
        super().__init__()
        res_net = list(models.resnet18(weights='IMAGENET1K_V1').children())
        d = res_net[-1].in_features
        self.enc = nn.Sequential(*res_net[:-1], nn.Flatten())
        for param in self.enc.parameters():
            param.requires_grad = False
        self.out = nn.Sequential(nn.Linear(d, emb_dim), nn.LayerNorm(emb_dim))
        self.dummy_param = nn.Parameter(torch.empty(0)) # For getting data type and device
        # required preprocessing for ResNet. See PyTorch's documentation
        # self.preprocessing = transforms.Compose([
        #     # transforms.Resize(256), transforms.CenterCrop(224),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def forward(self, x):
        ''' x needs to be already in [0, 1]
        '''
        dims = x.shape
        # required normalization for ResNet. See PyTorch's documentation
        x[...,0,:,:] = (x[...,0,:,:] - 0.485) / 0.229
        x[...,1,:,:] = (x[...,1,:,:] - 0.456) / 0.224
        x[...,2,:,:] = (x[...,2,:,:] - 0.406) / 0.225
        # if len(dims) < 5: # otherwise, see below preprocessing
        #     x = self.preprocessing(x)
        
        if len(dims) == 3: # add batch dim and restore
            return self.out(self.enc(x.unsqueeze(0))).squeeze(0)
        elif len(dims) == 4:
            return self.out(self.enc(x))
        elif len(dims) == 5: # flatten and then reshape back
            T, batch_size = dims[:2]
            x = torch.flatten(x, 0, 1)
            # x = self.preprocessing(x)
            return self.out(self.enc(x)).view(T, batch_size, -1)
        else:
            raise ValueError('Shape mismatch: %s' % dims)
