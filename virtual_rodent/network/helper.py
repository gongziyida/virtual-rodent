import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        ret = self.enc(x)
        if len(ret) == 2:
            mu, var = ret
        else:
            mid = ret.shape[-1]//2
            mu, var = ret[..., :mid], ret[..., mid:]
        z = mu + var * torch.randn_like(var)
        x_hat = self.dec(z)
        return x_hat

class MLPMirror(nn.Module):
    def __init__(self, in_dim, out_dim, d=2):
        super().__init__()
        a = in_dim
        l = []
        while True:
            b = min(a * d, out_dim)
            l.extend([nn.Linear(a, b), nn.ReLU()])
            a = b
            if b == out_dim:
                del l[-1]
                break
        self.net = nn.Sequential(*l)

    def forward(self, x):
        return self.net(x)
