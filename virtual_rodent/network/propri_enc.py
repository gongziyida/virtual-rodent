import torch
import torch.nn as nn

class MLPEnc(nn.Module):
    def __init__(self, propri_dim, emb_dim, hidden_dims=(300, 200), 
                 activation_func=nn.LeakyReLU):
        super().__init__()
        dims = [propri_dim, *hidden_dims, emb_dim]
        for i in range(len(dims)-2):
            li += [nn.Linear(dims[i], dims[i+1]), activation_func()]
        li.append(nn.Linear(dims[-2], dims[-1]))
        self.enc = nn.Sequential(*li)

    def forward(self, x):
        dims = x.shape
        if len(dims) > 2: # Flatten batch dims and reshape back
            return self.enc(x.view(-1, dims[-1])).view(*dims[:-1], -1)
        elif len(dims) == 1: # Add batch dim and restore
            return self.enc(x.unsqueeze(0)).squeeze(0)
        else: # len == 2
            return self.enc(x)
