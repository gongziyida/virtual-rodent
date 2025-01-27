import torch
import torch.nn as nn

class MLPEnc(nn.Module):
    def __init__(self, propri_dim, emb_dim, hidden_dims=(300, 200), 
                 activation_func=nn.ReLU):
        super().__init__()
        dims = [propri_dim, *hidden_dims, emb_dim]
        li = []
        for i in range(len(dims)-2):
            li += [nn.Linear(dims[i], dims[i+1]), activation_func(inplace=False)]
        li.append(nn.Linear(dims[-2], dims[-1]))
        self.enc = nn.Sequential(*li, nn.LayerNorm(emb_dim))

    def forward(self, x):
        return self.enc(x) # Automatically handle reshape since it's linear
