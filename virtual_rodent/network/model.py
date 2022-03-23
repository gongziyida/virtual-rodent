import torch
import torch.nn as nn

from virtual_rodent import VISUAL_DIM, PROPRI_DIM, ACTION_DIM
from .building_blocks import MLP

def fetch_reset_idx(done):
    reset_idx = []
    for i in range(len(done)):
        li = [-1]
        for j in range(len(done[i])):
            if done[i][j]:
                li.append(j) # Done state will not be passed to the model
        li.append(len(done[i]))
        reset_idx.append(li)
    return reset_idx

def iter_over_batch_with_reset(rnn, rnn_input, batch, reset_idx):
    h = []
    for i in range(batch):
        idx = reset_idx[i]
        li = []
        for j in range(len(idx) - 1):
            if idx[j] + 1 == idx[j+1]:
                continue
            # Note: LSTM hidden layers initiated to zero if not provided
            _, (h_ij, _) = rnn(rnn_input[idx[j]+1:idx[j+1], i:i+1])
            li.append(h_ij[-1:]) # Append the last layer only
        h.append(torch.cat(li, dim=0)) # Cat along temporal dim
    h = torch.cat(h, dim=1) # Cat along batch dim
    assert len(h.shape) == 3
    assert h.shape[0] == rnn_input.shape[0] and h.shape[1] == rnn_input.shape[1] 
    return h

class MerelModel(nn.Module):
    def __init__(self, visual_enc, propri_enc, core_hidden_dim,
                 sampling_dist=torch.distributions.Normal):
        super().__init__()

        self.visual_enc = visual_enc

        self.propri_enc = propri_enc

        with torch.no_grad():
            visual_emb_dim = self.visual_enc(torch.rand(1, *VISUAL_DIM)).squeeze().data.shape[0]
            propri_emb_dim = self.propri_enc(torch.rand(1, PROPRI_DIM)).squeeze().data.shape[0]

        self.core_in_dim = visual_emb_dim + propri_emb_dim
        self.core_hidden_dim = core_hidden_dim
        self.core = nn.LSTM(self.core_in_dim, self.core_hidden_dim, batch_first=False)

        self.value = MLP(in_dim=self.core_hidden_dim, out_dim=1, d=5)

        self.policy_in_dim = self.core_in_dim + self.core_hidden_dim + PROPRI_DIM
        self.policy = nn.LSTM(self.policy_in_dim, ACTION_DIM, num_layers=3, batch_first=False)

        self.sampling_dist = sampling_dist

    def forward(self, visual, propri, done=None):
        """ If the visual/propri are batched sequence, the shape is assumed to be
            (T, batch, channel, length, width,) and (T, batch, keypoints,); or
            (channel, length, width,) and (keypoints,)
            In the second case ignore done

            done: list or None
                If not None, assume shape: (batch)(T)
        """
        if len(visual.shape) == 3:
            assert len(propri.shape) == 1
            visual = visual.view(1, 1, *visual.shape)
            propri = propri.view(1, 1, *propri.shape)
            T, batch = 1, 1
        else:
            assert len(visual.shape) == 5 and len(propri.shape) == 3
            T, batch = visual.shape[:2]
            assert T == propri.shape[0] and batch == propri.shape[1]
        
        # Sensory encoding
        visual_ft = self.visual_enc(visual.view(-1, *visual.shape[-3:])).view(T, batch, -1)
        propri_ft = self.propri_enc(propri.view(-1, propri.shape[-1])).view(T, batch, -1)
        ft_concat = torch.cat((visual_ft, propri_ft), dim=-1)
        assert len(ft_concat.shape) == 3

        # RNNs
        if T == 1: # Do not need to concern about termination & reset
            _, (core_h, _) = self.core(ft_concat)
            policy_input = torch.cat((core_h.detach(), ft_concat, propri), dim=-1)
            _, (policy_h, _) = self.policy(policy_input)

        else: # Concern about termination & reset
            reset_idx = [[0, T] for _ in range(batch)] if done is None else fetch_reset_idx(done)
            core_h = iter_over_batch_with_reset(self.core, ft_concat, batch, reset_idx)
            policy_input = torch.cat((core_h.detach(), ft_concat, propri), dim=-1)
            policy_h = iter_over_batch_with_reset(self.policy, policy_input, batch, reset_idx)

        value = self.value(core_h)
        pi = self.sampling_dist(policy_h, torch.tensor(1).to(policy_h.device))
        return value, pi
