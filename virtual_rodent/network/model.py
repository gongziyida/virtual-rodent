import torch
import torch.nn as nn

from virtual_rodent import VISUAL_DIM, PROPRI_DIM, ACTION_DIM
from .building_blocks import MLP

def fetch_reset_idx(done, T, batch):
    if done is None:
        reset_idx = [[0, T] for _ in range(batch)] 
        return reset_idx 

    reset_idx = []
    assert batch == len(done)
    for i in range(len(done)):
        assert T == len(done[i])
        li = [0]
        for j in range(len(done[i])):
            if done[i][j]: # Done after action on state j
                li.append(j+1) # Note that state j should be included
        if len(done[i]) not in li: 
            li.append(len(done[i]))
        reset_idx.append(li)
    return reset_idx

def iter_over_batch_with_reset(rnn, rnn_input, reset_idx):
    out = []
    for i, idx in enumerate(reset_idx):
        li = []
        for j in range(len(idx) - 1):
            assert idx[j] != idx[j+1]
            # Note: LSTM hidden layers initiated to zero if not provided
            rnn_out, _ = rnn(rnn_input[idx[j]:idx[j+1], i:i+1])
            li.append(rnn_out)
        out.append(torch.cat(li, dim=0)) # Cat along temporal dim
    out = torch.cat(out, dim=1) # Cat along batch dim
    assert len(out.shape) == 3
    assert out.shape[0] == rnn_input.shape[0] and out.shape[1] == rnn_input.shape[1]
    return out

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
        """
        Parameters
        ----------
        visual/propri: torch.tensors
            If the visual/propri are batched sequence, the shape is assumed to be
            (T, batch, channel, length, width,) and (T, batch, keypoints,); or
            (channel, length, width,) and (keypoints,)
            In the second case ignore done and returns will be squeezed

            done: list or None
                If not None, assume shape: (batch)(T)
        returns
        -------
            value: torch.tensor
            pi: torch.Distribution
            reset_idx: nested list or None
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
            core_out, _ = self.core(ft_concat)
            policy_input = torch.cat((core_out.detach(), ft_concat, propri), dim=-1)
            policy_out, _ = self.policy(policy_input)
            reset_idx = None

        else: # Concern about termination & reset
            reset_idx = fetch_reset_idx(done, T, batch)
            core_out = iter_over_batch_with_reset(self.core, ft_concat, reset_idx)
            policy_input = torch.cat((core_out.detach(), ft_concat, propri), dim=-1)
            policy_out = iter_over_batch_with_reset(self.policy, policy_input, reset_idx)

        value = self.value(core_out)
        if T == 1 and batch == 1:
            value = value.squeeze()
            policy_out = policy_out.squeeze()
        pi = self.sampling_dist(policy_out, torch.tensor(1).to(policy_out.device))
        return value, pi, reset_idx
