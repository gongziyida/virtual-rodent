import torch
import torch.nn as nn

def fetch_reset_idx(done, T, batch):
    reset_idx = []
    assert batch == done.shape[1], '%d, %d' % (batch, done.shape[1])
    for i in range(done.shape[1]):
        assert T == done.shape[0] - 1, '%d, %d' % (T, done.shape[0])  # Including state -1
        li = []
        for t in range(done.shape[0]): # Here t is actually state t-1
            if done[t, i]: # Done after action on state t-1
                li.append(t) # Note that state t should be included
        if T not in li: # Termination 
            li.append(T)
        assert len(li) >= 1
        reset_idx.append(li)
    return reset_idx

def iter_over_batch_with_reset(rnn, rnn_input, reset_idx, rnn_hc_init):
    out, hc = [], []
    if rnn_hc_init is not None: assert len(rnn_hc_init) == len(reset_idx)
    for i, idx in enumerate(reset_idx):
        li = []
        for j in range(len(idx)):
            # Note: LSTM hidden layers initiated to zero if not provided
            if j == 0 and idx[j] != 0:  
                if rnn_hc_init is None: # First run
                    rnn_out, rnn_hc = rnn(rnn_input[:idx[j], i:i+1])
                else: # Continue from last time
                    assert len(rnn_hc_init[i]) == 2
                    assert type(rnn_hc_init[i]) is tuple
                    rnn_out, rnn_hc = rnn(rnn_input[:idx[j], i:i+1], rnn_hc_init[i])
            elif j != len(idx) - 1: # reset
                rnn_out, rnn_hc = rnn(rnn_input[idx[j]:idx[j+1], i:i+1])
            else: # The last one is always T, do nothing
                assert len(idx) > 1 # Otherwise it should proceed to the first cond.
                break

            li.append(rnn_out)
        hc.append(rnn_hc) # Only store the last
        out.append(torch.cat(li, dim=0)) # Cat along temporal dim
    out = torch.cat(out, dim=1) # Cat along batch dim
    assert len(out.shape) == 3
    assert out.shape[0] == rnn_input.shape[0] and out.shape[1] == rnn_input.shape[1]
    return out, hc

