import time, random
import numpy as np
import pandas as pd
import torch


def save_checkpoint(model, epoch, save_path, optimizer=None): 
    state_dict = model.state_dict() if isinstance(model, torch.nn.Module) else model
    d = {'model_state_dict': state_dict, 'epoch': epoch}
    if optimizer is not None:
        d['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(d, save_path)

def load_checkpoint(model, load_path, optimizer=None):
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, epoch, optimizer
    return model, epoch

def stats_to_dataframe(stats, exclude=[], key_alias={}):
    df = dict(episode=[], val=[], name=[])
    for k, v in stats.items():
        if k in exclude:
            continue
        data = list(v)
        df['episode'] += list(range(len(data)))
        df['val'] += data
        name = key_alias.get(k, k)
        df['name'] += [name for _ in range(len(data))]

    df = pd.DataFrame(df)
    return df

