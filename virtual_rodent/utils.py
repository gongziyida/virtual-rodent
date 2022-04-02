import time, random
import numpy as np
import pandas as pd
import torch

class Cache:
    def __init__(self, max_len):
        self._cache = []
        self._max_len = max_len

    def add(self, item):
        self._cache.append(item)
        if len(self._cache) > self._max_len:
            self._cache.pop(0)

    def sample(self, num):
        return random.sample(self._cache, num)

    def set_max_len(self, val):
        self._max_len = val

    def __len__(self):
        return len(self._cache)

def save_checkpoint(model, epoch, save_path, optimizer=None):
    d = {'model_state_dict': model.state_dict(), 'epoch': epoch}
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
    for k in stats.keys():
        if k in exclude:
            continue
        data = list(stats[k][stats[k] < np.inf])
        df['episode'] += list(range(len(data)))
        df['val'] += data
        name = key_alias.get(k, k)
        df['name'] += [name for _ in range(len(data))]

    df = pd.DataFrame(df)
    return df

