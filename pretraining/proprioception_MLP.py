import os
from tqdm import tqdm
from tqdm.contrib.itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from dm_control.locomotion.examples import basic_rodent_2020

from virtual_rodent.utils import save_checkpoint, load_checkpoint
from virtual_rodent.network import VAE, MLP, MLPMirror
from virtual_rodent.simulation import get_proprioception

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _simulate():
    propri_states = []
    env = basic_rodent_2020.rodent_two_touch() # Build an example environment
    action_spec = env.action_spec() # Get the `action_spec` describing the control inputs

    interval_spec = (1, 5, 10)
    p_spec = tuple(np.linspace(0, 1, num=11, endpoint=True))
    var_spec = (0.01, 0.05, 0.1, 0.2)
    for interval, p, var in product(interval_spec, p_spec, var_spec):
        # Step through the environment for one episode with random actions.
        time_step = env.reset()

        counter = 0
        action = np.zeros(action_spec.shape)
        while not time_step.last():
            if counter > 50 and counter % interval == 0:
                action += np.random.normal(0, var, size=action_spec.shape) * \
                          np.random.binomial(1, p, size=action_spec.shape)
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            time_step = env.step(action)
            propri_states.append(get_proprioception(time_step.observation))
            counter += 1
    return torch.from_numpy(np.stack(propri_states, axis=0))


def get_proprioception_data(path='../data/', fname='propriception.pt', device=_device):
    if os.path.exists(os.path.join(path, fname)):
        data = torch.load(os.path.join(path, fname), map_location=device).to(dtype=torch.float32)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        data = _simulate().to(device, dtype=torch.float32)
        print('Data Size', data.shape)
        torch.save(data, os.path.join(path, fname))

    train_size, test_size = int(data.shape[0] * 0.8), int(data.shape[0] * 0.2)
    data = random_split(data, [train_size, test_size])
    return TensorDataset(data[0].dataset), TensorDataset(data[1].dataset)


def get_proprioception_encoder(path, propri_dim, propri_emb_dim):
    model = VAE(enc=MLP(propri_dim, propri_emb_dim), dec=MLPMirror(propri_emb_dim, propri_dim))
    model, _ = load_checkpoint(model, path)
    return model.enc

if __name__ == '__main__':
    batch_size = 128
    epochs = 100
    train_set, test_set = get_proprioception_data()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    propri_dim = train_set.tensors[0].shape[1] # 107
    propri_emb_dim = 16

    model = VAE(enc=MLP(propri_dim, propri_emb_dim), dec=MLPMirror(propri_emb_dim//2, propri_dim))
    model = model.to(_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss(reduction='sum')

    path = 'proprioception/'
    if not os.path.exists(path):
        os.makedirs(path)

    loss_buff = np.zeros((2, epochs))

    for epoch in tqdm(range(epochs)):
        model.train()
        for (data,) in train_loader:
            optimizer.zero_grad()
            rec, kl = model(data)
            loss = mse(data, rec) + kl.sum()
            loss.backward()
            optimizer.step()
            loss_buff[0, epoch] = loss.item()

        model.eval()
        with torch.no_grad():
            for (data,) in train_loader:
                rec, kl = model(data)
                loss = mse(data, rec) + kl.sum()
                loss_buff[1, epoch] = loss.item()
        
        print(loss_buff[:, epoch])

        if (epoch+1) % 20 == 0 or epoch+1 == epochs:
            save_checkpoint(model, epoch, os.path.join(path, 'model_%d.pt' % (epoch+1)))
    np.save(os.path.join(path, 'loss.npy'), loss_buff)

