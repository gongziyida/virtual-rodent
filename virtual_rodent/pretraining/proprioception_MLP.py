import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dm_control.locomotion.examples import basic_rodent_2020

from virtual_rodent.utils import get_proprioception
from virtual_rodent.network import VAE, MLP, MLPMirror

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _simulate():
    propri_states = []
    env = basic_rodent_2020.rodent_two_touch() # Build an example environment
    action_spec = env.action_spec() # Get the `action_spec` describing the control inputs

    for interval in (1, 5, 10):
        for p in np.linspace(0, 1, num=11, endpoint=True):
            for var in (0.01, 0.05, 0.1, 0.2):
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
    return torch.from_numpy(np.stack(propri_state, axis=0))


def get_proprioception_data(path='../data/propriception.pt', device=_device):
    if os.path.exists(path):
        data = torch.load(path, map_location=device)
    else:
        data = _simulation().to(device)
        torch.save(data, path)

    train_size, test_size = int(data.shape[0] * 0.8), int(data.shape[0] * 0.2)
    data = random_split(data, [train_size, test_size])
    return TensorDataset(data[0]), TensorDataset(data[1])


def get_proprioception_encoder(path, propri_dim, propri_emb_dim):
    model = VAE(enc=MLP(propri_dim, propri_emb_dim), dec=MLPMirror(propri_emb_dim, propri_dim))
    model, _ = load_checkpoint(model, path)
    return model.enc


if __name__ == '__main__':
    batch_size = 128
    epochs = 50
    train_set, test_set = get_proprioception_data()
    train_loader = DataLoader(train_set, batch_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_batch_size, shuffle=True)

    propri_dim = train_set.tensors[0].shape[1]
    propri_emb_dim = 16

    model = VAE(enc=MLP(propri_dim, propri_emb_dim), dec=MLPMirror(propri_emb_dim, propri_dim))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    path = 'proprioception/'
    writer = SummaryWriter(path)

    for epoch in tqdm(range(epochs)):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            results = model(data)
            loss = loss_func(data, results)
            loss.backward()
            optimizer.step()
            writer.add_scalars('train/loss', {'loss': loss.item()}, epoch)

        model.eval()
        with torch.no_grad():
            for data in train_loader:
                results = model(data)
                loss = loss_func(data, results)
                writer.add_scalars('test/loss', {'loss': loss.item()}, epoch)

        if (epoch+1) % 10 == 0:
            save_checkpoint(vae, epoch, os.path.join(path, 'model_%d.pt' % (epoch+1)))

    writer.flush()
    writer.close()
