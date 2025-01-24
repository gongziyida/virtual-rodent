import os, socket, time
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8818'

import torch
import torch.distributed.rpc as rpc
from torch.multiprocessing import Lock
from virtual_rodent.utils import save_checkpoint
from .base import SampleQueue

import importlib.util
spec = importlib.util.spec_from_file_location('_', os.environ['model_init_method_path'])
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
model_init_method = script.model_init_method

target_model, behavior_model = None, None
lock_target, lock_behavior = Lock(), Lock()

sample_queue = None

training_stats_keys = ('total_loss', 'mean_vtrace', 'mean_value', 
                       'actor_loss', 'critic_loss', 'entropy', 'learning_rate')
training_stats = {k: [] for k in training_stats_keys}
reward_stats = dict()
lock_training_stats, lock_reward_stats = Lock(), Lock()

def store_batch(worker_id, env_name, batch):
    sample_queue.put(worker_id - 1, batch)
    reward = batch['reward'].detach().cpu().sum().item()
    with lock_reward_stats:
        try:
            reward_stats[env_name].append(reward)
        except KeyError:
            reward_stats[env_name] = [reward]

def fetch_batch(batch_size):
    while len(sample_queue) < batch_size: # Get enough samples first
            time.sleep(0.1)
    return sample_queue.sample(batch_size)

def record_loss(loss_dict):
    with lock_training_stats:
        for k, v in loss_dict.items():
            training_stats[k] += list(v)

def get_target_model():
    global target_model
    with lock_target:
        if target_model is None:            
            target_model = model_init_method()
    return target_model

def save(save_dir):
    save_checkpoint(target_model.state_dict(), None, os.path.join(save_dir, 'model.pt'))
    with open(os.path.join(save_dir, 'training_stats.pkl'), 'wb') as f:
        pickle.dump(training_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(reward_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_behavior_model():
    global behavior_model
    with lock_behavior:
        if behavior_model is None:
            behavior_model = model_init_method()
    return behavior_model

def param_rref(model):
    return [rpc.RRef(param) for param in model.parameters()]

def forward(model, *args, **kwargs):
    return model(*args, **kwargs)

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def run_parameter_server(rank, world_size, exit, exit_value, save_dir, n_workers, 
                         max_step, vision_dim, propri_dim, action_dim):
    os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
    os.environ['MASTER_PORT'] = '8818'
    print('ps', os.environ['MASTER_ADDR'])
    rpc.init_rpc(name='ParameterServer', rank=rank, world_size=world_size)
    print('ps', os.environ['MASTER_ADDR'])
    global sample_queue
    sample_queue = SampleQueue(1000, n_workers, max_step,
                               vision_dim, propri_dim, action_dim) 
    print("RPC initialized! Running parameter server...")
    while exit.value != exit_value:
        time.sleep(20)
        if target_model is not None and behavior_model is not None:
            behavior_model.load_state_dict(target_model.state_dict())
            save(save_dir)
    rpc.shutdown()
    print("RPC shutdown on parameter server.")
