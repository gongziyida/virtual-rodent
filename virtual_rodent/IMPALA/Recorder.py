import os, time, pickle
import torch

from torch.multiprocessing import Process
from virtual_rodent.visualization import video
from virtual_rodent.utils import save_checkpoint
from .base import WorkerBase

class StatsRecorder(Process):
    def __init__(self, state_dict, queue, exit, save_dir):
        super().__init__()
        self.queue = queue

        self.training_stats_keys = ('total_loss', 'mean_vtrace', 'mean_value', 
                                    'actor_loss', 'critic_loss', 'entropy', 'learning_rate')
        self.training_stats = {k: [] for k in self.training_stats_keys}

        self.reward_stats = dict()

        self.state_dict = state_dict

        self.exit, self.exit_value = exit
        self.save_dir = save_dir

    def record_training(self, **kwargs):
        for k, v in kwargs.items():
            self.training_stats[k] += list(v)

    def record_reward(self, env_name, reward):
        try:
            self.reward_stats[env_name].append(reward)
        except KeyError:
            self.reward_stats[env_name] = [reward]

    def save(self):
        save_checkpoint(dict(self.state_dict), None, os.path.join(self.save_dir, 'model.pt'))
        # Update the stats periodically
        with open(os.path.join(self.save_dir, 'training_stats.pkl'), 'wb') as f:
            pickle.dump(self.training_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.save_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump(self.reward_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        last_saved = time.time()
        while self.exit.value != self.exit_value:
            time.sleep(30)
            while not self.queue.empty():
                x = self.queue.get()
                if type(x) == tuple:
                    self.record_reward(*x)
                elif type(x) == dict:
                    self.record_training(**x)
            
            if time.time() - last_saved > 3 * 60:
                self.save()
                last_saved = time.time()
        self.save()


class VideoRecorder(WorkerBase):
    def __init__(self, ID, DEVICE_INFO, model, env_name, save_dir,
                 simulator_params={}, save_full_record=False): 
        super().__init__(ID, DEVICE_INFO, model, env_name)
        # Constants
        self.save_dir = save_dir
        self.save_full_record = save_full_record

        # Simulator parameters
        self.simulator_params = simulator_params

    def run(self):
        self.setup()
        print('\n[%s] Simulating on env "%s" for recording' % (self.PID, self.env_name))
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import simulate
        else:
            from virtual_rodent._test_simulation import simulate

        with torch.no_grad():
            ret = simulate(self.env, self.model, self.propri_attr, max_step=5000, device=self.device, 
                           **self.simulator_params)

        if self.simulator_params.get('ext_cam', True):
            ext_cam = self.simulator_params.get('ext_cam', (0,))
            ext_cam_size = self.simulator_params.get('ext_cam_size', (200, 200))
            for i in ext_cam:
                anim = video(ret['cam%d'%i])
                fname = '%s_%s_cam%d.mp4' % (self.env_name, self.PID, i)
                anim.save(os.path.join(self.save_dir, fname))

        if self.save_full_record:
            fname = '%s_%s_records.pt' % (self.env_name, self.PID)
            torch.save(ret, os.path.join(self.save_dir, fname))
