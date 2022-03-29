import os, time
import copy
import torch
from torch.multiprocessing import Process

from virtual_rodent.environment import MAPPER
from virtual_rodent.utils import video

class Recorder(Process):
    def __init__(self, DEVICE_ID, model, env_name, save_dir,
                 simulator_params={}, save_full_record=False): 
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.device = torch.device('cuda:%d' % DEVICE_ID)
        self.save_dir = save_dir
        self.save_full_record = save_full_record

        # Simulator parameters
        self.simulator_params = simulator_params

        # Variables
        self.model = model.to(self.device) # Need disabling CPU binding

        # Environment name; instantiate when started
        self.env_name = env_name

    def run(self):
        PID = os.getpid()
        print('\n[%s] Setting env "%s" on %s' % (PID, self.env_name, self.device))
        if self.DEVICE_ID == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'
        else: # dm_control/mujoco maps onto EGL_DEVICE_ID
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['EGL_DEVICE_ID'] = str(self.DEVICE_ID) 
        self.env = MAPPER[self.env_name]()
        print('\n[%s] Simulating on env "%s" for recording' % (PID, self.env_name))
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import simulate
        elif str(os.environ['SIMULATOR_IMPALA']) == 'hop_simple':
            from virtual_rodent._test_simulation import simulate

        with torch.no_grad():
            ret = simulate(self.env, self.model, lambda ts, s: ts.last() or s > 60*60*5, self.device, 
                           **self.simulator_params)

        if self.simulator_params.get('ext_cam', False):
            ext_cam_id = self.simulator_params.get('ext_cam_id', (0,))
            ext_cam_size = self.simulator_params.get('ext_cam_size', (200, 200))
            for i in ext_cam_id:
                anim = video(ret['cam%d'%i])
                anim.save(os.path.join(self.save_dir, '%s_%s_cam%d.mp4' % (self.env_name, PID, i)))

        if self.save_full_record:
            torch.save(ret, os.path.join(self.save_dir, '%s_%s_records.pt' % (self.env_name, PID)))
