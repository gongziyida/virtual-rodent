import os, time
import copy
import torch
from torch.multiprocessing import Process

from virtual_rodent.environment import MAPPER
from virtual_rodent.visualization import video

class Recorder(Process):
    def __init__(self, DEVICE_ID, model, env_name, save_dir,
                 simulator_params={}, save_full_record=False): 
        super().__init__()
        # Constants
        self.DEVICE_ID = DEVICE_ID
        self.device = 'cpu' if DEVICE_ID == 'cpu' else 'cuda:%d' % DEVICE_ID
        self.device = torch.device(self.device)
        self.save_dir = save_dir
        self.save_full_record = save_full_record

        # Simulator parameters
        self.simulator_params = simulator_params

        # Variables
        self.model = model.to(self.device) # Need disabling CPU binding

        # Environment name; instantiate when started
        self.env_name = env_name

    def set_env(self):
        self.PID = os.getpid()
        
        if self.DEVICE_ID == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'
        else:
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['EGL_DEVICE_ID'] = str(self.DEVICE_ID) 

        print('\n[%s] Setting env "%s" on %s with %s' % \
                (self.PID, self.env_name, self.device, os.environ['MUJOCO_GL']))
        self.env, self.propri_attr = MAPPER[self.env_name]()
        print('\n[%s] Simulating on env "%s" for recording' % (self.PID, self.env_name))


    def run(self):
        self.set_env()
        if str(os.environ['SIMULATOR_IMPALA']) == 'rodent':
            from virtual_rodent.simulation import simulate
        else:
            print('testing')
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
