import os, pickle
import numpy as np
import pandas as pd

from virtual_rodent.utils import stats_to_dataframe
from virtual_rodent.visualization import *

paths = [
        #'out_mlp_hopper_hop_1E-04',
        #'out_mlp_hopper_stand_1E-04',
        #'out_rnn_cheetah_1E-04',
        'out_rnn_gaps_ouch_rodent_1E-04',
        ]

if __name__ == '__main__':
    for i in range(len(paths)):
        exclude = ['policy_weight', 'value_weight', 'entropy_weight']
        try:
            with open(os.path.join(paths[i], 'training_stats.pkl'), 'rb') as f:
                d = pickle.load(f)
            if len(d['learning_rate']) == 0:
                exclude.append('learning_rate')
            df = stats_to_dataframe(d, exclude=exclude)

            plot_stats(df, os.path.join(paths[i], 'training_stats.png'), plot=plot_smooth_training_curve)

            with open(os.path.join(paths[i], 'rewards.pkl'), 'rb') as f:
                rewards = pickle.load(f)

            plot_rewards_dict(rewards, os.path.join(paths[i], 'rewards.png'))
            # rewards = np.load(os.path.join(paths[i], 'rewards.npy'))
            # plot_rewards_numpy(rewards, os.path.join(paths[i], 'rewards.png'))

        except FileNotFoundError as e:
            print(e)
            continue
